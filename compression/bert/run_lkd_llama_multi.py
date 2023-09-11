# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import time
import os
import random
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import datasets
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

import transformers
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
from huggingface_transformer.modeling_bert import BertForSequenceClassification
import deepspeed
from deepspeed.compression.compress import init_compression, redundancy_clean
from deepspeed.compression.helper import recursive_getattr
from util import *
logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='wikitext',
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_epochs", type=float, default=0, help="Number of epochs for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of epochs for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
 

    #############deepspeed, compression, and knowledage distillation#########
    parser.add_argument("--deepspeed", action="store_true", help="use deepspeed or not")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="deepspeed config")   
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="local-rank for distributed training on gpus")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print_rank_0 = print_rank(args)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.ERROR)
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.distributed.barrier()

    # Load pretrained model and tokenizer
    from util import get_llama
    model = get_llama(args.model_name_or_path)
    teacher_model = copy.copy(model)
    student_model = init_compression(model, args.deepspeed_config)  #<==========================================compression argument
    teacher_model.to(device)
    print_rank_0(f'################# init_compression finished ################')

    # get the dataset.
    from datautils import get_loaders
    nsamples = args.max_train_steps * args.train_batch_size
    train_dataset, testenc = get_loaders(
        args.dataset_name,
        nsamples=nsamples,
        model=args.model_name_or_path,
        seqlen=model.seqlen)
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size
    )
    
    print_rank_0(f'################# train_dataloader is ready! ################')

    start_time = time.time()
    
    skip_indexs = [2, 31]
    layer_start = 0
    for l in range(layer_start, student_model.config.num_hidden_layers):
        if l in skip_indexs:
            print_rank_0(f'skip layer {l}')
        else:
            print_rank_0(f"KD layer {l}")
            student_layer = recursive_getattr(student_model.model, f'layers.{l}')  # extract the lth layer of student
            student_layer.to(device)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_param = [
            {
                "params": [p for n, p in student_layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in student_layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            ]

            optimizer = torch.optim.AdamW(optimizer_param, lr=args.learning_rate)

            student_layer, optimizer, _, _ = deepspeed.initialize(
                args=None,
                model=student_layer,
                optimizer=optimizer,
                config=args.deepspeed_config)

            for i, batch in enumerate(train_dataloader):  # load each batch
                batch = to_device(batch, device)
                with torch.no_grad():
                    # for simplicity, we always run the full inference of the teacher model.
                    # To get the best performance, you can run the teacher model only for the first l layers,
                    # which requires some modifications to the modeling code.
                    teacher_out = teacher_model(**batch, output_hidden_states=True) # get the output of the teacher model
                layer_input = teacher_out.hidden_states[l] # extract the lth-layer's input of teacher
                teacher_o = teacher_out.hidden_states[l+1] # extract the lth-layer's output of teacher
                
                prepared_attention_mask = teacher_model.model.prepared_attention_mask
                student_o = student_layer(layer_input, prepared_attention_mask)[0] # run inference for the student

                loss = torch.nn.functional.mse_loss(student_o, teacher_o)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print_rank_0(f'step{i} loss: {loss.item()}')
            
            del student_layer
            
            student_model.to(device)
            ppl = evaluation(student_model, 
                            testenc, 
                            device,
                            batch_size=args.eval_batch_size,
                            is_engine=False)
            student_model.cpu()
            print_rank_0(f"layer {l} is finished, the (student) model's performance: {ppl}")
    
    del teacher_model

    student_model.to(device)
    ppl = evaluation(student_model, 
                     testenc, 
                     device,
                     batch_size=args.eval_batch_size,
                     is_engine=False)
    print_rank_0(f"After {time.time() - start_time}s, (with LKD) the (student) model's performance: {ppl}")
    

if __name__ == "__main__":
    main()
