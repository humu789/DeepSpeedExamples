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
        "--per_device_train_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
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
    parser.add_argument("--save_best_model", action="store_true",  help="save best checkpoint model")
    parser.add_argument("--clean_best_model", action="store_true", help="clean the  model")
    parser.add_argument("--lkd_enabled", action="store_true", help="using lkd or not")
    parser.add_argument("--distill_method", type=str, default=None, help="knowledage distillation")   
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="local-rank for distributed training on gpus")
    parser.add_argument(
        "--model_name_or_path_teacher",
        default=None,
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--pretrained_dir_student",
        type=str,
        default=None,
        help="Where to load the student pretrained model.")
    parser.add_argument(
        "--pretrained_dir_teacher",
        type=str,
        default=None,
        help="Where to load the teacher pretrained model.")
    parser.add_argument(
        "--eval_step",
        type=int,
        default=1000,
        help="when to eval the model.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print_rank_0 = print_rank(args)
    ds_config = None
    if args.deepspeed:
        with open(args.deepspeed_config) as f:
            ds_config = json.load(f)
        layer_reduction_enabled, prune_enabled, quantization_enabled = check_and_identify_compresssion(args, ds_config)
        args.layer_reduction_enabled = layer_reduction_enabled
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.lkd_enabled:
        assert args.distill_method != "zero_stage", "zero_stage is not supported for lkd since we need the teacher model"

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
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.distributed.barrier()

    # get the dataset.
    from datautils import get_loaders
    nsamples = args.max_train_steps * args.per_device_train_batch_size
    train_dataset, testenc = get_loaders(
        args.dataset_name,
        nsamples=nsamples,
        model=args.model_name_or_path,
        seqlen=2048)
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size
    )

    # Load pretrained model and tokenizer
    from util import get_llama
    model = get_llama(args.model_name_or_path)
    model.to(device)

    ppl = evaluation(model, 
                     testenc, 
                     device,
                     batch_size=args.per_device_eval_batch_size,
                     is_engine=False)
    print_rank_0(f"before compression, the float model's performance: {ppl}")

    teacher_model  = None
    #### load teacher models
    if args.distill_method != 'zero_stage':
        if not args.model_name_or_path_teacher:
            args.model_name_or_path_teacher = args.model_name_or_path
        teacher_model = get_llama(args.model_name_or_path_teacher)
        teacher_model.to(device)
        if args.pretrained_dir_teacher is not None:
            teacher_model.load_state_dict(
                torch.load(args.pretrained_dir_teacher))
            
    # model inititalization, config,
    if args.deepspeed:
        if quantization_enabled or prune_enabled or layer_reduction_enabled:
            model = init_compression(model, args.deepspeed_config, teacher_model=teacher_model)  #<==========================================compression argument

    if args.pretrained_dir_student is not None:
            model.load_state_dict(torch.load(args.pretrained_dir_student))  #<==========================================add weight to students if users provides difference models

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, _, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dist_init_required=True)
    
    ppl = evaluation(model, 
                     testenc, 
                     device,
                     batch_size=args.per_device_eval_batch_size,
                     is_engine=True)
    print_rank_0(f"at step 0 (without LKD) the (student) model's performance: {ppl}")
    
    model.eval()
    teacher_model.eval()
    start_time = time.time()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps =  math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    
    for l in range(model.module.config.num_hidden_layers):
        print_rank_0(f"layer {l}")
        student_layer = recursive_getattr(model.module.model, f'layers.{l}')  # extract the lth layer of student
        
        # import pdb;pdb.set_trace()
        
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

        optimizer = AdamW(optimizer_param, lr=args.learning_rate) 

        for batch in train_dataloader:  # load each batch
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
        
        # import pdb;pdb.set_trace()
    
    del teacher_model

    ppl = evaluation(model, 
                     testenc, 
                     device,
                     batch_size=args.per_device_eval_batch_size,
                     is_engine=True)
    print_rank_0(f"After {time.time() - start_time}s, (with LKD) the (student) model's performance: {ppl}")
    

if __name__ == "__main__":
    main()
