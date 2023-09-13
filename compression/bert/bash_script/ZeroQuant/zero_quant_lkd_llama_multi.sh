#!/bin/bash

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%. run_jobs.sh (for mnli)
# "quantize_groups": 128
export CUDA_VISIBLE_DEVICES=4,5,6,7
BATCH_SIZE_TRAIN=8
TRAIN_STEPS=$((200 * 8 / BATCH_SIZE_TRAIN))
CONFIG=./config/ZeroQuant/ds_config_zeroquant-lkd_custom.json
SAVE_PATH=./out/ZeroQuant/W4A8_quantization_llama_custom
mkdir -p ${SAVE_PATH}
OMP_NUM_THREADS=16 python -m torch.distributed.launch --nproc_per_node=4 \
  --master_port 10004 \
  run_lkd_llama_multi.py \
  --dataset_name wikitext2 \
  --seed 42 \
  --model_name_or_path /nvme/share_data/llama_ckpts/huggingface/7B \
  --train_batch_size ${BATCH_SIZE_TRAIN} \
  --eval_batch_size 8 \
  --deepspeed_config ${CONFIG} \
  --deepspeed \
  --max_train_steps ${TRAIN_STEPS} \
  --num_warmup_steps 0 \
  --learning_rate 5e-7 \
  --gradient_accumulation_steps 1 \
  --output_dir ${SAVE_PATH} #&>> ${SAVE_PATH}/lkd_llama_w4-sys-g128_a8-asys_step800_lr5e-7_skip2_31.log

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% users provide models  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MODEL_BASE=/blob/users/xwu/compression/huggingface_models/bert_base_uncased ## or you could use bert-base-uncased
# TEACHER=/blob/users/xwu/compression/huggingface_models/bert-base-uncased-${TASK_NAME}/pytorch_model.bin
# STUDENT=${TEACHER}
# python -m torch.distributed.launch --nproc_per_node=1 \
#   --master_port 66667 \
#   run_glue_no_trainer_clean.py \
#   --seed 42 \
#   --distill_method ${STAGE} \
#   --model_name_or_path ${MODEL_BASE} \
#   --pretrained_dir_student ${STUDENT} \
#   --pretrained_dir_teacher ${TEACHER} \
#   --task_name $TASK_NAME \
#   --max_length 128 \
#   --pad_to_max_length \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 18 \
#   --num_warmup_epochs 1 \
#   --deepspeed_config ${CONFIG} --weight_bit 1 \
#   --deepspeed \
#   --save_best_model --clean_best_model \
#   --gradient_accumulation_steps 1 \
#   --output_dir ${SAVE_PATH} &>> ${SAVE_PATH}/train.log
