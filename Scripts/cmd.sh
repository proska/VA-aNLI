#!/usr/bin/env bash

python token_vat.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --learning_rate 3e-5 \
  --do_train --task_name mnli \
  --data_dir data/TTANLI \
  --output_dir outputs/anli_tavat \
  --overwrite_output_dir \
  --max_seq_length 256 \
  --save_steps 5000 \
  --logging_steps 5000 \
  --evaluate_during_training \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --warmup_steps 5000 \
  --num_train_epochs 9 \
  --adv_lr 1e-1 \
  --adv_init_mag 3e-1 \
  --adv_max_norm 4e-1 \
  --adv_steps 2 \
  --vocab_size 30522 \
  --hidden_size 768 \
  --adv_train 1 \
  --gradient_accumulation_steps 1