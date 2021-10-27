#!/usr/bin/env bash

cd ../

CUDA_VISIBLE_DEVICES=2 python token_vat.py \
  --model_type bert \
  --model_name_or_path microsoft/deberta-v2-xlarge-mnli \
  --learning_rate 3e-5 \
  --do_train \
  --do_eval \
  --overwrite_cache \
  --task_name mnli \
  --data_dir Output/data/TTANLI/ \
  --output_dir Output/anli_deberta_tavat \
  --overwrite_output_dir \
  --max_seq_length 512 \
  --save_steps 5000 \
  --logging_steps 5000 \
  --per_gpu_train_batch_size 2 \
  --per_gpu_eval_batch_size 2 \
  --warmup_steps 5000 \
  --num_train_epochs 9 \
  --adv_lr 1e-1 \
  --adv_init_mag 3e-1 \
  --adv_max_norm 4e-1 \
  --adv_steps 2 \
  --vocab_size 30522 \
  --hidden_size 1536 \
  --adv_train 1 \
  --gradient_accumulation_steps 1 \
  --evaluate_during_training




