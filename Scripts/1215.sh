#!/usr/bin/env bash
cd ..
#for PAIRNUM in 2 1 4; do
for PAIRNUM in 1; do
  mkdir -p /nas/home/$(whoami)/VA-aNLI/Output/RegulTokenPair/$PAIRNUM/
  cd /nas/home/$(whoami)/VA-aNLI/Output/RegulTokenPair/$PAIRNUM/ || break
  CUDA_VISIBLE_DEVICES=2 python /nas/home/$(whoami)/VA-aNLI/TAVAT/token_vat.py \
    --select_pairs True \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --learning_rate 3e-5 \
    --do_train \
    --task_name mnli \
    --output_dir /nas/home/$(whoami)/VA-aNLI/Output/RegulTokenPair/$PAIRNUM/ \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --save_steps 5000 \
    --logging_steps 5000 \
    --evaluate_during_training \
    --per_gpu_train_batch_size 32 \
    --warmup_steps 5000 \
    --num_train_epochs 9 \
    --adv_lr 1e-1 \
    --adv_init_mag 3e-1 \
    --adv_max_norm 4e-1 \
    --adv_steps 2 \
    --vocab_size 30522 \
    --hidden_size 768 \
    --adv_train 1 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 32 \
    --k $PAIRNUM \
    --data_dir /nas/home/$(whoami)/VA-aNLI/Output/data/MNLI
#    --data_dir /nas/home/$(whoami)/VA-aNLI/Output/data/MNLI/toy

  cd - || break
done
#python token_vat.py --select_pairs True --model_type bert --model_name_or_path bert-base-uncased --learning_rate 3e-5 --do_train --task_name mnli --data_dir data/MNLI --output_dir outputs/mnli_tavat --overwrite_output_dir --max_seq_length 512 --save_steps 5000 --logging_steps 5000 --evaluate_during_training --per_gpu_train_batch_size 32 --warmup_steps 5000 --num_train_epochs 9 --adv_lr 1e-1 --adv_init_mag 3e-1 --adv_max_norm 4e-1 --adv_steps 2 --vocab_size 30522 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1 --per_device_eval_batch_size 32 --k 2
#python token_vat.py --select_pairs True --model_type bert --model_name_or_path bert-base-uncased --learning_rate 3e-5 --do_train --task_name mnli --data_dir data/MNLI --output_dir outputs/mnli_tavat --overwrite_output_dir --max_seq_length 512 --save_steps 5000 --logging_steps 5000 --evaluate_during_training --per_gpu_train_batch_size 32 --warmup_steps 5000 --num_train_epochs 9 --adv_lr 1e-1 --adv_init_mag 3e-1 --adv_max_norm 4e-1 --adv_steps 2 --vocab_size 30522 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1 --per_device_eval_batch_size 32 --k 4
#python token_vat.py --select_pairs True --model_type bert --model_name_or_path bert-base-uncased --learning_rate 3e-5 --do_train --task_name mnli --data_dir data/MNLI --output_dir outputs/mnli_tavat --overwrite_output_dir --max_seq_length 512 --save_steps 5000 --logging_steps 5000 --evaluate_during_training --per_gpu_train_batch_size 32 --warmup_steps 5000 --num_train_epochs 9 --adv_lr 1e-1 --adv_init_mag 3e-1 --adv_max_norm 4e-1 --adv_steps 2 --vocab_size 30522 --hidden_size 768 --adv_train 1 --gradient_accumulation_steps 1 --per_device_eval_batch_size 32 --k 1
