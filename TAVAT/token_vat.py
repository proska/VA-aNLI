# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
# 30522 bert
# 50265 roberta
# 21128 bert-chinese

# Todo 加上cls和sep
# Todo 去除重复
# Todo 加constraint
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert,
XLM-RoBERTa). """
import argparse
import glob
import json
import logging
import os
import sys

import hydra
import omegaconf
import IPython
from transformers import AutoModel
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    BertModel
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:  # multiple gpu
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    ##
    # initialize embedding delta 初始化preturbation

    delta_global_embedding = torch.zeros([args.vocab_size, args.hidden_size]).uniform_(-1, 1)

    # 30522 bert
    # 50265 roberta
    # 21128 bert-chinese

    dims = torch.tensor([args.hidden_size]).float()  # (768^(1/2))
    mag = args.adv_init_mag / torch.sqrt(dims)  # 1 const (small const to init delta)
    delta_global_embedding = (delta_global_embedding * mag.view(1, 1))
    delta_global_embedding = delta_global_embedding.to(args.device)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], ncols=80)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # adaptive seq len
            max_seq_len = torch.max(torch.sum(batch[1], 1)).item()
            if args.select_pairs == True:
                batch = [t[:, :max_seq_len] for t in batch[:5]] + [batch[5]]
            else:
                batch = [t[:, :max_seq_len] for t in batch[:3]] + [batch[3]]

            # BERT -only
            if args.select_pairs:
                inputs = {"attention_mask": batch[1], "token_type_ids": batch[2], "pairs": batch[3],
                          "pair_index": batch[4], "labels": batch[5]}
            else:
                inputs = {"attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            # Adv-Train

            # initialize delta
            input_ids = batch[0]

            input_ids_flat = input_ids.contiguous().view(-1)
            if isinstance(model, torch.nn.DataParallel):
                embeds_init = model.module.base_model.embeddings.word_embeddings(input_ids)
            else:
                embeds_init = model.base_model.embeddings.word_embeddings(input_ids)
            # embeds_init = embeds_init.clone().detach()
            input_mask = inputs['attention_mask'].float()
            input_lengths = torch.sum(input_mask, 1)  # B

            bs, seq_len = embeds_init.size(0), embeds_init.size(1)

            # here we calc idmask
            # deltamask=torch.zeros_like(input_mask)
            # where_is_sepx,where_is_sepy=torch.where(input_ids==(tokenizer('[SEP]')['input_ids'][1]))
            # for a in range(bs):
            #     deltamask[a][0]=1
            #     deltamask[a][1]=1
            #     deltamask[a][where_is_sepy[a*2]]=1
            #     deltamask[a][where_is_sepy[a*2]+1]=1
            #     deltamask[a][where_is_sepy[a*2+1]]=1
            if args.select_pairs:
                pair_index = inputs["pair_index"]
                deltamask = inputs["pairs"].float()
                input_ids_flat = (input_ids * deltamask).contiguous().view(-1).long()
            #
            delta_lb, delta_tok, total_delta = None, None, None

            dims = input_lengths * embeds_init.size(-1)  # B x(768^(1/2))
            mag = args.adv_init_mag / torch.sqrt(dims)  # B
            if args.select_pairs:
                sum = deltamask.sum()
                delta_lb = torch.zeros_like(embeds_init).uniform_(-1, 1) * deltamask.unsqueeze(2)
            else:
                delta_lb = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
            delta_lb = (delta_lb * mag.view(-1, 1, 1)).detach()

            gathered = torch.index_select(delta_global_embedding, 0, input_ids_flat)  # B*seq-len D

            if args.select_pairs:
                delta_tok = (gathered.view(bs, seq_len, -1) * deltamask.unsqueeze(2)).detach()  # B seq-len D
            else:
                delta_tok = gathered.view(bs, seq_len, -1).detach()
            denorm = torch.norm(delta_tok.view(-1, delta_tok.size(-1))).view(-1, 1, 1)
            delta_tok = delta_tok / denorm  # B seq-len D  normalize delta obtained from global embedding
            # B seq-len 1
            if args.adv_train == 0:
                # inputs['inputs_embeds'] = embeds_init
                inputs['input_ids'] = input_ids
                outputs = model(**inputs)
                loss = outputs[0]
                # 1) loss backward

                if args.n_to > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()
                loss.backward()

            else:

                # Adversarial-Training Loop
                for astep in range(args.adv_steps):

                    # craft input embedding
                    delta_lb.requires_grad_()
                    delta_tok.requires_grad_()

                    inputs_embeds = embeds_init + delta_lb + delta_tok

                    inputs['inputs_embeds'] = inputs_embeds
                    if "pairs" in inputs:
                        del inputs["pairs"]
                    if "pair_index" in inputs:
                        del inputs["pair_index"]
                    outputs = model(**inputs)
                    loss = outputs[0]

                    # 1) loss backward

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss_b = 0
                    for i in range(len(delta_lb)):
                        for j in range(args.k):
                            pair1 = pair_index[i][j][0]
                            pair2 = pair_index[i][j][1]
                            loss_b += abs(delta_lb[i][pair1] - delta_lb[i][pair2])
                    tr_loss += loss.item() + loss_b.sum()
                    loss = loss + args.lamda * loss_b.mean()
                    loss.backward(retain_graph=True)

                    if astep == args.adv_steps - 1:
                        # further updates on delta

                        delta_tok = delta_tok.detach()
                        delta_global_embedding = delta_global_embedding.index_put_((input_ids_flat,), delta_tok, True)

                        break

                    # 2) get grad on delta
                    if delta_lb is not None:
                        delta_lb_grad = delta_lb.grad.clone().detach()
                    if delta_tok is not None:
                        delta_tok_grad = delta_tok.grad.clone().detach()
                    # 3) update and clip

                    denorm_lb = torch.norm(delta_lb_grad.view(bs, -1), dim=1).view(-1, 1, 1)
                    denorm_lb = torch.clamp(denorm_lb, min=1e-8)
                    denorm_lb = denorm_lb.view(bs, 1, 1)

                    denorm_tok = torch.norm(delta_tok_grad, dim=-1)  # B seq-len
                    denorm_tok = torch.clamp(denorm_tok, min=1e-8)
                    denorm_tok = denorm_tok.view(bs, seq_len, 1)  # B seq-len 1

                    delta_lb = (delta_lb + args.adv_lr * delta_lb_grad / denorm_lb).detach()
                    delta_tok = (delta_tok + args.adv_lr * delta_tok_grad / denorm_tok).detach()

                    # calculate clip

                    delta_norm_tok = torch.norm(delta_tok, p=2, dim=-1).detach()  # B seq-len
                    mean_norm_tok, _ = torch.max(delta_norm_tok, dim=-1, keepdim=True)  # B,1
                    reweights_tok = (delta_norm_tok / mean_norm_tok).view(bs, seq_len, 1)  # B seq-len, 1

                    delta_tok = delta_tok * reweights_tok

                    total_delta = delta_tok + delta_lb

                    delta_norm = torch.norm(total_delta.view(bs, -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                    reweights = (args.adv_max_norm / delta_norm * exceed_mask \
                                 + (1 - exceed_mask)).view(-1, 1, 1)  # B 1 1

                    # clip

                    delta_lb = (delta_lb * reweights).detach()
                    delta_tok = (delta_tok * reweights).detach()

            # *************************** END *******************
            # End (2) '''

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        print(results)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_step == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    # print('saving gloabl embedding')
    # torch.save(delta_global_embedding, os.path.join("global_embedding.pt"))
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", filenameaddon=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=80):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                # if args.select_pairs:
                #     inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2], "pairs": batch[3],"labels": batch[4]}
                # else:
                #     inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                if "pairs" in inputs:
                    del inputs["pairs"]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        np.savetxt("resault.csv", out_label_ids, delimiter=",")
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results{}.txt".format(filenameaddon))
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (
                    key, str(result[key]) + ' k number is ' + str(args.k) + ' select_pairs ' + str(args.select_pairs)))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, data_size=0):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if False and os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if data_size > 0:
        features = features[:data_size]
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    if args.model_type == "roberta":
        all_token_type_ids = torch.tensor([[0] * 512 for f in features], dtype=torch.long)
    else:
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if args.select_pairs:
        pair_index = select_pais_index(args, all_input_ids, all_attention_mask)
        pair_index = torch.tensor(pair_index, dtype=torch.long)
        new_mask = all_attention_mask.clone()
        for i in range(len(new_mask)):
            pair = pair_index[i]
            for p in pair:
                new_mask[i][p[0]] = 0
                new_mask[i][p[1]] = 0
        new_mask = all_attention_mask - new_mask
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    if args.select_pairs and not evaluate:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, new_mask, pair_index, all_labels)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def select_pais_index(args, all_input_ids, all_attention_mask):
    if args.model_type == 'bert':
        user = os.getlogin()
        checkpoint = f'/nas/home/{user}/VA-aNLI/Output/checkpoint-20000'
        model = BertModel.from_pretrained(checkpoint, output_attentions=True)
    if args.model_type == 'roberta':
        model = AutoModel.from_pretrained(args.model_name_or_path, output_attentions=True)
    seq_len = torch.sum(all_attention_mask, dim=1)
    pair_index = []
    for i in range(len(seq_len)):
        index = []
        input_ids = all_input_ids[i]
        length = seq_len[i]
        inputs = input_ids[:length]
        inputs = inputs.unsqueeze(0)
        outputs = model(inputs)
        attention = outputs[-1]  # Output includes attention weights when output_attentions=True
        attention = np.array(attention[0].detach().numpy(), dtype=float)
        new_att = np.mean(attention, axis=1)
        if args.model_type == 'roberta':
            sepindex = int(torch.where(inputs == 2)[1][0])
        if args.model_type == 'bert':
            sepindex = int(torch.where(inputs == 102)[1][0])
        clsindex = 0
        sep_final = int(length) - 1
        pair_mask = torch.ones((len(inputs[0]), len(inputs[0])))
        for i in range(sepindex + 1):
            for j in range(sepindex + 1):
                pair_mask[i][j] = 0
        for i in range(sepindex, len(inputs[0])):
            for j in range(sepindex, len(inputs[0])):
                pair_mask[i][j] = 0
        # for i in range(len(inputs[0])):
        # pair_mask[i][-1] = 0
        # for i in range(len(inputs[0])):
        #    pair_mask[i][0] = 0
        new_att = torch.from_numpy(new_att)
        new_att = torch.squeeze(new_att)
        new_att = np.multiply(pair_mask, new_att)
        m = torch.argmax(new_att)
        new_att = new_att.detach().numpy()
        max_s = max(sepindex + 1, length - sepindex - 1)
        result = []
        flat = new_att.flatten()
        max_values = np.sort(flat)[::-1]
        r, c = divmod(int(m), len(inputs[0]))
        max_value = new_att[r][c]
        dict = {}
        if args.k > max_s:
            args.k = max_s
        for i in range(args.k):
            index = []
            tmp = ""
            tmp1 = ""
            value = max_values[i]
            value_index = np.where(new_att == value)
            value_index1 = value_index[0]
            value_index2 = value_index[1]
            tmp += str(value_index1) + str(value_index2)
            tmp1 += str(value_index2) + str(value_index1)
            if (tmp not in dict) and (tmp1 not in dict):
                index.append(value_index1)
                index.append(value_index2)
                dict[tmp] = 0
                dict[tmp1] = 0
            else:
                i = i - 1
                continue
            result.append(index)
            if len(result) >= max_s:
                break
        # max_value2 = max_values[0]
        # value_index = np.where(new_att == max_value)
        # value_index1 = value_index[0]
        # value_index2 = value_index[1]
        # index.append(r)
        # index.append(c)
        # index.append(clsindex)
        # index.append(sepindex)
        # index.append(sep_final)
        pair_index.append(result)
    return pair_index


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pre-trained model or shortcut name selected in the list: " + ", "}
        # .join(ALL_MODELS)}
    )
    model_type: str = field(metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_TYPES)})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pre-trained models downloaded from s3"}
    )


@dataclass
class DataProcessingArguments:
    task_name: str = field(
        metadata={"help": "The name of the task to train selected in the list: " + ", ".join(processors.keys())}
    )
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class ATArgs:
    adv_train: int = field()
    adv_steps: int = field()
    adv_init_mag: float = field()
    adv_max_norm: float = field()
    adv_lr: float = field()
    data_size: int = field(default=0)
    vocab_size: int = field(default=30522)
    hidden_size: int = field(default=768)
    # evaluate_during_training: bool = field(default=True)
    select_pairs: bool = field(default=True)
    mode: str = field(default="train")
    treshhold: int = field(default=0.3)
    k: int = field(default=2)
    save_step: int = field(default=500)  # 500
    local_rank1: int = field(default=2)
    lamda: float = field(default=10.0)
    per_device_eval_batch_size: int = field(default=5)
    add_constraint: bool = field(default=True)


@hydra.main(config_path="../Configs", config_name="TTVAT_config")
def main(config: omegaconf.dictconfig.DictConfig):
    # parser = HfArgumentParser((ModelArguments, DataProcessingArguments, TrainingArguments, ATArgs))
    # model_args, dataprocessing_args, training_args, at_args = parser.parse_args_into_dataclasses()

    # For now, let's merge all the sets of args into one,
    # but soon, we'll keep distinct sets of args, with a cleaner separation of concerns.

    # args = argparse.Namespace(**vars(model_args),
    # **vars(dataprocessing_args), **vars(training_args), **vars(at_args))

    args = argparse.Namespace(**dict(config))

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    print('gpu number' + str(torch.cuda.device_count()))
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir,
    )
    args.vocab_size = config.vocab_size
    args.hidden_size = config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, cache_dir=args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False,
                                                data_size=args.data_size)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    args.do_eval = True
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        checkpoints = [args.output_dir]
        args.eval_all_checkpoints = False
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
