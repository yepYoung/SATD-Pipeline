"""BERT finetuning runner."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import recall_score, precision_score

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pandas as pd

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization
from modeling_multitask import BertConfig, BertForSequenceClassification
from optimization import BERTAdam

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
torch.cuda.set_device(3)

class InputExample(object):
    def __init__(self, text_a, label=None, dataset_label=None):
        self.text_a = text_a
        self.label = label
        self.dataset_label = dataset_label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, dataset_label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.dataset_label_id = dataset_label_id


class Processor():
    def get_train_examples(self, data_dir):
        train_data_issue = pd.read_csv(
            os.path.join(data_dir, 'train_2issues_new.csv'), sep=",").values
        train_data_request = pd.read_csv(
            os.path.join(data_dir, 'train_3pull-requests_new.csv'), sep=",").values
        train_data_commit = pd.read_csv(
            os.path.join(data_dir, 'train_4commit-messages_new.csv'), sep=",").values
        train_data_code_comments = pd.read_csv(
            os.path.join(data_dir, 'train_1code-comments_new_1.csv'), sep=",").values
        return self.create_examples(train_data_issue, train_data_request, train_data_commit, train_data_code_comments)

    def get_dev_examples(self, data_dir):
        test_data_issue = pd.read_csv(
            os.path.join(data_dir, 'dev_2issues_new.csv'), header=None, sep=",").values
        test_data_request = pd.read_csv(
            os.path.join(data_dir, 'dev_3pull-requests_new.csv'), header=None, sep=",").values
        test_data_commit = pd.read_csv(
            os.path.join(data_dir, 'dev_4commit-messages_new.csv'), header=None, sep=",").values
        test_data_code_comments = pd.read_csv(
            os.path.join(data_dir, 'dev_1code-comments_new_1.csv'), header=None, sep=",").values
        test_data_issue = np.delete(test_data_issue, 0, 0)
        test_data_request = np.delete(test_data_request, 0, 0)
        test_data_commit = np.delete(test_data_commit, 0, 0)
        test_data_code_comments = np.delete(test_data_code_comments, 0, 0)
        return self._create_examples(test_data_issue, test_data_request, test_data_commit, test_data_code_comments,
                                     "test")
        
    def get_labels(self):
        return [["no", "yes"],
                ["no", "yes"],
                ["no", "yes"],
                ["no", "yes"]]

    def create_examples(self, lines_issue, lines_request, lines_commit, lines_code_comments):
        examples = []
        for line in  lines_issue:
            text_a = tokenization.convert_to_unicode(str(line[2]))
            label = tokenization.convert_to_unicode(str(line[4]))
            examples.append(InputExample(text_a=text_a,label=label, dataset_label="2"))

        for line in lines_request:
            text_a = tokenization.convert_to_unicode(str(line[2]))
            label = tokenization.convert_to_unicode(str(line[4]))
            examples.append(InputExample(text_a=text_a,label=label, dataset_label="3"))

        for line in lines_commit:
            text_a = tokenization.convert_to_unicode(str(line[2]))
            label = tokenization.convert_to_unicode(str(line[4]))
            examples.append(InputExample(text_a=text_a, label=label, dataset_label="4"))

        for line in lines_code_comments:
            text_a = tokenization.convert_to_unicode(str(line[2]))
            label = tokenization.convert_to_unicode(str(line[4]))
            examples.append(InputExample(text_a=text_a, abel=label, dataset_label="1"))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):

    features_1 = []
    features_2 = []
    features_3 = []
    features_4 = []
    for example in tqdm(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        # 填充
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        if example.label == "no": label_id = 0
        elif example.label == "yes": label_id = 1

        if example.dataset_label == "1":
            features_1.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))
        if example.dataset_label == "2":
            features_2.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))
        if example.dataset_label == "3":
            features_3.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))
        if example.dataset_label == "4":
            features_4.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))

    return features_1, features_2, features_3, features_4




def main():
    processor = Processor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 梯度积累
    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    bert_config = BertConfig.from_json_file(bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size * args.num_train_epochs)

    model = BertForSequenceClassification(bert_config, len(label_list))
    if args.init_checkpoint is not None:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'), strict=False)

    #  torch.save(model.bert.state_dict(),"pytorch_model.bin")
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ['bias', 'gamma', 'beta']
    if args.discr:
        group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.']
        group2 = ['layer.6.', 'layer.7.', 'layer.8.', 'layer.9.', 'layer.10.']
        group3 = ['layer.11.']
        group_all = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.',
                     'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
             'weight_decay_rate': 0.01, 'lr': args.learning_rate / 1.5},
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
             'weight_decay_rate': 0.01, 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
             'weight_decay_rate': 0.01, 'lr': args.learning_rate * 1.5},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.0},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay_rate': 0.0,
             'lr': args.learning_rate / 1.5},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay_rate': 0.0,
             'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay_rate': 0.0,
             'lr': args.learning_rate * 1.5},
        ]
    else:
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features_1, eval_features_2, eval_features_3, eval_features_4 = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features_1], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features_1], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features_1], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features_1], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in eval_features_1], dtype=torch.long)
    eval_data_1 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    eval_dataloader_1 = DataLoader(eval_data_1, batch_size=args.eval_batch_size, shuffle=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features_2], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features_2], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features_2], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features_2], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in eval_features_2], dtype=torch.long)
    eval_data_2 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    eval_dataloader_2 = DataLoader(eval_data_2, batch_size=args.eval_batch_size, shuffle=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features_3], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features_3], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features_3], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features_3], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in eval_features_3], dtype=torch.long)
    eval_data_3 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    eval_dataloader_3 = DataLoader(eval_data_3, batch_size=args.eval_batch_size, shuffle=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features_4], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features_4], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features_4], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features_4], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in eval_features_4], dtype=torch.long)
    eval_data_4 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    eval_dataloader_4 = DataLoader(eval_data_4, batch_size=args.eval_batch_size, shuffle=False)
    best_f1 = 0

    if args.do_train:
        all_input_ids = torch.tensor([f.input_ids for f in train_features_1], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features_1], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features_1], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features_1], dtype=torch.long)
        all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in train_features_1], dtype=torch.long)
        train_data_1 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                     all_dataset_label_ids)
        train_dataloader_1 = DataLoader(train_data_1, sampler=train_sampler_1, batch_size=args.train_batch_size)

        a = []
        for i in range(int(len(train_features_1) / args.train_batch_size)):
            a.append(1)
        for i in range(int(len(train_features_2) / args.train_batch_size)):
            a.append(2)
        for i in range(int(len(train_features_3) / args.train_batch_size)):
            a.append(3)
        for i in range(int(len(train_features_4) / args.train_batch_size)):
            a.append(4)
        print("len(a)=", len(a))
        random.shuffle(a)
        print("a[:20]=", a[:20])

        epoch = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            random.shuffle(a)
            print("a[:20]=", a[:20])
            epoch += 1
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, number in enumerate((tqdm(a, desc="Iteration"))):
                if number == 1: batch = train_dataloader_1.__iter__().__next__()
                if number == 2: batch = train_dataloader_2.__iter__().__next__()
                if number == 3: batch = train_dataloader_3.__iter__().__next__()
                if number == 4: batch = train_dataloader_4.__iter__().__next__()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, dataset_label_id = batch
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()  # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1

            model.eval()
            # issue
            eval_loss, eval_accuracy, eval_precision_issue, eval_recall_issue, eval_f1_issue = 0, 0, 0, 0, 0
            predict_label_list_issue = []
            origin_label_list_issue = []
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_issue_" + str(epoch) + ".txt"), "w") as f:
                for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in eval_dataloader_2:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output) + "\n")
                    tmp_eval_accuracy = np.sum(outputs == label_ids)
                    origin_label_list_issue.extend(label_ids)
                    predict_label_list_issue.extend(outputs)
                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            # request
            eval_loss, eval_accuracy, eval_precision_pr, eval_recall_pr, eval_f1_pr = 0, 0, 0, 0, 0
            predict_label_list_pr = []
            origin_label_list_pr = []
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_request_" + str(epoch) + ".txt"), "w") as f:
                for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in eval_dataloader_3:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output) + "\n")
                    tmp_eval_accuracy = np.sum(outputs == label_ids)
                    origin_label_list_pr.extend(label_ids)
                    predict_label_list_pr.extend(outputs)
                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            # commit
            eval_loss, eval_accuracy, eval_precision_commit, eval_recall_commit, eval_f1_commit = 0, 0, 0, 0, 0
            predict_label_list_commit = []
            origin_label_list_commit = []
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_commit_" + str(epoch) + ".txt"), "w") as f:
                for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in eval_dataloader_4:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output) + "\n")
                    tmp_eval_accuracy = np.sum(outputs == label_ids)
                    origin_label_list_commit.extend(label_ids)
                    predict_label_list_commit.extend(outputs)
                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps

            # code comments
            eval_loss, eval_accuracy, eval_precision_code_comments, eval_recall_code_comments, eval_f1_code_comments = 0, 0, 0, 0, 0
            predict_label_list_code_comments = []
            origin_label_list_code_comments = []
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_code_comments_" + str(epoch) + ".txt"), "w") as f:
                for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in eval_dataloader_1:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output) + "\n")
                    tmp_eval_accuracy = np.sum(outputs == label_ids)
                    origin_label_list_code_comments.extend(label_ids)
                    predict_label_list_code_comments.extend(outputs)
                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            eval_loss_code_comments = eval_loss

            if avg_f1 > best_f1:
                torch.save({
                    'bert': model.bert.state_dict(),
                    'classifier_1': model.classifier_1.state_dict(),
                    'classifier_2': model.classifier_2.state_dict(),
                    'classifier_3': model.classifier_3.state_dict(),
                    'classifier_4': model.classifier_4.state_dict(),
                }, os.path.join(args.output_dir, 'pytorch_model' + str(epoch) + '.bin'))

                best_f1 = avg_f1
