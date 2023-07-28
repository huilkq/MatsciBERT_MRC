import glob
import logging
import os
import json
from typing import List, Optional

import torch
from filelock import FileLock

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from callback.progressbar import ProgressBar
from tools.common import json_to_text, seed_everything
from tools.common import logger

from transformers import BertConfig, get_linear_schedule_with_warmup, AdamW, BertTokenizer, PreTrainedTokenizer
from models.bert_for_ner import BertSpanForNer
from processors.ner_span import convert_examples_to_features, BnerProcessor, CluenerProcessor
from processors.ner_span import ner_processors as processors
from processors.ner_span import collate_fn
from metrics.ner_metrics import SpanEntityScore
from MatsciBERT_MRC.processors.utils_ner import bert_extract_item
from MatsciBERT_MRC.config import model_config


# coding:gbk

def train(model, train_loader, optimizer, scheduler):
    """ Train the model """
    # Train!
    global_step = 0
    tr_loss = 0
    seed_everything(42)  # Added here for reproductibility (even between python 2 and 3)
    for step, batch in enumerate(tqdm(train_loader)):
        model.train()
        batch = tuple(t.to(model_config.device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                    "start_positions": batch[3], "end_positions": batch[4]}
        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
        loss.backward()
        tr_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # Update learning rate schedul
        model.zero_grad()
        global_step += 1
    if 'cuda' in str(model_config.device):
        torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(model, dev_dataset, prefix=""):
    metric = SpanEntityScore(model_config.ids_to_labels)
    eval_output_dir = model_config.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    for step, f in enumerate(dev_dataset):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(model_config.device)
        input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(model_config.device)
        segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(model_config.device)
        start_ids = torch.tensor([f.start_ids[:input_lens]], dtype=torch.long).to(model_config.device)
        end_ids = torch.tensor([f.end_ids[:input_lens]], dtype=torch.long).to(model_config.device)
        subjects = f.subjects
        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": input_ids, "attention_mask": input_mask,
                      "start_positions": start_ids, "end_positions": end_ids}
            outputs = model(**inputs)
        tmp_eval_loss, start_logits, end_logits = outputs[:3]
        R = bert_extract_item(start_logits, end_logits)
        T = subjects
        metric.update(true_subject=T, pred_subject=R)
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
    print("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    print("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    print(info)
    print("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        print("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        print(info)
    return results


def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test')
    print(len(test_dataset))
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)

    results = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_predict.json")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "start_positions": None, "end_positions": None}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        start_logits, end_logits = outputs[:2]
        R = bert_extract_item(start_logits, end_logits)
        if R:
            label_entities = [[args.id2label[x[0]], x[1], x[2]] for x in R]
        else:
            label_entities = []
        json_d = {}
        json_d['id'] = step
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    logger.info("\n")
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    if args.task_name == "cluener":
        output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
        test_text = []
        with open(os.path.join(args.data_dir, "test.json"), 'r') as fr:
            for line in fr:
                test_text.append(json.loads(line))
        test_submit = []
        for x, y in zip(test_text, results):
            json_d = {}
            json_d['id'] = x['id']
            json_d['label'] = {}
            entities = y['entities']
            words = list(x['text'])
            if len(entities) != 0:
                for subject in entities:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]
                    word = "".join(words[start:end + 1])
                    if tag in json_d['label']:
                        if word in json_d['label'][tag]:
                            json_d['label'][tag][word].append([start, end])
                        else:
                            json_d['label'][tag][word] = [[start, end]]
                    else:
                        json_d['label'][tag] = {}
                        json_d['label'][tag][word] = [[start, end]]
            test_submit.append(json_d)
        json_to_text(output_submit_file, test_submit)


def load_and_cache_examples(
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        data_type='train'
):
    # Load the dataset file from the cache or the dataset file
    cached_features_file = os.path.join(
        data_dir, "cached_{}_{}_{}".format(data_type, tokenizer.__class__.__name__, str(max_seq_length)),
    )
    # Ensure that only the first process handles the data set in distributed training,
    # Others will use the cache.
    lock_path = cached_features_file + ".lock"
    with FileLock(lock_path):

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            # Load the data set from the cache
            features = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {data_dir}")
            if data_type == 'train':
                examples = processors.get_train_examples(data_dir)
            elif data_type == 'dev':
                examples = processors.get_dev_examples(data_dir)
            else:
                examples = processors.get_test_examples(data_dir)
            # TODO clean up all this to leverage built-in features of tokenizers
            features = convert_examples_to_features(examples=examples,
                                                    tokenizer=tokenizer,
                                                    label_list=labels,
                                                    max_seq_length=max_seq_length,
                                                    )
            logger.info(f"Saving features into cached file {cached_features_file}")
            # Save the processed data set to the cache
            torch.save(features, cached_features_file)
    if data_type == 'dev':
        return features
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_input_lens)
    return dataset


def all_train(train_loader, dev_loader, model, optimizer, scheduler):
    for epoch in range(1, model_config.epochs + 1):
        print('=========train at epoch={}========='.format(epoch))
        train(model, train_loader, optimizer, scheduler)
        print('=========val at epoch={}========='.format(epoch))
        evaluate(model, dev_loader)


def main():

    # Prepare NER task
    processor = CluenerProcessor()
    label_list = processor.get_labels()
    model_config.ids_to_labels = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    print(label_list)

    # Load pretrained model and tokenizer
    config = BertConfig.from_pretrained(model_config.MATSCIBERT, num_labels=num_labels)
    config.soft_label = True
    config.loss_type = model_config.loss_type
    tokenizer = BertTokenizer.from_pretrained(model_config.MATSCIBERT)
    model = BertSpanForNer.from_pretrained(model_config.MODEL,
                                                          config=config).to(model_config.device)

    # Define training and validation set data
    train_dataset = load_and_cache_examples(
        data_dir=model_config.FILE_NAME,
        tokenizer=tokenizer,
        labels=label_list,
        max_seq_length=model_config.max_seq_length,
        data_type='train'
    )
    dev_dataset = load_and_cache_examples(
        data_dir=model_config.FILE_NAME,
        tokenizer=tokenizer,
        labels=label_list,
        max_seq_length=model_config.max_seq_length,
        data_type='dev'
    )
    # train_dataset, dev_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    # Obtain training and validation set data in batches
    train_sampler = RandomSampler(train_dataset)  # Random sampling
    dev_sampler = SequentialSampler(dev_dataset)  # Sequential sampling
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=model_config.batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=model_config.batch_size, sampler=dev_sampler,
                                collate_fn=collate_fn)

    # Cross entropy loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    no_decay = ['bias', 'LayerNorm.weight']  # The term in which the control factor does not decay
    # Define optimizer
    bert_parameters = model.bert.named_parameters()
    start_parameters = model.start_fc.named_parameters()
    end_parameters = model.end_fc.named_parameters()
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": model_config.weight_decay, 'lr': model_config.lr},
        {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': model_config.lr},

        {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": model_config.weight_decay, 'lr': 0.001},
        {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': 0.001},

        {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": model_config.weight_decay, 'lr': 0.001},
        {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': 0.001},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=model_config.lr, eps=model_config.adam_epsilon)  # AdamW
    #  Warmup The learning rate is warmed up
    total_steps = len(train_dataset) * model_config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=model_config.warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)

    # Train the model
    logging.info("--------Start Training!--------")
    all_train(train_dataloader, dev_dataset, model, optimizer, scheduler)


if __name__ == "__main__":
    main()
