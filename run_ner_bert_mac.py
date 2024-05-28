## Import custom modules and utility functions
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from MatsciBERT_MRC.tools.finetuning_argparse import get_argparse
from torch.optim import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything, json_to_text
from tools.common import init_logger, logger

from transformers import AutoTokenizer, AutoConfig
from models.bert_for_ner import BertSpanForNer, BertSpanBaseForNer
from processors.ner_span import convert_examples_to_features
from processors.ner_span import ner_processors as processors
from processors.ner_span import collate_fn
from metrics.ner_metrics import SpanEntityScore
from processors.utils_ner import bert_extract_item, bert_extract_items

# Set up CUDA visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (AutoConfig, BertSpanForNer, AutoTokenizer),
}


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    metric = SpanEntityScore(args.id2label)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                                  collate_fn=collate_fn)
    # Calculate total training steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    # param_optimizer = list(model.named_parameters())
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer
    #                 if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in param_optimizer
    #                 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    bert_parameters = model.bert.named_parameters()
    start_parameters = model.start_fc.named_parameters()
    end_parameters = model.end_fc.named_parameters()
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': args.learning_rate},

        {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.other_learning_rate},
        {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': args.other_learning_rate},

        {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.other_learning_rate},
        {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
            , 'lr': args.other_learning_rate},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    print('t_total:', t_total)

   # Start training
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args.num_train_epochs)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    # model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for _ in range(int(args.num_train_epochs)):
        print('Epoch:', _ + 1)
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                      "start_positions": batch[3], "end_positions": batch[4]}
            # optimizer.zero_grad()
            outputs = model(**inputs)
            loss, start_logits, end_logits = outputs[:3]  # model outputs are always tuple in pytorch-transformers (see doc)

            R_final = bert_extract_item(start_logits, end_logits)
            T = bert_extract_items(batch[3], batch[4])
            metric.update(true_subject=T, pred_subject=R_final)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()
            global_step += 1
            args.logging_steps = int(t_total / args.num_train_epochs)
            if args.local_rank in [-1, 0] and global_step >= 1 * args.logging_steps and global_step % args.logging_steps == 0:
                evaluate(args, model, tokenizer)
        logger.info("\n")
        tr_loss = tr_loss / len(train_dataloader)
        tr_info, entity_info = metric.result()
        results = {f'{key}': value for key, value in tr_info.items()}
        results['loss'] = tr_loss
        print("***** Train results %s *****")
        info = "-".join([f' {key}: {value:.6f} ' for key, value in results.items()])
        print(info)
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    # print(args, model, tokenizer)
    return tr_loss


def evaluate(args, model, tokenizer, prefix=""):
    """ Evaluation Model """
    metric = SpanEntityScore(args.id2label)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_features = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')
    # eval_sampler = SequentialSampler(eval_features)
    # eval_dataloader = DataLoader(eval_features, sampler=eval_sampler, batch_size=args.batch_size,
    #                              collate_fn=collate_fn)
    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_features), desc="Evaluating")
    for step, f in enumerate(eval_features):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(args.device)
        input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(args.device)
        segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(args.device)
        start_ids = torch.tensor([f.start_ids[:input_lens]], dtype=torch.long).to(args.device)
        end_ids = torch.tensor([f.end_ids[:input_lens]], dtype=torch.long).to(args.device)
        model.eval()
        with torch.no_grad():
            # batch = tuple(t.to(args.device) for t in batch)
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
            #           "start_positions": batch[3], "end_positions": batch[4]}
            inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids,
                      "start_positions": start_ids, "end_positions": end_ids}
            outputs = model(**inputs)
            tmp_eval_loss, start_logits, end_logits = outputs[:3]
        # input_lens_new = torch.sum(input_mask.view(-1) - segment_ids.view(-1))
        # print(start_logits)
        R_final = bert_extract_item(start_logits, end_logits)
        T = bert_extract_items(start_ids, end_ids)
        # print(R_final)
        # print(T)
        metric.update(true_subject=T, pred_subject=R_final)
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        pbar(step)
    logger.info("\n")
    eval_loss = eval_loss / len(eval_features)
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    print("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.6f} ' for key, value in results.items()])
    print(info)
    return results


def predict(args, model, tokenizer, prefix=""):
    "" Predictive Model ""
    metric = SpanEntityScore(args.id2label)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_features = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test')
    # eval_sampler = SequentialSampler(eval_features)
    # eval_dataloader = DataLoader(eval_features, sampler=eval_sampler, batch_size=args.batch_size,
    #                              collate_fn=collate_fn)
    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_features), desc="Predicting")
    for step, f in enumerate(eval_features):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(args.device)
        input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(args.device)
        segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(args.device)
        start_ids = torch.tensor([f.start_ids[:input_lens]], dtype=torch.long).to(args.device)
        end_ids = torch.tensor([f.end_ids[:input_lens]], dtype=torch.long).to(args.device)
        model.eval()
        with torch.no_grad():
            # batch = tuple(t.to(args.device) for t in batch)
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
            #           "start_positions": batch[3], "end_positions": batch[4]}
            inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids,
                      "start_positions": start_ids, "end_positions": end_ids}
            outputs = model(**inputs)
            tmp_eval_loss, start_logits, end_logits = outputs[:3]
        # input_lens_new = torch.sum(input_mask.view(-1) - segment_ids.view(-1))
        # print(start_logits)
        R_final = bert_extract_item(start_logits, end_logits)
        T = bert_extract_items(start_ids, end_ids)
        # print(R_final)
        # print(T)
        metric.update(true_subject=T, pred_subject=R_final)
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        pbar(step)
    logger.info("\n")
    eval_loss = eval_loss / len(eval_features)
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    print("***** Test results %s *****", prefix)
    info = "-".join([f' {key}: {value:.6f} ' for key, value in results.items()])
    print(info)
    return results


def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    """ Load and cache data set """
    processor = processors[task]()
    
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir, "cached_{}_{}_{}".format(data_type, tokenizer.__class__.__name__, str(args.max_seq_length)),
    )
    if os.path.exists(cached_features_file):
        print(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        print(label_list)
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type == 'train' else
                                                args.eval_max_seq_length
                                                )
        torch.save(features, cached_features_file)
        
    # Convert to Tensors and build dataset
    if data_type == 'dev' or data_type == 'test':
        return features
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_input_lens)
    return dataset


def main():
    args = get_argparse().parse_args()
    
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
        
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    # Set seed
    seed_everything(args.seed)
    
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    print(args.task_name)
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    config.soft_label = True
    config.loss_type = args.loss_type
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)

    # Training
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')
    tr_loss = train(args, train_dataset, model, tokenizer)
    print("average loss = %s",  tr_loss)


if __name__ == "__main__":
    main()
