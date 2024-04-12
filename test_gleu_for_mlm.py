from datasets import load_dataset, load_metric
from transformers import AdamW, get_scheduler, DebertaV2Tokenizer, T5Tokenizer, Adafactor, DataCollatorForWholeWordMask, DataCollatorForLanguageModeling, DataCollatorWithPadding

from Debertav2_transformers import DebertaV2ForMaskedLM, DebertaV2Config, DebertaV2ForSequenceClassification
from T5_transformers import T5ForSequenceClassification, T5Config
from utils import parse_args, tf_make_result_path, seed_fix

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm
import os
import json
import wandb
import argparse

MODEL_LIST = {
    "deberta":{
        "tokenizer" : DebertaV2Tokenizer,
        "model" : DebertaV2ForMaskedLM,
        "config" : DebertaV2Config,
        "model_load_path" : "microsoft/deberta-v3-large"
    }
}

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

args = parse_args()
seed_fix(args.seed)
device = "cuda:"+str(args.gpu)

model_type = None
for i in list(MODEL_LIST.keys()):
    if i in args.result_path.split("_"):
        model_type = i
if model_type is None:
    assert "result_path is must include modeltype!!"    

args.result_path = tf_make_result_path(args)


task = args.glue_task
if task == "rte":
    args.batch_size = 8

print("{}_{}".format(args.result_path, task))
print("check duplicate")
# for r in runs:
#     if r.state == "crashed" or r.state == "failed":
#         continue
    
#     if r.name == "{}_{}".format(args.result_path, task):
#         raise "duplicate experiment"


model_utils = MODEL_LIST[model_type]
if args.model_load_path is not None:
    model_utils['model_load_path'] = args.model_load_path
    
if args.dev:
    args.result_path = "test_"+args.result_path
    

os.makedirs(args.data_path, exist_ok=True)
dataset = load_dataset("glue", task , cache_dir=args.data_path)
metric = load_metric('glue', task , cache_dir=args.data_path)
print(dataset)

num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
tmp_matric = metric.compute(predictions=torch.Tensor([0,1]), references=torch.Tensor([0,1]))
    
tokenizer = model_utils["tokenizer"].from_pretrained(model_utils["model_load_path"])


def custom_collate_fn_for_train(batches):
    # assert 0
    sentence1_key, sentence2_key = task_to_keys[task]
    if sentence2_key is None:
        texts = batches[sentence1_key]
    else:
        texts = batches[sentence1_key] + "\n" + batches[sentence2_key]
    
    # inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(texts, max_length=tokenizer.model_max_length, padding=False, truncation=True, return_special_tokens_mask=True)

    return model_inputs

data_collator_mlm = DataCollatorForLanguageModeling(tokenizer)
data_collator_pad = DataCollatorWithPadding(tokenizer)
def custom_collate_fn_for_valid(batches):
    sentence1_key, sentence2_key = task_to_keys[task]
    if sentence2_key is None:
        texts = batches[sentence1_key]
    else:
        texts = batches[sentence1_key] + "\n" + batches[sentence2_key]
    
    # inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(texts, max_length=tokenizer.model_max_length, padding=False, truncation=True, return_special_tokens_mask=True, return_tensors="pt")
    
    model_inputs["input_ids"], model_inputs["labels"] = data_collator_mlm.torch_mask_tokens(model_inputs["input_ids"], model_inputs["special_tokens_mask"])
    # print(model_inputs)

    return model_inputs


column_names = dataset["train"].column_names
dataset["train"] = dataset["train"].map(
                custom_collate_fn_for_train,
                # batched=True,
                num_proc=1,
                remove_columns=column_names
            )
print(dataset["train"])
dataset["validation"] = dataset["validation"].map(
                custom_collate_fn_for_train,
                # batched=True,
                num_proc=1,
                remove_columns=column_names
            )
print(dataset["validation"])

train_dataloader = DataLoader(dataset["train"], batch_size=args.batch_size, collate_fn=data_collator_mlm, num_workers=4, shuffle=True)
val_dataloader = DataLoader(dataset["validation"], batch_size=args.batch_size, collate_fn=data_collator_mlm, num_workers=4, shuffle=False)


model = model_utils["model"].from_pretrained(model_utils["model_load_path"], num_labels=num_labels)


if "deberta" in args.result_path.split("_"):  
    pre_trained_model = model.deberta
elif "t5" in args.result_path.split("_"):
    model.config.problem_type="single_label_classification"
    pre_trained_model = model.transformer

if task == "stsb":
    model.config.problem_type = "regression"

if not args.no_add_linear:
    if args.add_linear_layer is None:
        if args.add_linear_num is None:
            args.add_linear_layer = range(model.config.num_hidden_layers)
        elif args.add_linear_num > 0:
            args.add_linear_layer = range(args.add_linear_num)
        else:
            args.add_linear_layer = range(model.config.num_hidden_layers + args.add_linear_num, model.config.num_hidden_layers)
    
    if args.add_position == "befdot":
        pre_trained_model.add_unit_init_before_dotpro(layer_num=args.add_linear_layer, head_indi=args.head_indi, init_type=args.init_type, act_type=args.act_type)
    elif args.add_position == "afterffnn":
        pre_trained_model.add_unit_init_after_ffnn(layer_num=args.add_linear_layer, init_type=args.init_type, act_type=args.act_type)
    elif args.add_position == "both":
        pre_trained_model.add_unit_init_after_ffnn(layer_num=args.add_linear_layer, init_type=args.init_type, act_type=args.act_type)
        pre_trained_model.add_unit_init_before_dotpro(layer_num=args.add_linear_layer, head_indi=args.head_indi, init_type=args.init_type, act_type=args.act_type)
        
    elif args.add_position == "aftffnn1":
        pre_trained_model.add_unit_init_after_ffnn1(layer_num=args.add_linear_layer, init_type=args.init_type, act_type=args.act_type)
    elif args.add_position == "aftffnn2":
        pre_trained_model.add_unit_init_after_ffnn2(layer_num=args.add_linear_layer, init_type=args.init_type, act_type=args.act_type)

if args.adapter:
    for name, param in model.named_parameters():
        if "added" not in name and "classifi" not in name:
            param.requires_grad_(requires_grad=False)
            #param.requires_grad=False
        else:
            param.requires_grad_(requires_grad=True)
    
    
    for name, param in model.named_parameters():
        print(param.requires_grad,"\t/\t",name)


model.to(device)
print(model)


args.model_config = model.config
save_path='checkpoints/'+args.result_path+"_"+task
os.makedirs(save_path,exist_ok=True)

optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=[args.beta1,args.beta2], weight_decay=args.weight_decay, eps=args.eps)
# optimizer = Adafactor(model.parameters(), lr=args.learning_rate, relative_step=False, warmup_init=False)
scheduler = get_scheduler("linear", optimizer, args.warmup_steps, len(train_dataloader)* args.epoch)


for name, param in model.named_parameters():
    if "added" in name:
        print(name, param)
        break


for E in range(1, args.epoch+1):
    model.train()
    
    losses = []
    dl = tqdm(train_dataloader)
    for batches in dl:
        for idx in batches.keys():
            batches[idx] = batches[idx].to(device)
        
        out = model(**batches)
        
        out.loss.backward()
        losses.append(out.loss.item())
        
        if not args.adapter:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        dl.set_description("loss="+str(out.loss.item()))

    print("train_loss = {}".format(sum(losses)/len(losses)))
    
    change_score_name = dict()
    change_score_name["{}_train_loss".format(task)] = sum(losses)/len(losses)

    model.eval()
    losses = []
    accs = []
    best_dev_score = 0
    best_test_score = 0
    with torch.no_grad():
        for batches in tqdm(val_dataloader):
            for idx in batches.keys():
                batches[idx] = batches[idx].to(device)
                
            out = model(**batches)

            pred = torch.argmax(out.logits, dim=-1).view(-1)
            label = batches["labels"].view(-1)

            acc = sum(pred == label) / (len(label) - sum(-100 == label))
            accs.append(acc.item())

            losses.append(out.loss.item())

        print("dev_loss = {}".format(sum(losses)/len(losses)))

        print("dev_acc = {}".format(sum(accs)/len(accs)))
        
        change_score_name["{}_dev_loss".format(task)] = sum(losses)/len(losses)
        change_score_name["{}_dev_acc".format(task)] = sum(accs)/len(accs)

        # change_score_name["{}_dev_{}".format(task, "acc")] = sum(pred_list == label_list)/pred_list.size()[0]
        change_score_name["epoch"] = E+1

    for name, param in model.named_parameters():
        if "added" in name:
            print(name, param)
            break

    change_score_name["epoch"] = E+1
    
    # model_name="model_"+str(E)+".pt"
    # torch.save(model.state_dict(), os.path.join(save_path, model_name))
    