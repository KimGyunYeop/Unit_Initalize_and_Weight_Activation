from datasets import load_dataset, load_metric
from transformers import AdamW, get_scheduler, get_cosine_schedule_with_warmup
from transformers import ViTImageProcessor #, ViTForImageClassification
from transformers import AutoImageProcessor, ConvNextForImageClassification
# from transformers.models.vit.configuration_vit import ViTConfig
from modeling_vit import ViTForImageClassification, ViTConfig

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils import parse_args, tf_make_result_path, seed_fix

from tqdm import tqdm
import os
import json
import wandb
import argparse


MODEL_LIST = {
    "vit":{
        "model" : ViTForImageClassification,
        "image_processor" : ViTImageProcessor,
        "model_load_path" : "google/vit-base-patch16-224-in21k",
        # "image_processor_load_path" : "google/vit-base-patch16-224-in21k" 
        "image_processor_load_path" : "google/vit-base-patch16-384"   
    },
    "convnext":{
        "model" : ConvNextForImageClassification,
        "image_processor" : AutoImageProcessor,
        "model_load_path" : "facebook/convnext-tiny-224"
    }
}

dataset_to_imagedict = { 
    "cifar10": 'img',
    "cifar100": 'img',
    "imagenet-1k": 'image', 
}
dataset_to_labeldict = { 
    "cifar10": 'label',
    "cifar100": 'fine_label',
    "imagenet-1k": 'label', 
}

args = parse_args()
seed_fix(args.seed)
device = "cuda:"+str(args.gpu) 

model_type = None
for i in list(MODEL_LIST.keys()): # "vit", "convnext"
    if i in args.result_path.split("_"):
        model_type = i
if model_type is None:
    assert "result path must include model type!"    

args.result_path = tf_make_result_path(args)

model_utils = MODEL_LIST[model_type] # model, image_processor, model_load_path
if args.model_load_path is not None:
    model_utils['model_load_path'] = args.model_load_path
    
if args.dev:
    args.result_path = "test_"+args.result_path

##################################################
    
dataset_name = args.image_classification_dataset 
dataset_labeldict = dataset_to_labeldict[dataset_name]
dataset_imagedict = dataset_to_imagedict[dataset_name]
dataset = load_dataset(dataset_name , cache_dir=args.data_path, use_auth_token=True) 
metric = load_metric("accuracy") 

if dataset_name=="cifar100":
    eval_step=args.cifar_eval_step
elif dataset_name=="imagenet-1k":
    eval_step=args.imagenet_eval_step

processor = model_utils["image_processor"].from_pretrained(model_utils["image_processor_load_path"])

def custom_collate_fn(batches):
    if dataset_name=="imagenet-1k":
      inputs = processor([batch[dataset_imagedict].convert('RGB') for batch in batches], return_tensors='pt') #inputs['pixel_values']
    else:
      inputs = processor([batch[dataset_imagedict] for batch in batches], return_tensors='pt') #inputs['pixel_values']

    inputs['labels'] = torch.tensor([batch[dataset_labeldict] for batch in batches])
    
    return inputs

train_dataloader = DataLoader(dataset["train"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4, shuffle=True)
try:
    val_dataloader = DataLoader(dataset["validation"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4)
except:
    val_dataloader = DataLoader(dataset["test"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4)
test_dataloader = DataLoader(dataset["test"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4)
if dataset_name=="imagenet-1k":
    test_dataloader = DataLoader(dataset["validation"], batch_size=args.batch_size, collate_fn=custom_collate_fn, num_workers=4)

##################################################

labels = dataset['train'].features[dataset_labeldict].names


model = model_utils["model"].from_pretrained(
    model_utils["model_load_path"], 
    num_labels=len(labels),
    #id2label={str(i): c for i, c in enumerate(labels)},
    #label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True
    )

if not args.no_add_linear:
    if args.add_linear_layer is None:
        if args.add_linear_num is None:
            args.add_linear_layer = range(model.config.num_hidden_layers)
        elif args.add_linear_num > 0:
            args.add_linear_layer = range(args.add_linear_num)
        else:
            args.add_linear_layer = range(model.config.num_hidden_layers + args.add_linear_num, model.config.num_hidden_layers)
    print(args.add_linear_layer)
    if args.add_position == "befdot":
        model.vit.add_unit_init_before_dotpro(layer_num=args.add_linear_layer, head_indi=args.head_indi, init_type=args.init_type, act_type=args.act_type)
    elif args.add_position == "afterffnn":
        model.vit.add_unit_init_after_ffnn(layer_num=args.add_linear_layer, init_type=args.init_type, act_type=args.act_type)
    elif args.add_position == "both":
        model.vit.add_unit_init_after_ffnn(layer_num=args.add_linear_layer, init_type=args.init_type, act_type=args.act_type)
        model.vit.add_unit_init_before_dotpro(layer_num=args.add_linear_layer, head_indi=args.head_indi, init_type=args.init_type, act_type=args.act_type)
    

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
# assert 0

# optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=[args.beta1,args.beta2], weight_decay=args.weight_decay, eps=args.eps)
# scheduler = get_scheduler("linear", optimizer, args.warmup_steps, len(train_dataloader)* args.epoch) 
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.epoch*len(train_dataloader))
# num_training_steps 다시 확인하기

##################################################

args.model_config = model.config

tmp_metric = metric.compute(predictions=torch.Tensor([0,1]), references=torch.Tensor([0,1]))
#####?????

save_path='checkpoints/'+args.result_path+"_"+args.image_classification_dataset
os.makedirs(save_path,exist_ok=True)

print("Model Type:", model_type)
print("Dataset Name:", dataset_name)

step_cnt=0

for E in range(1, args.epoch+1):
    model.train()
    
    
    losses = []
    dl = tqdm(train_dataloader)
    for batches in dl:
        step_cnt += 1

        for idx in batches.keys():
            batches[idx] = batches[idx].to(device)

        # print(batches)
        # print(batches["pixel_values"].shape) # torch.Size([bs, 3, 224, 224])
        # print(batches["labels"].shape) # torch.Size([bs])        
        batches["interpolate_pos_encoding"]=True
        
        out = model(**batches)
        #print(out.logits.shape) # torch.Size([bs, num_labels])
        
        out.loss.backward()
        losses.append(out.loss.item())
        
        # if step_cnt == args.accumulate_step:
        if True:
            #step_cnt = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        dl.set_description("loss="+str(out.loss.item()))

    # print("train_loss = {}".format(sum(losses)/len(losses)))
    
    ##########
        if step_cnt % eval_step==0:
            model.eval()

            losses = []
            # best_dev_score = 0
            # best_test_score = 0

            with torch.no_grad():
                for batches in tqdm(val_dataloader):
                    for idx in batches.keys():
                        batches[idx] = batches[idx].to(device)
                    batches["interpolate_pos_encoding"]=True
                    
                    out = model(**batches)

                    losses.append(out.loss.item())
                    metric.add_batch(predictions=torch.argmax(out.logits, dim=-1), references=batches["labels"])
                    # pred_list.append(torch.argmax(out.logits, dim=-1))
                    # label_list.append(batches["labels"])
                    
                final_score = metric.compute()

                print("dev_loss = {}".format(sum(losses)/len(losses)))
                print("dev", final_score)

                change_score_name = dict()
                for i,j in final_score.items():
                        change_score_name["{}_dev_{}".format(args.image_classification_dataset, i)] = j
                # change_score_name["{}_dev_{}".format(task, "acc")] = sum(pred_list == label_list)/pred_list.size()[0]
                change_score_name["epoch"] = E ##### 균엽이 코드에서 왜 +1 인지 확인하기
                change_score_name["step"] = step_cnt ##### 균엽이 코드에서 왜 +1 인지 확인하기
                change_score_name["lr"] = optimizer.param_groups[0]["lr"]



                for batches in tqdm(test_dataloader):
                        for idx in batches.keys():
                            batches[idx] = batches[idx].to(device)
                        batches["interpolate_pos_encoding"]=True
                        
                        out = model(**batches)

                        losses.append(out.loss.item()) 
                        metric.add_batch(predictions=torch.argmax(out.logits, dim=-1), references=batches["labels"])
                    
                final_score = metric.compute()      

                print("test", final_score)
                for i,j in final_score.items():
                    change_score_name["{}_test_{}".format(args.image_classification_dataset, i)] = j
                change_score_name["epoch"] = E ##### 균엽이 코드에서 왜 +1 인지 확인하기

            model_name="model_"+str(step_cnt)+".bin"
            torch.save(model.state_dict(), os.path.join(save_path, model_name))
            