import argparse
import os

import random
import torch
import numpy as np

AVAIL_BATCH = {
    "rte" : 8
}


def seed_fix(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def parse_args():
    parser = argparse.ArgumentParser(description="various_attention")

    parser.add_argument(
        "--for_cv", default=False, action="store_true" 
    )
    parser.add_argument(
        "--result_path", type=str, required=True
    )
    parser.add_argument(
        "--model_load_path", type=str, default=None, required=False
    )
    parser.add_argument(
        "--gpu", type=int, default=0, required=False
    )
    parser.add_argument(
        "--epoch", type=int, default=15, required=False
    )
    parser.add_argument(
        "--gen_train_step", type=int, default=15000, required=False
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, required=False
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, required=False
    )
    parser.add_argument(
        "--seed", type=int, default=1234, required=False
    )
    parser.add_argument(
        "--accumulate_step", type=int, default=16, required=False
    )
    parser.add_argument(
        "--data_path", type=str, default="datasets", required=False
    )
    
    
    #optimizer & scheduler detail
    parser.add_argument(
        "--beta1", type=float, default=0.9, required=False
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, required=False
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, required=False
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, required=False
    )
    parser.add_argument(
        "--eps", type=float, default=1e-6, required=False
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500, required=False
    )
    
    
    #transformer_unit_init
    parser.add_argument(
        "--no_add_linear", default=False, action="store_true"
    )
    parser.add_argument(
        "--add_linear_num", type=int, default=None
    )
    parser.add_argument(
        "--add_linear_layer", type=str, default=None
    )
    parser.add_argument(
        "--init_type", type=str, default="unit", choices=["unit", "he"]
    )
    parser.add_argument(
        "--head_indi", default=True, action="store_true"
    )
    parser.add_argument(
        "--add_position", default="befdot", choices=["befdot", "aftffnn1", "aftffnn2", "afterffnn", "both"]
    )
    parser.add_argument(
        "--act_type", type=str, default=None, choices=["midgelu", "gelu", "midrelu", "relu"]
    )
    parser.add_argument(
        "--adapter", default=False, action="store_true"
    )
    
    #glue
    GLUE_TASKS = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "mnli_matched", "mnli_mismatched", "qnli", "rte", "wnli"]
    parser.add_argument(
        "--glue_task", type=str, default="mrpc", choices=GLUE_TASKS
    )
    
    #cv-image classification
    IMAGE_CLASSIFICATION_DATASETS = ["cifar10", "cifar100", "imagenet-1k"]
    parser.add_argument(
        "--image_classification_dataset", type=str, default="cifar10", choices=IMAGE_CLASSIFICATION_DATASETS
    )
    parser.add_argument(
        "--image_size", type=int, default=384
    )
    parser.add_argument(
        "--cifar_eval_step", type=int, default=500
    )
    parser.add_argument(
        "--imagenet_eval_step", type=int, default=50000
    )

    
    #text_generation
    parser.add_argument(
        "--generation_task", type=str, default="mrpc", choices=["cnndm", "wmt_en_ro"]
    )
    parser.add_argument(
        "--generate_full_batch", type=int, default=128
    )


    parser.add_argument(
        "--logging_step", type=int, default=10000,
    )
    parser.add_argument(
        "--dev", default=False, action="store_true"
    )
    
    args = parser.parse_args()
    return args

def tf_make_result_path(args):
    result_path = [args.result_path]
    
    if args.for_cv==False:
        try:
            result_path.append(args.task)
        except:
            result_path.append(args.glue_task)
    
    if args.for_cv:
        result_path.append(str(args.learning_rate))
    
    if args.no_add_linear:
        result_path.append("baseline")
        return "_".join(result_path)
    
    result_path.append(args.add_position)

    result_path.append(args.init_type)
    
    if args.add_linear_num is not None:
        if args.add_linear_num > 0:
            result_path.append("bottom"+str(args.add_linear_num))
        else:
            result_path.append("top"+str(args.add_linear_num))
            
    if args.add_linear_layer is not None:
        result_path.append("layer"+str(args.add_linear_layer))
    
    if args.head_indi:
        result_path.append("indi")
        
    if args.act_type is None:
        result_path.append("no_act")
    else:
        result_path.append(args.act_type)

        
    if args.adapter:
        result_path.append("adapter")
        
    if args.dev:
        result_path.append("dev")
        

        
    return '_'.join(result_path)



def gen_make_result_path(args):
    result_path = [args.result_path]
    
    if args.for_cv==False:
        try:
            result_path.append(args.task)
        except:
            result_path.append(args.generation_task)
    
    if args.for_cv:
        result_path.append(str(args.learning_rate))
    
    if args.no_add_linear:
        result_path.append("baseline")
        return "_".join(result_path)
    
    result_path.append(args.add_position)

    result_path.append(args.init_type)
    
    if args.add_linear_num is not None:
        if args.add_linear_num > 0:
            result_path.append("bottom"+str(args.add_linear_num))
        else:
            result_path.append("top"+str(args.add_linear_num))
            
    if args.add_linear_layer is not None:
        result_path.append("layer"+str(args.add_linear_layer))
    
    if args.head_indi:
        result_path.append("indi")
        
    if args.act_type is None:
        result_path.append("no_act")
    else:
        result_path.append(args.act_type)

        
    if args.adapter:
        result_path.append("adapter")
        
    if args.dev:
        result_path.append("dev")
        

        
    return '_'.join(result_path)
