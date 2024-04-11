# Unit_Initalize_and_Weight_Activation

This repository is experiment code of {paper_name}

## Usage

#### For GLEU test  
!! experiment_name must include model name ["deberta","t5"]
``` 
python test_gleu.py \  
    --result_path {experiment_name} \    
    --glue_task {[cola, sst2, mrpc, stsb, qqp, mnli, mnli_matched, mnli_mismatched, qnli, rte, wnli]} \  
    --init_type {[unit, xavier]} \  
    --act_type {[gelu, midgelu]} \  
    --add_position {[befdot, afterffnn, both]}  
```

if you wand testing mrpc with our proposed method and additional layer is inserted in multi-haed attention position

``` 
python test_gleu.py \  
    --result_path deberta_test \
    --glue_task mrpc \  
    --init_type unit \  
    --act_type midgelu \  
    --add_position befdot
```

#### For Text Generation

!! experiment_name must include model name ["t5"]
``` 
python test_generation.py \  
    --result_path {experiment_name} \    
    --generation_task {[cnndm, wmt_en_ro]} \  
    --init_type {[unit, xavier]} \  
    --act_type {[gelu, midgelu]} \  
    --add_position {[befdot, afterffnn, both]}  
```

#### For Image Classificaiton

!! experiment_name must include model name ["vit"]
``` 
python test_image_classification.py \  
    --result_path {experiment_name} \    
    --image_classification_dataset {["cifar100", "imagenet-1k"]} \  
    --init_type {[unit, xavier]} \  
    --act_type {[gelu, midgelu]} \  
    --add_position {[befdot, afterffnn, both]}  
```
