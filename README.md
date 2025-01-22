
# X-Ray report generation demo

Provide 2900+ chest X-Ray images, report notes and a set of annotations from the public IU X-Ray dataset for model development and validation. This demo will tackle prompt engineering (1) and efficient fine-tuning (2).

General plan
- 1: building eval (GREEN), first prompt on batch, scripting for iteration, wrapping in fastapi call, pause
- 2: mostly recycling eval from task 1, importing models, tracking, script training, find infra, run and 
- documenting, report writing, proper testing will stay out of scope

overall structure
- python package 'xray_report_gen' that will accomplish the tasks
- very minimalistic fastapi app to expose and experiment


## Environments and Requirements

To install requirements:

- install docker, gpu drivers (ex. aws ubuntu AMI on g4dn.4xlarge may be sufficient)
clone repo

- make GREEN model available
    https://endpoints.huggingface.co/ (did not work)
    host the model on at least g4dn EC2 instance

- build and run the various procedures in ./scripts

```setup
docker build -t xray_app .
```

Then to run use the equivalent of
```
docker run -it --rm --gpus all -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 finetune_report_generation.py
```
or 
```
docker run -it --rm --gpus all -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 finetune_report_generation.py
```


## Dataset

- A link to download the data (if publicly available)
- A description about how to prepare the data (e.g., folder structures)

## Preprocessing

Running the data preprocessing code:

```bash
docker run -it --rm -p 8080:8080 -v /home/freinque/pycharm_projects/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 data_prep.py
```
Image folders with extra '-0001', etc. were trimmed

## 1) Annotating reports

Annotation of original reports, using gpt-4-mini (can be reconfigured).
```bash
docker run -it --rm -p 8080:8080 -v /home/freinque/pycharm_projects/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 annotate_reports.py
```
Evaluate annotations on train and test samples, using 2 prompts examples suggested by ChatGPT (defined in report_annotation.py).

```bash
docker run -it --rm --gpus all -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 evaluate_report_annotation.py
```
The scores are inspected with the evaluate_annotations.ipynb notebook. The following results are found:

average GREEN scores on sample of training set

prompt_version  | region    |  score   
:----:          |:----:     |:--------:|
1               | bone       | 0.800000 
1               | heart       | 0.900000    
1                | lung        | 1.000000    
1                | mediastinal  | 0.300000    
2               | bone           | 0.800000 
2                | heart          | 0.900000    
2                | lung           | 0.866667    
2                | mediastinal    | 0.300000    

average GREEN scores on samples of test set

prompt_version  | region    |  score   
:----:          |:----:     |:--------:|
1               | bone       |  0.520000
1               | heart       | 0.900000    
1                | lung        |  0.955333   
1                | mediastinal  |  0.440000   
2               | bone           |  0.512000
2                | heart          | 0.920000    
2                | lung           |  0.908667   
2                | mediastinal    |  0.440000   


Prompt version 1 is then chosen, and annotations on the validation set are put in data/reports_annotations_val.csv


## 2) Fine-tuning multimodal model

In this task, we want to extract the same torax region information from an xray image instead, using a image+text to text model. The two models compared are https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct (and potentially https://huggingface.co/allenai/Molmo-7B-D-0924).

To train the model(s) in the paper, run this command:

```train
ocker run -it --rm --gpus all -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 finetune_rep_im_annotation.py
```

That finetuning routine taken and adapted from https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py, uses the transformers library and QLoRA (peft).

## Inference

To infer the validation cases, run this command:

```bash
docker run -it --rm --gpus device=0 -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 annotate_rep_im.py <test/val> --version <version>
```



## Evaluation

To compute the multimodal evaluation metrics, run:

```bash
docker run -it --rm --gpus device=0 -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 evaluate_report_annotation.py test --multi 1
```

Average GREEN scores on samples of test set (see evaluate_annotations.ipynb)

model_version  | region    |  score   
:----:          |:----:     |:--------:|
1               | bone       |  0.44
1               | heart       | 0.92   
1                | lung        |  1.0  
1                | mediastinal  |  0.84   
2               | bone           |  0.44
2                | heart          | 0.92    
2                | lung           |  1.0  
2                | mediastinal    |  0.84   

where
model_version = 1 is 
Qwen/Qwen2-VL-2B-Instruct
and
model_version = 2 is
Qwen/Qwen2-VL-2B-Instruct-finetuned

## Results

Our method achieves the following performance on [Brain Tumor Segmentation (BraTS) Challenge](https://www.med.upenn.edu/cbica/brats2020/)

| Model name       |  DICE  | 95% Hausdorff Distance |
| ---------------- | :----: | :--------------------: |
| My awesome model | 90.68% |         32.71          |
| My awesome model | 90.68% |         32.71          |



## TODOs
Testing everything, reviewing
Isolate general config
General clean-up
Improve scientific, aspect, since this gives an experimental setup, mostly.
Parallelize finetunings.
Inference endpoint for task 2

## Acknowledgement

Ref report template used:
https://github.com/JunMa11/MICCAI-Reproducibility-Checklist/blob/ebd879ec740962e37ed99eb0fee84cb4c96a4b8f/templates/MICCAI-Code-Checklist.md

green_score taken from https://stanford-aimi.github.io/green.html

finetuning script taken from https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py

some code snippets were modified from ChatGPT