
# X-Ray report annotation/generation demo

We use a dataset of 2900+ chest X-Ray images, report notes and annotations from the public IU X-Ray dataset for model development and validation. This repository aims to facilitate prompt engineering experimentation/evaluation (1) and 'efficient' fine-tuning of an opensource (multimodal) model (2).

**General approach**
- general evaluation methodology (GREEN) for comparing ML-generated x-ray exam annotations to curated ones
- Approach 1: generate annotations by invoking a proprietary text-only generative model on radiological finding notes. An experimental setup is made available for comparing annotation results from different (system) prompts
- Approach 2: generate annotations using an opensource multimodal (image+text to text) model on images and finding notes together
- Minimize compute costs, but having clear options to scale up. 

**General structure**
- python package 'xray_report_gen' that will accomplish the tasks invoked by a set of scripts
- very minimalistic fastapi app to eventually expose inference endpoints
- minimal configurability, documentation, report writing, etc. (proper testing, deployment, parametrization, etc. will stay out of scope)

## Environments and Requirements

- on a gpu-enabled linux VM, install docker, gpu drivers (ex. aws ubuntu AMI 'Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5.1 (Ubuntu 22.04) 20250112' on g4dn.8xlarge hardware is sufficient-ish), clone repo (ex. in /home/ubuntu folder)

- build and run the various procedures in ./scripts as indicated below:

```setup
docker build -t xray_app .
```

Then to run scripts, use the equivalent of
```
docker run -it --rm --gpus device=0 -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 <script_name>.py
```

- Add hf_token.txt and oa_token.txt to base folder.

- The data folder contains a partial version of https://paperswithcode.com/dataset/iu-x-ray / https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university/versions/1 (you will need to copy the /images folder from there).

## Dataset

The overall dataset contains 2900+ chest X-Ray images, a report for each set of images from an exam. For four predifined anatomical regions: lung, heart, mediastinal, and bone, annotations are provided on train and test slices. A validation set of around 300 patients has those annotations missing.

Base data files:

- `data/indiana_reports.csv`: Set of anonymized chest x-ray reports.
- `data/indiana_projections.csv`: Set of metadata associated with chest x-ray images.
- `data/annotation_quiz_all.json`: Set of 'curated' image annotations. Also defines what splits are going to be used (for training, validation and testing).

## Preprocessing

Running the data preprocessing script:

```bash
docker run -it --rm -p 8080:8080 -v /home/freinque/pycharm_projects/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 data_prep.py
```

Note: image folders with extra '-0001', etc. were trimmed prior to running that routine.

## Approach 1) Annotating textual reports using gpt

Annotation of original reports, using OpenAI's gpt-4-mini (can be reconfigured).

```bash
docker run -it --rm -p 8080:8080 -v /home/freinque/pycharm_projects/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 annotate_reports.py <split>
```
where split in ['train', 'test', 'val'].

Evaluate annotations on train and test samples, using 2 prompts examples defined in config.py.

```bash
docker run -it --rm --gpus device=0 -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 evaluate_report_annotation.py <split>
```
where split in ['train', 'test'].

The scores are inspected with the evaluate_annotations.ipynb notebook. The following results are found:


**average GREEN scores on sample (N=50) of training set**

prompt_version  | region    |  score   
:----:          |:----:     |:--------:|
1               | bone       | 0.430000 
1               | heart       | 0.956667    
1                | lung        | 0.961667    
1                | mediastinal  | 0.433333    
2               | bone           | 0.410000 
2                | heart          | 1.000000    
2                | lung           | 0.877000    
2                | mediastinal    | 0.510000    


**average GREEN scores on samples (N=50) of test set**

prompt_version  | region    |  score   
:----:          |:----:     |:--------:|
1               | bone       |  0.470000
1               | heart       | 0.930000    
1                | lung        |  0.961667   
1                | mediastinal  |  0.430000   
2               | bone           |  0.480000
2                | heart          | 0.960000    
2                | lung           |  0.902667  
2                | mediastinal    |  0.400000   


Prompt version 1 is then chosen, and annotations on the validation set are put in data/reports_annotations_val.csv.


## Approach 2) Annotating using a multimodal model, fine-tuning it using curated annotations

In this task, we want to extract the same chest region-specific information from an xray image followed by the textual findings instead, using an image+text to text model. The models compared are based on https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct (TODO add https://huggingface.co/allenai/Molmo-7B-D-0924).

The fine-tuning of the model can be done by running this command:

```train
docker run -it --rm --gpus device=0 -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 finetune_rep_im_annotation.py
```

That fine-tuning routine was adapted from https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py, the latter being based on the transformers library and QLoRA (via peft) hoping that a 16G gpu memory will be sufficient.

## Inference

To infer using the base, and fine-tuned models on test and val cases, run this command:

```bash
docker run -it --rm --gpus device=0 -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 annotate_rep_im.py <test/val> --version <version>
```
where mode in ['test', 'val']
and version in [1,2]

## Evaluation

To compute the multimodal evaluation metrics, run:

```bash
docker run -it --rm --gpus device=0 -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 evaluate_report_annotation.py test --multi 1
```

The estimations made on a sample of the test set are as follows:

**Average GREEN scores on samples (N=25) of test set**

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

(see evaluate_annotations.ipynb for more details)

where
model_version = 1 corresponds to the pretrained Qwen/Qwen2-VL-2B-Instruct (https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
and
model_version = 2 corresponds to Qwen/Qwen2-VL-2B-Instruct-finetuned
which was fine-tuned on the current dataset for a few hundred examples (before running out of memory...).

## Results, discussion (TBC)

At first sight, we notice that the base opensource multimodal model results may be better overall that those of gpt-4o-mini on reports only. However, **more fine-tuning still has to be done on the multimodal option to assess its ability to acquire large-scale, real-world, task-specific performance.**

**Advantages the opensource multimodal option**
- leverages images that pure language models can't consider
- control over fine-tuning: we expect that fine-tuning will improve performance on the task at hand
- better prompt engineering flexibility, customizability
- likely more cost-effective
- control over information privacy, security, governance
- portability

**Disadvantages of opensource multimodal option**
- much harder to set up and maintain. no viable serverless options found
- probably less effective than SOTA proprietary if used only on text
- dependency on high-quality training data
- lack of support, uncertain reliability
- security of dependencies


## TODOs

- Parallelize fine-tuning to allow for multi-gpu more significant fine-tuning.
- Testing key functionalities, reviewing
- Isolate general config from modules/scripts
- General clean-up, hardening
- Improve scientific aspect, since this exercise focuses mostly on establishing a technical experimental setup.
- Set up inference endpoint for task 2, harden fastapi app for real-life usage

## Acknowledgements

- Ref report template used:
https://github.com/JunMa11/MICCAI-Reproducibility-Checklist/blob/ebd879ec740962e37ed99eb0fee84cb4c96a4b8f/templates/MICCAI-Code-Checklist.md

- green_score taken from https://stanford-aimi.github.io/green.html

- fine-tuning script taken from https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py

- some code snippets were modified from ChatGPT