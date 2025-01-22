
# X-Ray report generation demo

We use a dataset of 2900+ chest X-Ray images, report notes and a set of annotations from the public IU X-Ray dataset for model development and validation. This demo will tackle prompt engineering using a proprietary model (1) and 'efficient' fine-tuning of a multimodal opensource model (2).

**General approach**
- general evaluation methodology for comparing curated and ML-generated annotations (GREEN)
- 1: using a proprietary generative model on radiological finding notes. A small setup is made available to compare annotation results from different system prompts
- 2: using an opensource multimodal (image+text to text) model on images and finding notes to generate annotations.
- minimal documentation, report writing, etc. (proper testing, deployment, etc. will stay out of scope)

**General structure**
- python package 'xray_report_gen' that will accomplish the tasks invoked by a set of scripts
- very minimalistic fastapi app to eventually expose inference endpoints

## Environments and Requirements

To install requirements:

- on a gpu-enabled linux VM, install docker, gpu drivers (ex. aws ubuntu AMI on g4dn.8xlarge may be sufficient), clone repo (ex. in /home/ubuntu folder)

- build and run the various procedures in ./scripts as inducated below

```setup
docker build -t xray_app .
```

Then to run scripts, use the equivalent of
```
docker run -it --rm --gpus device=0 -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 <script_name>.py
```

Add hf_token.txt and oa_token.txt to base folder.

The data folder contains a partial version of https://paperswithcode.com/dataset/iu-x-ray / https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university/versions/1 (you will need to copy the /images folder from there).

## Dataset

The overall dataset contains 2900+ chest X-Ray images, a report for each set of images from an exam. For four predifined anatomical regions: lung, heart, mediastinal, and bone, annotations are provided on train and test slices. A validation set of around 300 patients has those annotations missing.

## Preprocessing

Running the data preprocessing code:

```bash
docker run -it --rm -p 8080:8080 -v /home/freinque/pycharm_projects/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 data_prep.py
```

Note: image folders with extra '-0001', etc. were trimmed prior to running that routine.

## 1) Annotating textual reports using gpt

Annotation of original reports, using gpt-4-mini (can be reconfigured).
```bash
docker run -it --rm -p 8080:8080 -v /home/freinque/pycharm_projects/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 annotate_reports.py <mode>
```
where mode in ['train', 'test', 'val'].

Evaluate annotations on train and test samples, using 2 prompts examples suggested by ChatGPT (defined in report_annotation.py).

```bash
docker run -it --rm --gpus device=0 -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 evaluate_report_annotation.py <mode>
```

The scores are inspected with the evaluate_annotations.ipynb notebook. The following results are found:


**average GREEN scores on sample (N=25) of training set**

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


**average GREEN scores on samples (N=25) of test set**

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


Prompt version 1 is then chosen, and annotations on the validation set are put in data/reports_annotations_val.csv.


## 2) Annotating using a multimodal model, fine-tuning it

In this task, we want to extract the same chest region information from an xray image followed by the textual findings instead, using a image+text to text model. The models compared are based on https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct (TODO add https://huggingface.co/allenai/Molmo-7B-D-0924).

The finetuning of the model can be done by running this command:

```train
ocker run -it --rm --gpus device=0 -p 8080:8080 -v /home/ubuntu/xray_report_gen/data/:/xray_report_gen/data -v /home/ubuntu/xray_report_gen/data/huggingface/:/root/.cache/huggingface xray_app python3 finetune_rep_im_annotation.py
```

That finetuning routine taken and adapted from https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py, uses the transformers library and QLoRA (peft) hoping that a 16G gpu memory will be sufficient.

## Inference

To infer using the base, and finetuned models on test and val cases, run this command:

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
model_version = 1 is Qwen/Qwen2-VL-2B-Instruct
and
model_version = 2 is Qwen/Qwen2-VL-2B-Instruct-finetuned
which was finetuned on the current dataset for a few hundred examples (before running out of memory...).

## Results (TBC)

Right off the bat, we notice that the base opensource multimodal model results may be better overall that those of gpt-4o-mini.

**Advantages the opensource multimodal option**
- leverages images that pure language models can't consider
- control over finetuning: we expect that finetuning will improve performance on the task at hand
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

More finetuning still has to be done on the multimodal option to assess its ability to acquire task-specific performance.

## TODOs

- Testing everything, reviewing
- Isolate general config
- General clean-up
- Improve scientific aspect, since this gives an technical experimental setup, mostly.
- Parallelize finetuning to allow for multi-gpu more significant finetuning.
- Set up inference endpoint for task 2, harden app for real-life usage

## Acknowledgements

Ref report template used:
https://github.com/JunMa11/MICCAI-Reproducibility-Checklist/blob/ebd879ec740962e37ed99eb0fee84cb4c96a4b8f/templates/MICCAI-Code-Checklist.md

green_score taken from https://stanford-aimi.github.io/green.html

finetuning script taken from https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py

some code snippets were modified from ChatGPT