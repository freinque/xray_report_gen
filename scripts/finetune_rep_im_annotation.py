"""
adapted from
https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py
"""

import torch
import json
import os

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from functools import partial

from peft import LoraConfig, get_peft_model
from bitsandbytes.optim import Adam8bit

from xray_report_gen.utils import process_vision_info, init_logger, get_logger
from xray_report_gen import utils
from xray_report_gen import rep_im_annotation

utils.set_api_keys()
print(len(os.environ["HUGGINGFACEHUB_API_TOKEN"] ))

MODEL_PATH = '/xray_report_gen/data/models/'
DATA_PATH = '/xray_report_gen/data/'
TRAINING_DATASET = "/xray_report_gen/data/finetune_data_train.json"

output_dir = os.path.join(MODEL_PATH, "Qwen/Qwen2-VL-2B-Instruct-finetuned")
init_logger(output_dir)
logger = get_logger()

device = "cuda"

REPORT_GENERATION_PROMPT = """
You are an advanced medical assistant designed to analyze radiology report findings and chest x-ray images. Your task is to extract sentences from a chest X-Ray radiology report and a chest x-ray images into four predefined anatomical regions: lung, heart, mediastinal, and bone. If a finding cannot be confidently assigned to any of these regions, categorize it under others.

### Instructions:
1. Consider the input radiology report sentence by sentence, and the corresponding chest X-Ray images.
2. Extract findings for the following categories:
   - **lung**: Findings related to lungs, pulmonary vasculature, or pleura.
   - **heart**: Findings related to the cardiac silhouette or heart size.
   - **mediastinal**: Findings related to the mediastinum or its contours.
   - **bone**: Findings related to bony structures such as the spine, ribs, or other skeletal elements.
   - **others**: Findings or sentences that cannot be confidently classified under the above categories.
3. If multiple findings belong to the same category, concatenate them into a single string within that category in the output.
4. Format the output as a JSON object with the keys: "lung", "heart", "mediastinal", "bone", and "others". If no finding were found for a given category, the output value for its JSON key should be empty.

Now, analyze the following X-ray images and report, and generate findings organized into these categories.

**Input:**
"""



def train():
    # Load the model on the available device(s)
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-2B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # ** WARNING ** When run below line , we got below warning message:
    #   Unrecognized keys in `rope_scaling` for 'rope_type'='default': {'mrope_section'}"
    # It is a issue, see https://github.com/huggingface/transformers/issues/33401
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, #attn_implementation="flash_attention_2", #mod
        device_map="auto"
    ) #mod

    #mod
    lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

    model = get_peft_model(model, lora_config)
    #model.gradient_checkpointing_enable() #mod

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=256 * 28 * 28,
                                              max_pixels=512 * 600, padding_side="right")

    train_loader = DataLoader(
        rep_im_annotation.FinetuneDataSet(TRAINING_DATASET),
        batch_size=1,
        collate_fn=partial(rep_im_annotation.collate_fn, processor=processor, device=device)
    )

    model.train()
    epochs = 1

    optimizer = Adam8bit(model.parameters(), lr=1e-5) # mod
    NUM_ACCUMULATION_STEPS = 5
    # mod
    # Directory to save checkpoints
    os.makedirs(MODEL_PATH, exist_ok=True)
    SAVE_FREQUENCY = 100

    for epoch in range(epochs):
        accumulated_avg_loss = 0
        steps = 0
        for batch in train_loader:
            steps += 1
            inputs, labels = batch
            # import pdb
            # pdb.set_trace()
            outputs = model(**inputs, labels=labels)

            loss = outputs.loss / NUM_ACCUMULATION_STEPS
            accumulated_avg_loss += loss.item()
            loss.backward()

            if steps % NUM_ACCUMULATION_STEPS == 0:
                logger.info(
                    f"Batch {steps} of epoch {epoch + 1}/{epochs}, average training loss of previous {NUM_ACCUMULATION_STEPS} batches: {accumulated_avg_loss}")
                accumulated_avg_loss = 0
                optimizer.step()
                optimizer.zero_grad()
                # Save model checkpoint every `SAVE_FREQUENCY` steps
            if steps % SAVE_FREQUENCY == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}_step_{steps}")
                model.save_pretrained(checkpoint_path)
                processor.save_pretrained(checkpoint_path)
                logger.info(f"Model checkpoint saved at step {steps} in epoch {epoch + 1}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    write_chat_template(processor, output_dir)


if __name__ == "__main__":
    train()
