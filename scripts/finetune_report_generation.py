import os
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM, AutoModel, Qwen2VLForConditionalGeneration, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments

from datasets import load_dataset

MODEL_PATH = '/xray_report_gen/data/models/'
DATA_PATH = '/xray_report_gen/data/'

MODEL_NAME_1 = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME_2 = 'allenai/Molmo-7B-D-0924' #'allenai/MolmoE-1B-0924'
REPORT_GENERATION_PROMPT = "Provide a medical analysis of the following image."

model_name = MODEL_NAME_1

def preprocess_data(example):
    image = Image.open(example["image_path"])
    # Add system prompt to describe the task
    system_prompt = REPORT_GENERATION_PROMPT
    text_input = f"{system_prompt}\n{example['original_report']}"
    inputs = processor(
        images=image,
        text=text_input,
        padding=True,
        return_tensors="pt"
    )
    #import pdb; pdb.set_trace()
    return inputs

def preprocess_data(example):
    image = Image.open(example["image_path"]).convert("RGB").resize((224, 224))
    system_prompt = REPORT_GENERATION_PROMPT
    text_input = f"{system_prompt}\n{example['original_report']}"
    inputs = processor(
        images=image,
        text=text_input,
        padding=True,
        truncation=True,
        return_tensors="np"
    )
    # Debug pixel_values
    print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    import pdb; pdb.set_trace()
    return inputs#{
#        "input_ids": inputs["input_ids"][0],
#        "attention_mask": inputs["attention_mask"][0],
#        "pixel_values": inputs["pixel_values"][0],
#        "labels": inputs["input_ids"][0],
#    }

if model_name == MODEL_NAME_1:

    processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=512*28*28, padding_side="right")

    dataset = load_dataset('json',  data_files={"train": os.path.join(DATA_PATH, 'finetune_data_train.json'),
                                                      "test": os.path.join(DATA_PATH, 'finetune_data_test.json'),
                                                      "validation": os.path.join(DATA_PATH, 'finetune_data_val.json')},
                           streaming=True)
    processed_dataset = dataset.map(preprocess_data, remove_columns=dataset["train"].column_names)

    #model = AutoModelForCausalLM.from_pretrained(model_name)
    #model = AutoModel.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=5,
        max_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="../logs",
        report_to="tensorboard",  # or "wandb"
        save_total_limit=2,
        fp16=True,  # Use mixed precision for efficiency
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['test'],
        tokenizer=processor,
    )

    trainer.train()

    trainer.evaluate(processed_dataset['test'])

    model.save_pretrained("fine_tuned_"+model_name)
    processor.save_pretrained("fine_tuned_"+model_name)


