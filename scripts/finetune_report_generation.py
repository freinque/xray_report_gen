import os
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

from datasets import load_dataset

MODEL_PATH = '..data/models/'
DATA_PATH = '..data/'

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
    return inputs


if model_name == MODEL_NAME_1:

    processor = AutoProcessor.from_pretrained(model_name)

    train_dataset = load_dataset(os.path.join(DATA_PATH, 'finetune_data_train.json'))
    processed_train_dataset = train_dataset.map(preprocess_data)
    test_dataset = load_dataset(os.path.join(DATA_PATH, 'finetune_data_test.json'))
    processed_test_dataset = test_dataset.map(preprocess_data)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=5,
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
        train_dataset=processed_train_dataset,
        eval_dataset=processed_test_dataset,
        tokenizer=processor,
    )

    trainer.train()

    trainer.evaluate(processed_test_dataset)

    model.save_pretrained("fine_tuned_"+model_name)
    processor.save_pretrained("fine_tuned_"+model_name)


