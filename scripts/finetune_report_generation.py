import click
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, GenerationConfig
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
#from qwen_vl_utils import process_vision_info
import os
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM, AutoModel, Qwen2VLForConditionalGeneration, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments

from datasets import load_dataset

MODEL_PATH = '/xray_report_gen/data/models/'
DATA_PATH = '/xray_report_gen/data/'

from xray_report_gen import utils

utils.set_api_keys()
import os
print(len(os.environ["HUGGINGFACEHUB_API_TOKEN"] ))

DATA_PATH = '../data/'
MODEL_NAME_1 = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME_2 = 'allenai/Molmo-7B-D-0924' #'allenai/MolmoE-1B-0924'
MODEL_VERSION = 1

REPORT_GENERATION_PROMPT = """
You are an advanced medical assistant designed to analyze radiology report findings and chest x-ray images. Your task is to extract sentences from a chest X-Ray radiology report and a chest x-ray image into four predefined anatomical regions: lung, heart, mediastinal, and bone. If a finding cannot be confidently assigned to any of these regions, categorize it under others.

### Instructions:
1. Consider the input radiology report sentence by sentence, and the corresponding chest X-Ray image.
2. Extract findings for the following categories:
   - **lung**: Findings related to lungs, pulmonary vasculature, or pleura.
   - **heart**: Findings related to the cardiac silhouette or heart size.
   - **mediastinal**: Findings related to the mediastinum or its contours.
   - **bone**: Findings related to bony structures such as the spine, ribs, or other skeletal elements.
   - **others**: Findings or sentences that cannot be confidently classified under the above categories.
3. If multiple findings belong to the same category, concatenate them into a single string within that category in the output.
4. Format the output as a JSON object with the keys: "lung", "heart", "mediastinal", "bone", and "others". If no finding were found for a given category, the output value for its JSON key should be empty.

Now, analyze the following X-ray image and report, and generate findings organized into these categories.

**Input:**
{report}
"""

"""
### Expected Output:
{{ "lung": "...", "heart": "...", "mediastinal": "...", "bone": "...", "others": "..." }}
"""

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

gpu = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_name(version: int) -> str:
    if version == 1:
        return MODEL_NAME_1
    elif version == 2:
        return MODEL_NAME_2
    else:
        return MODEL_NAME_1


@click.command()
@click.option('--version', type=int, default=MODEL_VERSION)
def main(version):
    """Run inference on images using a vision-language model."""

    model_name = get_model_name(version)
    if model_name.startswith("Qwen"):

        # from https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/main/README.md
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        # default processer
        processor = AutoProcessor.from_pretrained(model_name)

        def preprocess_data(example):
            image = Image.open(example["image_path"]).resize(size=(512, 512))
            text_input = REPORT_GENERATION_PROMPT.format(report=example["original_report"])
            inputs = processor(
                images=image,
                text=text_input,
                padding=True,
                truncation=True,
                return_tensors="np"
            )
            inputs = inputs.to("cuda")
            # Debug pixel_values
            #print(f"Pixel values shape: {inputs['pixel_values'].shape}")
            return inputs

        dataset = load_dataset('json', data_files={"train": os.path.join(DATA_PATH, 'finetune_data_train.json'),
                                                   "test": os.path.join(DATA_PATH, 'finetune_data_test.json'),
                                                   "validation": os.path.join(DATA_PATH, 'finetune_data_val.json')},
                               streaming=True)

        processed_dataset = dataset.map(preprocess_data, remove_columns=dataset["train"].column_names)

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

        model.save_pretrained("fine_tuned_" + model_name)
        processor.save_pretrained("fine_tuned_" + model_name)


    else:
        # from https://huggingface.co/allenai/MolmoE-1B-0924

        # load the processor
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # process the image and text
        inputs = processor.process(
            images=images,
            text=REPORT_GENERATION_PROMPT
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=400, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        print(generated_text)


    return generated_reports

if __name__ == "__main__":
    main()







