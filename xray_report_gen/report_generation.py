import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, GenerationConfig
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
#from qwen_vl_utils import process_vision_info

from xray_report_gen import utils

utils.set_api_keys()
import os
print(len(os.environ["HUGGINGFACEHUB_API_TOKEN"] ))

DATA_PATH = '../data/'
MODEL_NAME_1 = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME_2 = 'allenai/Molmo-7B-D-0924' #'allenai/MolmoE-1B-0924'
REPORT_GENERATION_PROMPT = """
You are a radiology assistant. Your task is to analyze the provided X-ray image and extract findings related to four predefined anatomical regions: lung, heart, mediastinal, and bone. If you cannot assign a finding to any of these regions, classify it under others.

Here is an example of the expected output format:
{
  "lung": "Lungs are mildly hypoinflated but grossly clear of focal airspace disease, pneumothorax, or pleural effusion.",
  "heart": "Cardiac silhouette within normal limits in size.",
  "mediastinal": "Mediastinal contours within normal limits in size.",
  "bone": "Mild degenerative endplate changes in the thoracic spine. No acute bony findings.",
  "others": ""
}

Now, analyze the following X-ray image and generate findings organized into these categories.
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

def run_inference(images, task_description=REPORT_GENERATION_PROMPT, model_version=1):
    """Run inference on images using a vision-language model."""

    model_name = get_model_name(model_version)
    if model_name.startswith("Qwen"):

        print('running inference on images: {}'.format(images))
        # from https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/main/README.md
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        # default processer
        processor = AutoProcessor.from_pretrained(model_name)

        for img in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": task_description}],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # image_inputs, video_inputs = images, None
            # added
            # texts = [
            #    processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            #    for message in messages
            # ]

            inputs = processor(
                text=[text],
                images=[img],
                # videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            generated_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # print the generated text
            print('generated text for {img}: {generated_text}'.format(img=img, generated_text=generated_text))

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
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        print(generated_text)


    return generated_text


