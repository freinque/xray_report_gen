"""
methods intended for report+image multimodal annotation

adapted from
https://github.com/zhangfaen/finetune-Qwen2-VL/
"""

import torch
import json
import os
from functools import partial
from itertools import islice

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from xray_report_gen import utils
from xray_report_gen.config import DATA_PATH, MODEL_PATH, REPORT_GENERATION_PROMPT, OS_MODEL_NAME_1, OS_MODEL_NAME_2

utils.set_api_keys()
import os
print(len(os.environ["HUGGINGFACEHUB_API_TOKEN"] ))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_name(version: int) -> str:
    if version == 1:
        return OS_MODEL_NAME_1
    elif version == 2:
        return OS_MODEL_NAME_2
    else:
        return OS_MODEL_NAME_1


class DataSet(Dataset):  # for toy demo
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)
        for p in self.data:
            p['messages'][0]['content'] = REPORT_GENERATION_PROMPT

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class FinetuneDataSet(Dataset):  # for toy demo
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as f:
            self.data = json.load(f)
        for p in self.data:
            p['messages'][0]['content'] = REPORT_GENERATION_PROMPT

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def write_chat_template(processor, output_dir):
    '''
    ***Note**

    We should have not had this function, as normal processor.save_pretrained(output_dir) would save chat_template.json file.
    However, on 2024/09/05, I think a commit introduced a bug to "huggingface/transformers", which caused the chat_template.json file not to be saved.
    See the below commit, src/transformers/processing_utils.py line 393, this commit avoided chat_template.json to be saved.
    https://github.com/huggingface/transformers/commit/43df47d8e78238021a4273746fc469336f948314#diff-6505546ec5a9ab74b2ce6511681dd31194eb91e9fa3ce26282e487a5e61f9356

    To walk around that bug, we need manually save the chat_template.json file.

    I hope this bug will be fixed soon and I can remove this function then.
    '''
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)
        print.info(f"chat template saved in {output_chat_template_file}")


def find_assistant_content_sublist_indexes(l):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i + 1] == 77091 and l[i + 2] == 198:
            start_indexes.append(i + 3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i + 3, len(l) - 1):
                if l[j] == 151645 and l[j + 1] == 198:
                    end_indexes.append(
                        j + 2)  # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


def collate_fn(batch, processor, device, inference=False):
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant")
    # [151644, 77091]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>")
    # [151645]
    messages = [m['messages'] for m in batch]
    ids = [m['id'] for m in batch]
    if inference:
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    else:
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = utils.process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    if inference:
        return inputs, labels_ids, ids
    else:
        return inputs, labels_ids

def run_inference(dataset, n=1000, version=1):
    """Run inference on images using a vision-language model."""

    model_name = get_model_name(version)
    print('running inference using : {}'.format(model_name))
    if "Qwen" in model_name:

        print('running inference on dataset: {}'.format(dataset))
        # from https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/main/README.md
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype='auto', device_map="auto"
        )
        # default processer
        processor = AutoProcessor.from_pretrained(model_name, padding_side="left")

        inference_loader = DataLoader(
            DataSet(dataset),
            batch_size=1,
            collate_fn=partial(collate_fn, processor=processor, device=device, inference=True)
        )
        generated_reports = []
        ids_list = []
        for batch in islice(inference_loader, n):
            inputs, labels, ids = batch

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=300)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            generated_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # print the generated text
            print('generated text for {ids}: {generated_text}'.format(ids=ids, generated_text=generated_text))

            ids_list.extend(ids)
            generated_reports.extend(generated_text)

        return ids_list, generated_reports

    else:
        # from https://huggingface.co/allenai/MolmoE-1B-0924
        # TODO
        generated_reports = []

    return generated_reports


