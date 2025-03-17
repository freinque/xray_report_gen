# General configuration
DATA_PATH = '/xray_report_gen/data/'
MODEL_PATH = '/xray_report_gen/data/models/'
IMAGES_DIR = '/xray_report_gen/data/images'
REGIONS = ['bone', 'heart', 'lung', 'mediastinal']

# Dataset paths
TRAIN_DATASET = "/xray_report_gen/data/finetune_data_train.json"
TEST_DATASET = "/xray_report_gen/data/finetune_data_test.json"
VAL_DATASET = "/xray_report_gen/data/finetune_data_val.json"

# Dataset sizes
TRAIN_DATASET_N = 50
TEST_DATASET_N = 50
VAL_DATASET_N = 2500

# Training configuration
DEVICE = "cuda"
EPOCHS = 1
NUM_ACCUMULATION_STEPS = 5
SAVE_FREQUENCY = 500
LEARNING_RATE = 1e-5

######################################################################################
# annotation generation prompt used on reports
# Constants from report_annotation.py
GPT_MODEL_NAME = "gpt-4o-mini"

REPORT_ANNOTATION_N = 50
REPORT_ANNOTATION_PROMPT_VERSIONS = [1, 2]
REPORT_ANNOTATION_BEST_PROMPT_VERSION = 1

# version 1, simpler prompt
REPORT_ANNOTATION_SYSTEM_PROMPT_1 = """
You are an advanced medical assistant designed to analyze radiology report findings. Your task is to categorize sentences from a chest X-Ray radiology report into four predefined anatomical regions: lung, heart, mediastinal, and bone. If a sentence cannot be confidently assigned to any of these regions, categorize it under others.

### Instructions:
1. Parse the input text sentence by sentence.
2. Assign each sentence to one of the following categories:
   - **lung**: Findings related to lungs, pulmonary vasculature, or pleura.
   - **heart**: Findings related to the cardiac silhouette or heart size.
   - **mediastinal**: Findings related to the mediastinum or its contours.
   - **bone**: Findings related to bony structures such as the spine, ribs, or other skeletal elements.
   - **others**: Findings or sentences that cannot be confidently classified under the above categories.
3. If multiple findings belong to the same category, concatenate them into a single string within that category in the output.
4. Format the output as a JSON object with the keys: "lung", "heart", "mediastinal", "bone", and "others".

Now, categorize the following radiology report findings:
"""

REPORT_ANNOTATION_USER_PROMPT_1 = """ 
**Input:**
{report}

### Expected Output:
{{ "lung": "...", "heart": "...", "mediastinal": "...", "bone": "...", "others": "..." }}
"""

# version 2, prompt that includes examples
REPORT_ANNOTATION_SYSTEM_PROMPT_2= """
You are a highly intelligent assistant specializing in processing radiology reports. Your task is to categorize sentences or findings from chest X-ray radiology reports into one of five categories: lung, heart, mediastinal, bone, and others. Follow these instructions:

    1) Read the input radiology report carefully.
    2) Assign each sentence or finding to the most appropriate category based on the anatomical region it describes:
        - lung: Findings related to the lungs or pulmonary structures.
        - heart: Findings related to the heart or cardiac silhouette.
        - mediastinal: Findings related to the mediastinum or mediastinal contours.
        - bone: Findings related to the bones, including the thoracic spine, ribs, or other bony structures.
        - others: Sentences that cannot be clearly assigned to the above categories.
    3) Provide the output in a structured JSON format, with the categories as keys and their corresponding findings as values. Leave the value empty if there are no findings for a specific category.

Input:

The cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size. The lungs are mildly hypoinflated but grossly clear of focal airspace disease, pneumothorax, or pleural effusion. There are mild degenerative endplate changes in the thoracic spine. There are no acute bony findings.

Expected Output:

{
  "lung": "The lungs are mildly hypoinflated but grossly clear of focal airspace disease, pneumothorax, or pleural effusion. Pulmonary vasculature are within normal limits in size.",
  "heart": "The cardiomediastinal silhouette is within normal limits in size.",
  "mediastinal": "",
  "bone": "Mild degenerative endplate changes in the thoracic spine. There are no acute bony findings.",
  "others": ""
}

Input:

The heart, pulmonary XXXX and mediastinum are within normal limits. There is no pleural effusion or pneumothorax. There is no focal air space opacity to suggest a pneumonia. There are degenerative changes of the thoracic spine. There is a calcified granuloma identified in the right suprahilar region. The aorta is mildly tortuous and ectatic. There is asymmetric right apical smooth pleural thickening. There are severe degenerative changes of the XXXX.

Expected Output:
{
    "lung": "No pleural effusion or pneumothorax. No focal air space opacity to suggest a pneumonia. Calcified granuloma in the right suprahilar region. Asymmetric right apical smooth pleural thickening.",
    "heart": "Cardiac contours within normal limits.",
    "bone": "Degenerative changes of the thoracic spine. Severe degenerative changes of the XXXX.",
    "mediastinal": "Mediastinal contours within normal limits. Aorta is mildly tortuous and ectatic.",
    "others": ""
}

Now process the following radiology report and provide the categorized findings in the same JSON format:

"""

REPORT_ANNOTATION_USER_PROMPT_2 = '{report}'

######################################################################################
# annotation generation prompt used on images+reports
REPORT_IMAGE_ANNOTATION_GENERATION_PROMPT = """
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

REPORT_IMAGE_ANNOTATION_DEFAULT_MODEL_VERSION = 1

# Constants from rep_im_annotation.py
REPORT_IMAGE_ANNOTATION_BATCH_SIZE = 1
REPORT_IMAGE_ANNOTATION_OS_MODEL_NAME_1 = "Qwen/Qwen2-VL-2B-Instruct"
REPORT_IMAGE_ANNOTATION_OS_MODEL_NAME_2 = MODEL_PATH + "Qwen/Qwen2-VL-2B-Instruct-finetuned/checkpoint_epoch_0_step_100"

# Constants from eval.py
EVAL_MODEL_NAME = "StanfordAIMI/GREEN-radllama2-7b"

# Constants from utils.py
PATH_DOCKER = "/xray_report_gen"
PATH_HOST = "/home/freinque/pycharm_projects/xray_report_gen"
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 / 4 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 / 4 * 28 * 28
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768 / 4
