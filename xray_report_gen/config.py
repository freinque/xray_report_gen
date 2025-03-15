# General configuration
DATA_PATH = '/xray_report_gen/data/'
MODEL_PATH = '/xray_report_gen/data/models/'
IMAGES_DIR = '/xray_report_gen/data/images'
MODEL_VERSION = 1
N = 25
PROMPT_VERSIONS = [1, 2]
BEST_PROMPT_VERSION = 1
REGIONS = ['bone', 'heart', 'lung', 'mediastinal']

# Dataset paths
TRAIN_DATASET = "/xray_report_gen/data/finetune_data_train.json"
TEST_DATASET = "/xray_report_gen/data/finetune_data_test.json"
VAL_DATASET = "/xray_report_gen/data/finetune_data_val.json"

# Dataset sizes
TRAIN_DATASET_N = 25
TEST_DATASET_N = 25
VAL_DATASET_N = 2500

# Training configuration
DEVICE = "cuda"
EPOCHS = 1
NUM_ACCUMULATION_STEPS = 5
SAVE_FREQUENCY = 100
LEARNING_RATE = 1e-5

# Report generation prompt
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