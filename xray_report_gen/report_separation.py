import os
from openai import OpenAI

from xray_report_gen import utils

utils.set_api_keys()

# Set your OpenAI API key
#openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# method 1 and 2: using ChatGPT to design a first prompt naively
"""
Please write a prompt to use GPT4 to separate radiology report findings into the four predifined anatomical regions: lung, heart, mediastinal, and bone. If the model cannot assign the sentence to any anatomical region, please put it in others. Here is an example:

Input:
------
a report of a typical chest X-Ray radiology findings.
The cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size. The lungs are mildly hypoinflated but grossly clear of focal airspace disease, pneumothorax, or pleural effusion. There are mild degenerative endplate changes in the thoracic spine. There are no acute bony findings.
-------
Expected output:
--------
{
"lung": "Lungs are mildly hypoinflated but grossly clear of focal
airspace disease, pneumothorax, or pleural effusion. Pulmonary vasculature
are within normal limits in size.",
"heart": "Cardiac silhouette within normal limits in size.",
"mediastinal": "Mediastinal contours within normal limits in size.",
"bone": "Mild degenerative endplate changes in the thoracic spine. No
acute bony findings.",
"others": ""
}
------

"""
SYSTEM_PROMPT_1 = """
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

USER_PROMPT_1 = """ 
**Input:**
{report}

### Expected Output:
{{ "lung": "...", "heart": "...", "mediastinal": "...", "bone": "...", "others": "..." }}
"""

SYSTEM_PROMPT_2= """
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

Now process the following radiology report and provide the categorized findings in the same JSON format:

"""

USER_PROMPT_2 = '{report}'

def get_system_prompt(version: int) -> str:
    if version == 1:
        return SYSTEM_PROMPT_1
    elif version == 2:
        return SYSTEM_PROMPT_2
    else:
        return SYSTEM_PROMPT_1

def get_user_prompt(version: int) -> str:
    if version == 1:
        return USER_PROMPT_1
    elif version == 2:
        return USER_PROMPT_2
    else:
        return USER_PROMPT_1

# TODO better names
def separate_report(report, prompt_version=1):

    system_prompt = get_system_prompt(prompt_version)
    user_prompt = get_user_prompt(prompt_version)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": system_prompt},
                {"role": "user", "content": user_prompt.format(report=report)}
            ],
            max_tokens=500,
            temperature=0.0
        )
        separated_report = response.choices[0].message.content.strip()
        print('got separated report:', separated_report)
    except Exception as e:
        print(f"Error processing report: {e}")
        separated_report = "None"
        pass

    return separated_report


def separate_reports(reports, prompt_version=1):
    results = []
    for report in reports:
        #try:
        result = separate_report(report, prompt_version=prompt_version)
        #except Exception as e:
        #    print(f"Error processing report: {e}")
        #    #raise HTTPException(status_code=500, detail=f"Error processing report: {e}")
        #    pass

        results.append(result)

    return results
