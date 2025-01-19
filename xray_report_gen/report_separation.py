import os
import openai

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# method 1: using ChatGPT to design a first prompt naively
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
PROMPT_1 = """
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

**Input:**
{report}

### Expected Output:
{{ "lung": "...", "heart": "...", "mediastinal": "...", "bone": "...", "others": "..." }}
"""

# TODO better names
def separate_report(report, prompt=PROMPT_1):

    try:
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=prompt.format(report=report),
            max_tokens=500,
            temperature=0.0
        )
        separated_report = response.choices[0].text.strip()
    except Exception as e:
        print(f"Error processing report: {e}")
        separated_report = "None"
        pass

    return separated_report


def separate_reports(reports, prompt=PROMPT_1):
    results = []
    for report in reports:
        #try:
        result = separate_report(report, prompt=prompt)
        #except Exception as e:
        #    print(f"Error processing report: {e}")
        #    #raise HTTPException(status_code=500, detail=f"Error processing report: {e}")
        #    pass

        results.append(result)
