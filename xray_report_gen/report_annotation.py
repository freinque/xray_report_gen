"""
methods intended for report annotation using OpenAI LLMs
configured as gpt-4o-mini
"""

import os
from openai import OpenAI

from xray_report_gen import utils
from .config import GPT_MODEL_NAME, REPORT_ANNOTATION_SYSTEM_PROMPT_1, REPORT_ANNOTATION_SYSTEM_PROMPT_2, REPORT_ANNOTATION_USER_PROMPT_1, REPORT_ANNOTATION_USER_PROMPT_2

utils.set_api_keys()

client = OpenAI()

def get_system_prompt(version: int) -> str:
    if version == 1:
        return REPORT_ANNOTATION_SYSTEM_PROMPT_1
    elif version == 2:
        return REPORT_ANNOTATION_SYSTEM_PROMPT_2
    else:
        return REPORT_ANNOTATION_SYSTEM_PROMPT_1

def get_user_prompt(version: int) -> str:
    if version == 1:
        return REPORT_ANNOTATION_USER_PROMPT_1
    elif version == 2:
        return REPORT_ANNOTATION_USER_PROMPT_2
    else:
        return REPORT_ANNOTATION_USER_PROMPT_1

def annotate_report(report, prompt_version=1):

    system_prompt = get_system_prompt(prompt_version)
    user_prompt = get_user_prompt(prompt_version)
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL_NAME,
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


def annotate_reports(reports, prompt_version=1):
    results = []
    for report in reports:
        #try:
        result = annotate_report(report, prompt_version=prompt_version)
        #except Exception as e:
        #    print(f"Error processing report: {e}")
        #    #raise HTTPException(status_code=500, detail=f"Error processing report: {e}")
        #    pass

        results.append(result)

    return results
