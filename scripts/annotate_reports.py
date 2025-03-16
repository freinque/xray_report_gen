"""
script intented to invoke report LLM annotation inference on preprocessed datasets
"""

import os
import pandas as pd
import click

from xray_report_gen import report_annotation
from xray_report_gen.config import DATA_PATH, REPORT_ANNOTATION_PROMPT_VERSIONS, REPORT_ANNOTATION_BEST_PROMPT_VERSION, REPORT_ANNOTATION_N

def process_split(df_split, split_name):
    reports_split = df_split['original_report']
    if split_name == 'val':
        prompt_versions = [REPORT_ANNOTATION_BEST_PROMPT_VERSION]
    else:
        prompt_versions = REPORT_ANNOTATION_PROMPT_VERSIONS

    for prompt_version in prompt_versions:
        reports_annotations = report_annotation.annotate_reports(reports_split, prompt_version=prompt_version)
        df_split['annotation_{prompt_version}'.format(prompt_version=prompt_version)] = reports_annotations

    df_split.to_csv(os.path.join(DATA_PATH, f'reports_annotations_{split_name}.csv'), index=False)


@click.command()
@click.argument('split', type=str, default='train')
def main(split):
    # reads data from reports in train and test in storage
    df = pd.read_csv(os.path.join(DATA_PATH, 'data_prep.csv'))
    df_train = df[df['split'] == 'train'].sample(n=REPORT_ANNOTATION_N)
    df_test = df[df['split'] == 'test'].sample(n=REPORT_ANNOTATION_N)
    df_val = df[df['split'] == 'val']

    if split == 'train':
        process_split(df_train, 'train')

    if split == 'test':
        process_split(df_test, 'test')

    if split == 'val':
        process_split(df_val, 'val')

if __name__ == '__main__':
    main()