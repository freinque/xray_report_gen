"""
script intented to invoke report LLM annotation inference on preprocessed datasets
"""

import os
import pandas as pd
import click

from xray_report_gen import report_annotation
from xray_report_gen.config import DATA_PATH, PROMPT_VERSIONS, BEST_PROMPT_VERSION, N

@click.command()
@click.argument('mode', type=str, default='train')
def main(mode):
    # reads data from reports in train and test in storage
    df = pd.read_csv(os.path.join(DATA_PATH, 'data_prep.csv'))
    df_train = df[df['split'] == 'train'].sample(n=N)
    df_test = df[df['split'] == 'test'].sample(n=N)
    df_val = df[df['split'] == 'val']

    reports_train = df_train['original_report']
    reports_test = df_test['original_report']
    reports_val = df_val['original_report']

    if mode == 'train':
        # looking at results on df_train
        for prompt_version in PROMPT_VERSIONS:
            # processing df_train with various prompt variants
            reports_annotations_train = report_annotation.annotate_reports(reports_train, prompt_version=prompt_version)
            df_train['annotation_{prompt_version}'.format(prompt_version=prompt_version)] = reports_annotations_train

        # write to data storage
        df_train.to_csv(os.path.join(DATA_PATH, 'reports_annotations_train.csv'), index=False)

    if mode == 'test':
        # looking at results on df_test
        for prompt_version in PROMPT_VERSIONS:
            # processing df_test with various prompt variants
            reports_annotations_test = report_annotation.annotate_reports(reports_test, prompt_version=prompt_version)
            df_test['annotation_{prompt_version}'.format(prompt_version=prompt_version)] = reports_annotations_test

        # write to data storage
        df_test.to_csv(os.path.join(DATA_PATH, 'reports_annotations_test.csv'), index=False)

    if mode == 'val':
        # processing df_val with various prompt variants
        prompt_version = BEST_PROMPT_VERSION
        reports_annotations_val = report_annotation.annotate_reports(reports_val, prompt_version=prompt_version)
        df_val['annotation_{prompt_version}'.format(prompt_version=prompt_version)] = reports_annotations_val

        # write to data storage
        df_val.to_csv(os.path.join(DATA_PATH, 'reports_annotations_val.csv'), index=False)

if __name__ == '__main__':
    main()