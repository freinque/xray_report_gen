"""
script intented to invoke report+image multimodal annotation inference on preprocessed datasets
"""

import os
import click
import pandas as pd

from xray_report_gen import utils, rep_im_annotation
from xray_report_gen.config import DATA_PATH, REPORT_IMAGE_ANNOTATION_DEFAULT_MODEL_VERSION, TRAIN_DATASET, TEST_DATASET, VAL_DATASET, TRAIN_DATASET_N, TEST_DATASET_N, VAL_DATASET_N

def get_dataset_path(mode) -> str:
    if mode == "train":
        return TRAIN_DATASET
    elif mode == "test":
        return TEST_DATASET
    else:
        return VAL_DATASET

def get_n(mode) -> int:
    if mode == "train":
        return TRAIN_DATASET_N
    elif mode == "test":
        return TEST_DATASET_N
    else:
        return VAL_DATASET_N

@click.command()
@click.argument('mode')
@click.option('--version', type=int, default=REPORT_IMAGE_ANNOTATION_DEFAULT_MODEL_VERSION)
def main(mode, version):

    dataset = get_dataset_path(mode)
    n = get_n(mode)
    # Run inference
    ids, annotations = rep_im_annotation.run_inference(dataset, n=n, version=version)
    df = pd.DataFrame(ids, columns=['id'])
    df['annotation_{version}'.format(version=version)] = annotations

    # write to data storage
    df.to_csv(os.path.join(DATA_PATH, 'rep_im_annotations_{mode}_model_{version}.csv'.format(mode=mode, version=version)), index=False)

if __name__ == "__main__":
    main()
