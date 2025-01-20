import os
import pandas as pd
import json

from xray_report_gen import eval
from xray_report_gen import utils

DATA_PATH = '../data/'
PROMPT_VERSIONS = [1, 2]
REGIONS = ['bone', 'heart', 'lung', 'mediastinal']

# evaluating on the training set

# load the processed data from storage
df_train = pd.read_csv(os.path.join(DATA_PATH, 'reports_annotations_train.csv'))
df_train = utils.parse_annotations(df_train, 'annotation', REGIONS)

# load the processed data from storage
df_test = pd.read_csv(os.path.join(DATA_PATH, 'reports_annotations_test.csv'))
df_test = utils.parse_annotations(df_test, 'annotation', REGIONS)

for prompt_version in PROMPT_VERSIONS:
    for region in REGIONS:
        df_train = utils.parse_annotations(df_train, 'annotation_{prompt_version}'.format(prompt_version=prompt_version), REGIONS)

        print('evaluating on {prompt_version} and {region}'.format(prompt_version=prompt_version, region=region))
        res = eval.get_green_scorer_res(df_train[f'annotation_{region}'.format(region)], df_train['annotation_{prompt_version}_{region}'.format(prompt_version=prompt_version, region=region)])


for prompt_version in PROMPT_VERSIONS:
    for region in REGIONS:
        df_test = utils.parse_annotations(df_test, 'annotation_{prompt_version}'.format(prompt_version=prompt_version), REGIONS)

        print('evaluating on {prompt_version} and {region}'.format(prompt_version=prompt_version, region=region))
        res = eval.get_green_scorer_res(df_test[f'annotation_{region}'.format(region)], df_test['annotation_{prompt_version}_{region}'.format(prompt_version=prompt_version, region=region)])

