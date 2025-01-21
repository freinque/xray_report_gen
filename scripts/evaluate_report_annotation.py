import os
import pandas as pd
import json

from xray_report_gen import eval
from xray_report_gen import utils

DATA_PATH = '../data/'
PROMPT_VERSIONS = [1, 2]
REGIONS = ['bone', 'heart', 'lung', 'mediastinal']

def main(mode='train'):
    # evaluating on the training set
    print('mode:', mode)

    # load the processed data from storage
    df_train = pd.read_csv(os.path.join(DATA_PATH, 'reports_annotations_train.csv'))
    df_train = utils.parse_annotations(df_train, 'annotation', REGIONS)

    # load the processed data from storage
    df_test = pd.read_csv(os.path.join(DATA_PATH, 'reports_annotations_test.csv'))
    df_test = utils.parse_annotations(df_test, 'annotation', REGIONS)

    #df_val = pd.read_csv(os.path.join(DATA_PATH, 'reports_annotations_val.csv'))
    #df_val = utils.parse_annotations(df_val, 'annotation', REGIONS)

    if mode == 'train':
        # evaluation on train set
        for prompt_version in PROMPT_VERSIONS:
            for region in REGIONS:
                df_train = utils.parse_annotations(df_train, 'annotation_{prompt_version}'.format(prompt_version=prompt_version), REGIONS)

                print('evaluating on {prompt_version} and {region}'.format(prompt_version=prompt_version, region=region))
                ref = df_train[f'annotation_{region}'.format(region)]
                hyp = df_train['annotation_{prompt_version}_{region}'.format(prompt_version=prompt_version, region=region)]
                print('ref = {ref}'.format(ref=ref))
                print('hyp = {hyp}'.format(hyp=hyp))
                res_df = eval.get_green_scorer_res(ref, hyp)
                res_df.to_csv(os.path.join(DATA_PATH, 'reports_annotations_train_eval_{prompt_version}_{region}.csv'.format(prompt_version=prompt_version, region=region)), index=False)

    if mode == 'test':
        # evaluation on test set
        for prompt_version in PROMPT_VERSIONS:
            for region in REGIONS:
                df_test = utils.parse_annotations(df_test, 'annotation_{prompt_version}'.format(prompt_version=prompt_version), REGIONS)

                print('evaluating on {prompt_version} and {region}'.format(prompt_version=prompt_version, region=region))
                ref = df_test[f'annotation_{region}'.format(region)]
                hyp = df_test['annotation_{prompt_version}_{region}'.format(prompt_version=prompt_version, region=region)]
                print('ref = {ref}'.format(ref=ref))
                print('hyp = {hyp}'.format(hyp=hyp))
                res_df = eval.get_green_scorer_res(ref, hyp)
                res_df.to_csv(os.path.join(DATA_PATH, 'reports_annotations_test_eval_{prompt_version}_{region}.csv'.format(
                    prompt_version=prompt_version, region=region)), index=False)

    if mode == 'val':
        # evaluation on test set
        for prompt_version in PROMPT_VERSIONS:
            for region in REGIONS:
                df_val = utils.parse_annotations(df_val, 'annotation_{prompt_version}'.format(prompt_version=prompt_version), REGIONS)

                print('evaluating on {prompt_version} and {region}'.format(prompt_version=prompt_version, region=region))
                ref = df_val[f'annotation_{region}'.format(region)]
                hyp = df_val['annotation_{prompt_version}_{region}'.format(prompt_version=prompt_version, region=region)]
                print('ref = {ref}'.format(ref=ref))
                print('hyp = {hyp}'.format(hyp=hyp))
                res_df = eval.get_green_scorer_res(ref, hyp)
                res_df.to_csv(os.path.join(DATA_PATH, 'reports_annotations_val_eval_{prompt_version}_{region}.csv'.format(
                    prompt_version=prompt_version, region=region)), index=False)


if __name__ == '__main__':
    main('test')