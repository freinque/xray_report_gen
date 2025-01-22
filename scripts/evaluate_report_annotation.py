import os
import pandas as pd
import json
import click

from xray_report_gen import eval
from xray_report_gen import utils

DATA_PATH = '../data/'
PROMPT_VERSIONS = [1, 2]
REGIONS = ['bone', 'heart', 'lung', 'mediastinal']

@click.command()
@click.argument('mode')
@click.option('--multi', type=int, default=0)
def main(mode, multi):
    # evaluating on the training set
    print('mode:', mode)

    if multi == 0:
        # evaluation on train set
        for prompt_version in PROMPT_VERSIONS:
            for region in REGIONS:
                # load the processed data from storage
                df_mode = pd.read_csv(os.path.join(DATA_PATH, 'reports_annotations_{mode}.csv'.format(mode=mode)))
                df_mode = utils.parse_annotations(df_mode, 'annotation', REGIONS)

                df_mode = utils.parse_annotations(df_mode, 'annotation_{prompt_version}'.format(prompt_version=prompt_version), REGIONS)

                print('evaluating on {prompt_version} and {region}'.format(prompt_version=prompt_version, region=region))
                ref = df_mode['annotation_{region}'.format(region=region)]
                hyp = df_mode['annotation_{prompt_version}_{region}'.format(prompt_version=prompt_version, region=region)]
                print('ref = {ref}'.format(ref=ref))
                print('hyp = {hyp}'.format(hyp=hyp))
                res_df = eval.get_green_scorer_res(ref, hyp)
                res_df.to_csv(os.path.join(DATA_PATH, 'reports_annotations_{mode}_eval_{prompt_version}_{region}.csv'.format(mode=mode, prompt_version=prompt_version, region=region)), index=False)
    else:
        # evaluation on train set
        for prompt_version in PROMPT_VERSIONS:
            for region in REGIONS:
                # load the processed data from storage
                df_mode = pd.read_csv(os.path.join(DATA_PATH,
                                                   'rep_im_annotations_{mode}_model_{prompt_version}.csv'.format(
                                                       mode=mode, prompt_version=prompt_version)))
                df = pd.read_csv(os.path.join(DATA_PATH, 'data_prep.csv'))[['id','annotation']].drop_duplicates()
                df_mode = pd.merge(df_mode, df, how='left', on='id')

                df_mode = utils.parse_annotations(df_mode, 'annotation', REGIONS)

                df_mode = utils.parse_annotations(df_mode, 'annotation_{prompt_version}'.format(prompt_version=prompt_version), REGIONS)

                print('evaluating mutimodal on {prompt_version} and {region}'.format(prompt_version=prompt_version, region=region))
                ref = df_mode['annotation_{region}'.format(region=region)]
                hyp = df_mode['annotation_{prompt_version}_{region}'.format(prompt_version=prompt_version, region=region)]
                print('ref = {ref}'.format(ref=ref))
                print('hyp = {hyp}'.format(hyp=hyp))
                res_df = eval.get_green_scorer_res(ref, hyp)
                res_df.to_csv(os.path.join(DATA_PATH, 'rep_im_annotations_{mode}_eval_{prompt_version}_{region}.csv'.format(mode=mode, prompt_version=prompt_version, region=region)), index=False)

if __name__ == '__main__':
    main()