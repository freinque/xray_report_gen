"""
invocation of evaluation routines on
- report LLM annotation inference on preprocessed datasets
- report+image multimodal annotation inference on preprocessed datasets
using GREEN (https://stanford-aimi.github.io/green.html)
"""

import os
import pandas as pd
import click

from xray_report_gen import eval
from xray_report_gen import utils
from xray_report_gen.config import DATA_PATH, REPORT_ANNOTATION_PROMPT_VERSIONS, REGIONS

@click.command()
@click.argument('split')
@click.option('--multi', type=int, default=0)
def main(split, multi):
    # evaluating on the training set
    print('split:', split)

    if multi == 0:
        # evaluation on train set
        for prompt_version in REPORT_ANNOTATION_PROMPT_VERSIONS:
            for region in REGIONS:
                # load the processed data from storage
                df_split = pd.read_csv(os.path.join(DATA_PATH, 'reports_annotations_{split}.csv'.format(split=split)))
                df_split = utils.parse_annotations(df_split, 'annotation', REGIONS)

                df_split = utils.parse_annotations(df_split, 'annotation_{prompt_version}'.format(prompt_version=prompt_version), REGIONS)

                print('evaluating on {prompt_version} and {region}'.format(prompt_version=prompt_version, region=region))
                ref = df_split['annotation_{region}'.format(region=region)]
                hyp = df_split['annotation_{prompt_version}_{region}'.format(prompt_version=prompt_version, region=region)]
                print('ref = {ref}'.format(ref=ref))
                print('hyp = {hyp}'.format(hyp=hyp))
                res_df = eval.get_green_scorer_res(ref, hyp)
                res_df.to_csv(os.path.join(DATA_PATH, 'reports_annotations_{split}_eval_{prompt_version}_{region}.csv'.format(split=split, prompt_version=prompt_version, region=region)), index=False)
    else:
        # evaluation on train set
        for prompt_version in REPORT_ANNOTATION_PROMPT_VERSIONS:
            for region in REGIONS:
                # load the processed data from storage
                df_split = pd.read_csv(os.path.join(DATA_PATH,
                                                   'rep_im_annotations_{split}_model_{prompt_version}.csv'.format(
                                                       split=split, prompt_version=prompt_version)))
                df = pd.read_csv(os.path.join(DATA_PATH, 'data_prep.csv'))[['id','annotation']].drop_duplicates()
                df_split = pd.merge(df_split, df, how='left', on='id')

                df_split = utils.parse_annotations(df_split, 'annotation', REGIONS)

                df_split = utils.parse_annotations(df_split, 'annotation_{prompt_version}'.format(prompt_version=prompt_version), REGIONS)

                print('evaluating mutimodal on {prompt_version} and {region}'.format(prompt_version=prompt_version, region=region))
                ref = df_split['annotation_{region}'.format(region=region)]
                hyp = df_split['annotation_{prompt_version}_{region}'.format(prompt_version=prompt_version, region=region)]
                print('ref = {ref}'.format(ref=ref))
                print('hyp = {hyp}'.format(hyp=hyp))
                res_df = eval.get_green_scorer_res(ref, hyp)
                res_df.to_csv(os.path.join(DATA_PATH, 'rep_im_annotations_{split}_eval_{prompt_version}_{region}.csv'.format(split=split, prompt_version=prompt_version, region=region)), index=False)

if __name__ == '__main__':
    main()