# evaluation methods based on  https://stanford-aimi.github.io/green.html and references therein
import os
import torch

from green_score import GREEN
from .config import EVAL_MODEL_NAME

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def get_green_scorer_res(refs, hyps):
    """
    evaluation methods based on  https://stanford-aimi.github.io/green.html and references therein

    :param refs: list of strings, reports taken as reference
    :param hyps: list of strings, reports taken as candidates

    :return:
    """
    cpu = not torch.cuda.is_available()
    print('cpu: ', cpu)
    print('evaluation using model: ', EVAL_MODEL_NAME)
    green_scorer = GREEN(EVAL_MODEL_NAME, output_dir=".", cpu=False)
    mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)
    print('mean: ', mean)
    print('green_score_list: ', green_score_list)
    print('summary: ', summary)
    return result_df

def print_green_scorer_report(result_df):
    for index, row in result_df.iterrows():
        print(f"Row {index}:\n")
        for col_name in result_df.columns:
            print(f"{col_name}: {row[col_name]}\n")
        print('-' * 80)

