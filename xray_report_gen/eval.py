# evaluation methods based on  https://stanford-aimi.github.io/green.html and references therein

from green_score import GREEN

MODEL_NAME = "StanfordAIMI/GREEN-radllama2-7b"

def get_green_scorer_res(refs, hyps):
    """
    evaluation methods based on  https://stanford-aimi.github.io/green.html and references therein

    :param refs: list of strings, reports taken as reference
    :param hyps: list of strings, reports taken as candidates

    :return:
    """
    green_scorer = GREEN(MODEL_NAME, output_dir=".", cpu=True) # TODO make more flexible
    mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)
    return mean, std, green_score_list, summary, result_df

def print_green_scorer_report(result_df):
    for index, row in result_df.iterrows():
        print(f"Row {index}:\n")
        for col_name in result_df.columns:
            print(f"{col_name}: {row[col_name]}\n")
        print('-' * 80)

"""
from transformers import pipeline

def
messages = [
    {"role": "user", "content": "Who are you?"},
]
trans_pipeline = pipeline("text-generation", model="StanfordAIMI/GREEN-RadLlama2-7b")
print(trans_pipeline(messages))

print(trans_pipeline(messages))

"""
