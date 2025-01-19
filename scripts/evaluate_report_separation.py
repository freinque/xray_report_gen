import pandas as pd

from xray_report_gen import eval

# load the processed data from storage

refs = ''
hyps = ''

# call the evaluation methods on processed data
mean, std, green_score_list, summary, result_df = eval.get_green_scorer_res(refs, hyps)

