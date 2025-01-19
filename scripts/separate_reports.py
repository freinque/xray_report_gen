import os
import pandas as pd

from xray_report_gen import report_separation

DATA_PATH = '../data/'

# reads data from reports in train and test in storage
reports = pd.read_csv(os.path.join(DATA_PATH, 'indiana_reports.csv'))

# precessing
sep_reports = report_separation.separate_reports(reports)

# write to data storage
sep_reports.to_csv(os.path.join(DATA_PATH, 'indiana_reports.csv'), index=False)