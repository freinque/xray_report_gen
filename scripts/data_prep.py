import os
import json
import pandas as pd

DATA_PATH = '/xray_report_gen/data/'
IMAGES_DIR = '/xray_report_gen/data/images'

def main():
    # reads data from reports in train and test in storage
    reports = pd.read_csv(os.path.join(DATA_PATH, 'indiana_reports.csv'))
    print(reports.shape)
    ids = pd.read_csv(os.path.join(DATA_PATH, 'indiana_projections.csv'))
    print(ids.shape)

    # merging
    df = pd.merge(reports, ids, on=['uid'], how='left')

    # stardardizing ids
    df['uid'] = df['uid'].astype(int)
    df['im_1'] = df['filename'].apply(lambda x: x.split('-')[1])
    df['im_2'] = df['filename'].apply(lambda x: x.split('-')[2][:4])

    # adding annotations provided
    annotations_json = json.load(open(os.path.join(DATA_PATH, 'annotation_quiz_all.json')))
    annotations_train = pd.DataFrame.from_records(annotations_json['train'])
    print(annotations_train.shape)
    annotations_test = pd.DataFrame.from_records(annotations_json['test'])
    print(annotations_test.shape)
    annotations_val = pd.DataFrame.from_records(annotations_json['val'])
    print(annotations_val.shape)
    annotations = pd.concat([annotations_train, annotations_test, annotations_val], axis=0)
    annotations = annotations.rename(columns={'report':'annotation'})
    print(annotations.shape)

    annotations['uid'] = annotations['id'].apply(lambda x: x.split('_')[0].replace('CXR','')).astype(int)
    annotations['im_1'] = annotations['id'].apply(lambda x: x.split("IM-")[-1][:4])

    # merging into overall dataset
    df = pd.merge(df, annotations, how='left', on=['uid', 'im_1'], suffixes=('', '_annotated'))

    # filling nulls to get uniform 'original_report'
    df['original_report'] = df['original_report'].fillna(df['findings'])

    df['image_folder'] = df['filename'].apply(lambda x: 'CXR' + '-'.join(x.split('-')[:2]) )
    #df['image_number'] = df['filename'].apply(lambda x: x.split('-')[-1].split('0')[0].split('.')[0] )
    # TODO can't reconnect image specific to row because of mix-up in data folder
    df['image_number'] = df.groupby('image_folder').cumcount().astype(str)
    df['image_filename'] = df.apply(lambda x: os.path.join(IMAGES_DIR, x['image_folder'], x['image_number']+'.png'), axis=1 )

    df['image_found'] = df['image_filename'].apply(lambda x: os.path.exists(x) )

    # writing to storage
    df.to_csv(os.path.join(DATA_PATH, 'data_prep.csv'), index=False)

def write_finetuning_datasets():
    df = pd.read_csv(os.path.join(DATA_PATH, 'data_prep.csv'))

    df['image_path'] = df['image_filename']
    df['original_report'] = df['original_report']

    df_train = df[(df['split'] == 'train') & df['image_found']][['image_path', 'original_report']]
    data = df_train.to_dict(orient='records')
    # Write the list of dictionaries to a JSON file
    output_file = os.path.join(DATA_PATH, 'finetune_data_train.json')
    with open(output_file, 'w') as f:
        import json
        json.dump(data, f, indent=4)

    df_test = df[(df['split'] == 'test') & df['image_found']][['image_path', 'original_report']]
    data = df_test.to_dict(orient='records')
    # Write the list of dictionaries to a JSON file
    output_file = os.path.join(DATA_PATH, 'finetune_data_test.json')
    with open(output_file, 'w') as f:
        import json
        json.dump(data, f, indent=4)

    df_val = df[(df['split'] == 'val') & df['image_found']][['image_path', 'original_report']]
    data = df_val.to_dict(orient='records')
    # Write the list of dictionaries to a JSON file
    output_file = os.path.join(DATA_PATH, 'finetune_data_val.json')
    with open(output_file, 'w') as f:
        import json
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    main()
    write_finetuning_datasets()