"""
This script preprocesses data found in DATA_PATH for usage in annotation, evaluation, and finetuning scripts.

Functions:
    main(): Preprocesses the data by reading, merging, and transforming datasets, then writes the processed data to a CSV file.
    get_data_dict(images, reports, annotations, ids): Creates a list of dictionaries containing image paths, reports, and annotations for finetuning datasets.
    write_finetuning_datasets(): Reads the processed data, creates finetuning datasets, and writes them to JSON files.

"""
import os
import json
import pandas as pd

from xray_report_gen.config import DATA_PATH, IMAGES_DIR

def write_base_dataset():
    # reads data from reports in train and test in storage
    reports = pd.read_csv(os.path.join(DATA_PATH, 'indiana_reports.csv'))
    print("report data loaded, has shape ", reports.shape)
    ids = pd.read_csv(os.path.join(DATA_PATH, 'indiana_projections.csv'))
    print("image metadata loaded, has shape ", ids.shape)

    # merging reports and metadata
    df = pd.merge(reports, ids, on=['uid'], how='left')

    # stardardizing ids
    df['uid'] = df['uid'].astype(int)
    df['im_1'] = df['filename'].apply(lambda x: x.split('-')[1])
    df['im_2'] = df['filename'].apply(lambda x: x.split('-')[2][:4])

    # adding 'curated' annotations provided
    annotations_json = json.load(open(os.path.join(DATA_PATH, 'annotation_quiz_all.json')))
    annotations_train = pd.DataFrame.from_records(annotations_json['train'])
    print("annotation data loaded, train set has shape ",  annotations_train.shape)
    annotations_test = pd.DataFrame.from_records(annotations_json['test'])
    print("annotation data loaded, test set has shape ",annotations_test.shape)
    annotations_val = pd.DataFrame.from_records(annotations_json['val'])
    print("annotation data loaded, val set has shape ",annotations_val.shape)
    annotations = pd.concat([annotations_train, annotations_test, annotations_val], axis=0)
    annotations = annotations.rename(columns={'report':'annotation'})
    print("annotation data concatenated, has shape ",annotations.shape)

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
    print("overall dataset concatenated and transformed, has shape ", df.shape)


def get_data_dict(images, reports, annotations, ids):
    d = []
    for image, report, annotation, i in zip(images, reports, annotations, ids):
        d.append({
            "id": i,
            "messages": [
                {"role": "system", "content": "system_prompt"},
                {
                    "role": "user",
                    "content": [
                                   {"type": "image", "image": im} for im in image
                               ]
                               +
                               [
                                   {"type": "text", "text": report}
                               ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": annotation}
                    ]
                }
            ]
        })
    return d


def write_finetuning_datasets():
    """
    Reads a CSV file, processes the data, and writes fine-tuning datasets 
    for training, testing, and validation splits to JSON files.

    The function performs the following steps:
    1. Reads the CSV file located at DATA_PATH/data_prep.csv.
    2. Processes the data to extract image paths, original reports, and annotations.
    3. Groups the data by 'id' and creates dictionaries for each split ('train', 'test', 'val').
    4. Writes the processed data to JSON files for each split.

    Returns:
        None
    """
    df = pd.read_csv(os.path.join(DATA_PATH, 'data_prep.csv'))

    df['image_path'] = df['image_filename']
    df['original_report'] = df['original_report']
    df[('annotation')] = df['annotation']

    for split in ['train', 'test', 'val']:
        df_train = df[(df['split'] == split) & df['image_found']][['id', 'image_path', 'original_report', 'annotation']]

        images = df_train.groupby('id').apply(lambda df: list(df['image_path']))
        reports = df_train.groupby('id').apply(lambda df: df['original_report'].max())
        annotations = df_train.groupby('id').apply(lambda df: df['annotation'].max())
        ids = annotations.index.values

        data = get_data_dict(images, reports, annotations, ids)
        # Write the list of dictionaries to a JSON file
        output_file = os.path.join(DATA_PATH, f'finetune_data_{split}.json')
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print("{} finetuning dataset written".format(split))


def main():
    write_base_dataset()
    write_finetuning_datasets()

if __name__ == '__main__':
    main()