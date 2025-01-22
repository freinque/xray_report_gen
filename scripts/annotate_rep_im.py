import os
import click
import pandas as pd

from xray_report_gen import utils, report_generation

DATA_PATH = '../data/'
MODEL_VERSION = 1
N = 25

@click.command()
@click.argument('mode')
@click.option('--version', type=int, default=MODEL_VERSION)
def main(mode, version):
    df = pd.read_csv(os.path.join(DATA_PATH, 'data_prep.csv'))

    df = df[df['split'] == mode]
    df = df[df['image_found']].sample(n=N)
    image_filenames = df['image_filename']
    print('image_filenames: ', image_filenames)

    # Load images
    images = utils.load_images(image_filenames)
    print('images: ', images)

    reports = df['original_report'].to_list()

    # Run inference
    results = report_generation.run_inference(images, reports, version=version)

    df['annotation_{version}'.format(version=version)] = results

    # write to data storage
    df.to_csv(os.path.join(DATA_PATH, 'reports_annotations_{mode}_model_{version}.csv'.format(mode=mode, version=version)), index=False)

if __name__ == "__main__":
    main()
