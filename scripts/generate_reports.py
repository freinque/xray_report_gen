import os
import pandas as pd

from xray_report_gen import utils, report_generation

DATA_PATH = '../data/'
MODEL_VERSION = 1
N = 4

def main():
    df = pd.read_csv(os.path.join(DATA_PATH, 'data_prep.csv'))

    df = df[df['image_found']].sample(n=N)
    image_filenames = df['image_filename']
    print('image_filenames: ', image_filenames)

    # Load images
    images = utils.load_images(image_filenames)
    print('images: ', images)

    # Run inference
    results = report_generation.run_inference(images, model_version=MODEL_VERSION)

    # Print results
    for i, result in enumerate(results, 1):
        print(f"Image {i} Analysis: {result}")


if __name__ == "__main__":
    main()
