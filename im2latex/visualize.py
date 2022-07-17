from preprocess import preprocess_df
from os.path import join
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def visualize_formula(data_dir, df):
    # Show some images as sanity check
    sample = df.sample(n=1)
    print(sample['text'])
    img_dir = join(data_dir, 'formula_images_processed', sample.iloc[0]['file_name']+".png")
    img = cv2.imread(img_dir)
    cv2.imshow('image', img)
    cv2.waitKey(5000) 
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default='./data')
    parser.add_argument("-m", "--max_length_token", default=100)
    args = parser.parse_args()

    formulas_file = join(args.data_dir, "im2latex_formulas_utf.lst")
    # linux 인코딩 변환 : iconv -c -f ISO-8859-1 -t utf-8 im2latex_formulas.lst > im2latex_formulas_utf.lst
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    # train_df/test_df/validate_df
    types = ['train', 'test', 'validate']
    for type in types:
        df = preprocess_df(formulas, data_dir = args.data_dir, type = type, max_length_token=args.max_length_token)
        globals()["{}_df".format(type)] = df

    visualize_formula(args.data_dir, train_df)