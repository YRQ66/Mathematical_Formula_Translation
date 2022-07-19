from re import A
import pandas as pd
from os.path import join

def preprocess_df(data_dir, type = None, max_length_token=None):

    pth = 'training_56/df_{}.pkl'.format(type)
    df = pd.read_pickle(join(data_dir, pth))
    
    
    # Replace text with formulas
    # df['text'] = df.apply (lambda row: formulas[int(row['text_index'])], axis=1)
    df['len'] = df.apply (lambda row: row['latex_ascii'].count(' '), axis=1)

    # Sort by ascending length of formula
    df_sorted = df.sort_values(by="len")
    df_filtered = df_sorted[df_sorted['len'] > 0 ]
    df_trunc = df_filtered[df_filtered['len'] <= max_length_token ]
    df = df_trunc
    df = df.reset_index(drop=True)
    return df

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
    types = ['train', 'valid', 'test']
    for type in types:
        df = preprocess_df(formulas, data_dir=args.data_dir, type = type, max_length_token=args.max_length_token)
        globals()["{}_df".format(type)] = df
        print(type+'_dataframe',df.head())