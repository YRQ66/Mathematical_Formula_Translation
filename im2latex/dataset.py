import pandas as pd
from PIL import Image
from os.path import join, isfile
from preprocess import preprocess_df
from tokenizer import tokenizer

import json

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, tokenizer, max_target_length=None):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['image'][idx]
        text = self.df['formula'][idx]
        # prepare image (i.e. resize + normalize)
        # image = Image.open(self.root_dir + file_name +'.png').convert("RGB")
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
#        # add labels (input_ids) by encoding the text
#        labels = self.processor.tokenizer(text, 
#                                          padding="max_length", 
#                                          max_length=self.max_target_length).input_ids
#        # important: make sure that PAD tokens are ignored by the loss function
#        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        labels = self.tokenizer.encode(text).ids
        labels = [label if label != self.tokenizer.token_to_id("[PAD]") else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def prepare_dataset(data_dir, max_length_token, vocab_size, dataset_type='140K'):

    dataset_dir = join(data_dir, dataset_type)
    formulas_file = join(dataset_dir, "formulas.txt")
    # linux 인코딩 변환 : iconv -c -f ISO-8859-1 -t utf-8 im2latex_formulas.lst > im2latex_formulas_utf.lst
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    # train_df/test_df/valid_df
    types = ['train', 'valid', 'test']
    for type in types:
        df_dir = '{}.pkl'.format(type)
        df = pd.read_pickle(join(dataset_dir, df_dir))
        globals()["{}_df".format(type)] = df

    tokenizer_ = tokenizer(formulas_file = formulas_file, data_dir = data_dir, max_length = max_length_token, vocab_size=vocab_size)
    
    root_dir = join(data_dir, 'images/',) 
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed", Use_fast= False)
    train_dataset = IAMDataset(root_dir=root_dir,
                            df=train_df,
                            processor=processor,
                            tokenizer=tokenizer_)
    val_dataset = IAMDataset(root_dir=root_dir,
                            df=valid_df,
                            processor=processor,
                            tokenizer=tokenizer_)
    test_dataset = IAMDataset(root_dir=root_dir,
                            df=test_df,
                            processor=processor,
                            tokenizer=tokenizer_)
    return train_dataset, val_dataset, test_dataset, tokenizer_

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default='./data')
    # tokenize
    parser.add_argument("-m", "--max_length_token", default=100)
    parser.add_argument("-v", "--vocab_size", default=600)
    # dataset
    parser.add_argument("-b", "--batch_size", default=16)
    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(data_dir = args.data_dir, max_length_token=args.max_length_token, vocab_size=args.vocab_size)
    
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))
    print("Number of test examples:", len(test_dataset))

    encoding = train_dataset[0]
    for k,v in encoding.items():
        print(k, v.shape)
    labels = encoding['labels']
    labels[labels == -100] = tokenizer.token_to_id("[PAD]")
    label_str = tokenizer.decode(labels.tolist(), skip_special_tokens=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)