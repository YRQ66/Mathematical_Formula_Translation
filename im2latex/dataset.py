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
        image_path = join(self.root_dir, file_name+'.png')
        if not isfile(image_path):
            index = index - 1 if index > 0 else index + 1 
            return self.__getitem__(index)
        image = Image.open(join(self.root_dir, file_name+'.png')).convert("RGB")
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

def prepare_dataset(data_dir, max_length_token, vocab_size, processor_path, dataset_type='140K'):

    dataset_dir = join(data_dir, dataset_type)
    formulas_file = join(dataset_dir, "formulas.txt")
    # linux 인코딩 변환 : iconv -c -f ISO-8859-1 -t utf-8 im2latex_formulas.lst > im2latex_formulas_utf.lst

    # train_df/test_df/valid_df
    types = ['train', 'valid', 'test']
    for type in types:
        df_dir = '{}.pkl'.format(type)
        df = pd.read_pickle(join(dataset_dir, df_dir))
        globals()["{}_df".format(type)] = df
        

    tokenizer_ = tokenizer(formulas_file = formulas_file, data_dir = data_dir, max_length = max_length_token, vocab_size=vocab_size)
    root_dir = join(dataset_dir, 'images') 
    processor = TrOCRProcessor.from_pretrained(processor_path, Use_fast= False)
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
    return train_dataset, val_dataset, test_dataset, tokenizer_, processor

if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_train', default='config/config_train.yaml', type=str, help='path of train configuration yaml file')

    pre_args = parser.parse_args()

    with open(pre_args.config_train) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    train_dataset, val_dataset, test_dataset, tokenizer, processor = \
    prepare_dataset(data_dir = args['data_dir'], max_length_token=args['max_length_token'], \
                    vocab_size=args['vocab_size'], processor_path=args['processor_path'], dataset_type=args['dataset_type'])

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))
    print("Number of test examples:", len(test_dataset))

