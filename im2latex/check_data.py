import torch
from torch.utils.data import DataLoader

import argparse
import yaml
import os
from tqdm import tqdm
import numpy as np
import random
from dataset import prepare_dataset



# Set up environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

def check_data(args):
  seed_everything(args['seed'])

  train_dataset, val_dataset, test_dataset, tokenizer, processor = \
  prepare_dataset(data_dir = args['data_dir'], max_length_token=args['max_length_token'], \
                  vocab_size=args['vocab_size'], processor_path=args['processor_path'], dataset_type=args['dataset_type'])

  train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'])
  test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'])

  # train_cnt = 0
  # for i, batch in enumerate(tqdm(train_dataloader)):
  #   label = batch["labels"]
  #   label_str = tokenizer.decode_batch(label.tolist(), skip_special_tokens=True)
  #   if '' in label_str:
  #         print(f"{i}th label:", label)
  #         print("\n")
  #         print(f"{i}th label_str:{label_str}")
  # print(f'train empty data cnt: {val_cnt}')

  val_cnt = 0
  for i, batch in enumerate(tqdm(val_dataloader)):
    label = batch["labels"]
    label_str = tokenizer.decode_batch(label.tolist(), skip_special_tokens=True)
    if '' in label_str:
      val_cnt += 1
      print(f"{i}th label:", label)
      print("\n")
      print(f"{i}th label_str:{label_str}")
  print(f'val empty data cnt: {val_cnt}')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--config_train', default='config/config_train.yaml', type=str, help='path of train configuration yaml file')
  
  pre_args = parser.parse_args()

  with open(pre_args.config_train) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

  check_data(args)