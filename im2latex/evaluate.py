from ntpath import join
import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig

import argparse
import os
from tqdm import tqdm
import numpy as np
import random

from dataset import prepare_dataset

from metric import get_pred_and_label_str
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu

import wandb
import yaml

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

def compute_bleu(args):
  seed_everything(args['seed'])
  
  # settings
  print("pytorch version: {}".format(torch.__version__))
  gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
  print("GPU 사용 가능 여부: {}".format(gpu))
  print(torch.cuda.get_device_name(0))
  print(torch.cuda.device_count())

  train_dataset, val_dataset, test_dataset, tokenizer = \
  prepare_dataset(data_dir = args['data_dir'], max_length_token=args['max_length_token'], \
                  vocab_size=args['vocab_size'], processor_path=args['processor_path'])

  # train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
  if args['mode'] == 'val':
    eval_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'])
  else:
    eval_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'])

  device = torch.device("cuda" if torch.cuda.is_available() \
                        else "mps" if torch.backends.mps.is_available() else "cpu")

  model_config = VisionEncoderDecoderConfig.from_pretrained(args['model_path'])
  model = VisionEncoderDecoderModel.from_pretrained(args['model_path'],
                                    config=model_config)
  model.to(device)

  # set special tokens used for creating the decoder_input_ids from the labels
  model.config.decoder_start_token_id = tokenizer.token_to_id("[CLS]")
  model.config.pad_token_id = tokenizer.token_to_id("[PAD]")
  # make sure vocab size is set correctly
  model.config.vocab_size = args['vocab_size']

  # set beam search parameters
  model.config.eos_token_id = tokenizer.token_to_id("[SEP]")
  model.config.max_length = args['max_length_token'] #64
  model.config.early_stopping = True
  model.config.no_repeat_ngram_size = 3
  model.config.length_penalty = 2.0
  model.config.num_beams = 4

  model.tokenizer = tokenizer

  step = 0
  # validate
  print(f'Start evaluation_nltk!!!')
  model.eval()
  val_bleu = 0.0 
  
  with torch.no_grad():
    for i, batch in enumerate(tqdm(eval_dataloader)):
      # generate ver.
      # generate output : sequence of token, scores(optional)

      for k,v in batch.items():
        batch[k] = v.to(device)
      
      # beam search
      outputs = model.generate(batch["pixel_values"].to(device))
      pred, label = get_pred_and_label_str(outputs, batch["labels"], tokenizer)

      label = [[l] for l in label]
      
      bleu =  corpus_bleu(
              label, pred,
              weights=(0.25, 0.25, 0.25, 0.25),
              smoothing_function=SmoothingFunction().method1
      )
      
      val_bleu += bleu

      if args['wandb'] == True:
        wandb.log({
                    'Val/val_iter_bleu': bleu, 
                }, step=step)
        step += 1


  epoch_bleu = val_bleu / len(eval_dataloader)
  print(f"Val BLEU:{epoch_bleu}")
  if args['wandb'] == True:
    wandb.log({
                'Val/val_bleu': epoch_bleu, 
                }, step=step)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--config_eval', default='config/config_eval.yaml', type=str, help='path of train configuration yaml file')
  
  pre_args = parser.parse_args()

  with open(pre_args.config_eval) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

  if args['wandb'] == True:
    wandb.login()
    wandb.init(project="latex-OCR", entity='gome', name=args['name'])
    wandb.config.update(args)

  compute_bleu(args)