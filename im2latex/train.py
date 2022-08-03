import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig

import argparse
import os
from tqdm import tqdm
import numpy as np
import random

from dataset import prepare_dataset

from transformers import AdamW

from metric import compute_cer
from metric import get_pred_and_label_str
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from datasets import load_metric
cer_metric = load_metric("cer")

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

def train(args):
  seed_everything(args['seed'])
  
  # settings
  print("pytorch version: {}".format(torch.__version__))
  gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
  print("GPU 사용 가능 여부: {}".format(gpu))
  print(torch.cuda.get_device_name(0))
  print(torch.cuda.device_count())

  train_dataset, val_dataset, test_dataset, tokenizer, processor = \
  prepare_dataset(data_dir = args['data_dir'], max_length_token=args['max_length_token'], \
                  vocab_size=args['vocab_size'], processor_path=args['processor_path'])

  train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'])
  test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'])

  device = torch.device("cuda" if torch.cuda.is_available() \
                        else "mps" if torch.backends.mps.is_available() else "cpu")

  print(f"----- Use pretrained weight : {args['model_path']} -----")
  print(f"----- Use pretrained processor : {args['processor_path']} -----")
  model_config = VisionEncoderDecoderConfig.from_pretrained(args['model_path'])
  model = VisionEncoderDecoderModel.from_pretrained(args['model_path'],
                                  config=model_config)
  model.to(device)
  if args['wandb'] == True:
    wandb.watch(model)
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

  optimizer = AdamW(model.parameters(), lr=5e-5)

  step = 0
  for epoch in range(args['num_epoch']):  # loop over the dataset multiple times
    # train
    model.train()
    train_loss = 0.0
    for i, batch in enumerate(tqdm(train_dataloader)):
      # get the inputs
      # batch: {'pixel_values': (batch_size, 3, 384, 384), 'labels':(batch_size, 100)}
      for k,v in batch.items():
        batch[k] = v.to(device)

      # forward + backward + optimize
      # outputs : (loss, logit)  
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss.item()
      if args['wandb'] == True:
        wandb.log({'Train/train_loss': loss.item(), 'epoch':epoch}, step=step)
        step += 1
      
      if i % args['report_step'] == 0: 
        print(f"Loss: {loss.item()}")
        
    print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
    # validate
    print(f'Start {epoch}th evaluation!!!')
    model.eval()
    # val_loss = 0.0
    val_cer = 0.0
    val_bleu = 0.0
    best_cer = int(1e9)
    with torch.no_grad():
      for i, batch in enumerate(tqdm(val_dataloader)):
        # generate ver.
        # generate output : sequence of token, scores(optional)
        # outputs = model.generate(batch["pixel_values"].to(device))
        
        # greedy ver.
        # prediction = torch.argmax(outputs.logits, dim=2)

        # loss = outputs.loss.item()
        # val_loss += loss

        for k,v in batch.items():
          batch[k] = v.to(device)
        outputs = model(**batch)

        prediction = model.generate(batch["pixel_values"].to(device))

        pred_str, label_str = get_pred_and_label_str(prediction, batch["labels"], tokenizer)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        label_str = [[l] for l in label_str]
        
        bleu =  corpus_bleu(
                label_str, pred_str,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=SmoothingFunction().method1
        )
        val_cer += cer
        val_bleu += bleu
        
    # epoch_loss = val_loss / len(val_dataloader)
    epoch_cer = val_cer / len(val_dataloader)
    epoch_bleu = val_bleu / len(val_dataloader)

    # print(f"{epoch}th epoch Val Loss:{epoch_loss}")
    print(f"{epoch}th epoch Val CER:{epoch_cer}")
    print(f"{epoch}th epoch Val BLEU:{epoch_bleu}")
    if args['wandb'] == True:
      wandb.log({
                  # 'Val/val_loss': epoch_loss, 
                  'Val/val_cer': epoch_cer,
                  'Val/val_bleu': epoch_bleu,
                  'epoch':epoch}, step=step)
    
    # if epoch_cer < best_cer:
    #   best_cer = epoch_cer
    #   model.save_pretrained(f"model/{args['name']}/epoch_{epoch}")
    #   processor.save_pretrained(f"model/{args['name']}/epoch_{epoch}")
    
    model.save_pretrained(f"model/{args['name']}/ep{epoch}_b{epoch_bleu:.2f}_c{epoch_cer:.2f}")
    processor.save_pretrained(f"model/{args['name']}/ep{epoch}_b{epoch_bleu:.2f}_c{epoch_cer:.2f}")
    


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--config_train', default='config/config_train.yaml', type=str, help='path of train configuration yaml file')
  
  pre_args = parser.parse_args()

  with open(pre_args.config_train) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

  if args['wandb'] == True:
    wandb.login()
    wandb.init(project="latex-OCR", entity='gome', name=args['name'])
    wandb.config.update(args)

  train(args)