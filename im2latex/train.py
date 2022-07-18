import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel

import argparse
import os
from tqdm import tqdm

from tokenizer import tokenizer
from dataset import prepare_dataset

from transformers import AdamW

from metric import compute_cer
from metric import get_pred_and_label_str
import nltk

import wandb

# Set up environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(args):
  train_dataset, valid_dataset, eval_dataset, tokenizer = \
  prepare_dataset(data_dir = args.data_dir, max_length_token=args.max_length_token, vocab_size=args.vocab_size)

  train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
  eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
  validate_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-stage1")
  model.to(device)
  wandb.watch(model)
  # set special tokens used for creating the decoder_input_ids from the labels
  model.config.decoder_start_token_id = tokenizer.token_to_id("[CLS]")
  model.config.pad_token_id = tokenizer.token_to_id("[PAD]")
  # make sure vocab size is set correctly
  model.config.vocab_size = args.vocab_size

  # set beam search parameters
  model.config.eos_token_id = tokenizer.token_to_id("[SEP]")
  model.config.max_length = args.max_length_token #64
  model.config.early_stopping = True
  model.config.no_repeat_ngram_size = 3
  model.config.length_penalty = 2.0
  model.config.num_beams = 4

  model.tokenizer = tokenizer

  optimizer = AdamW(model.parameters(), lr=5e-5)

  step = 0
  for epoch in range(args.num_epoch):  # loop over the dataset multiple times
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
      if i % args.report_step == 0: 
        print(f"Loss: {loss.item()}")
        wandb.log({'Train/train_loss': loss.item()})


    print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))

    # evaluate
    model.eval()
    valid_loss = 0.0
    valid_cer = 0.0
    valid_bleu = 0.0 
    candidate_corpus = []
    references_corpus = []
    best_bleu = 0
    with torch.no_grad():
      for i, batch in enumerate(tqdm(eval_dataloader)):
        # run batch generation
        outputs = model(**batch)
        # print(f'1:{outputs}')
        outputs = model.generate(batch["pixel_values"].to(device))

        # print(f'2:{outputs}')
        # compute metrics
        cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"], tokenizer=tokenizer)
        valid_cer += cer

        pred, label = get_pred_and_label_str(outputs, batch["labels"], tokenizer)
          
        for s in pred: s = s.split(" ")
        for s in label: s = s.split(" ")
        candidate_corpus.extend(pred)
        references_corpus.extend(label)
    
        bleu =  nltk.translate.bleu_score.corpus_bleu(
                references_corpus, candidate_corpus,
                weights=(0.25, 0.25, 0.25, 0.25)
        )
        valid_bleu += bleu

    epoch_cer = valid_cer / len(eval_dataloader)
    epoch_bleu = valid_bleu / len(eval_dataloader)
    print(f"Validation CER:{epoch_cer}")
    print(f"Validation BLEU:{epoch_bleu}")
    wandb.log({'Val/val_cer': epoch_cer, 'Val/val_bleu': epoch_bleu, 'epoch':epoch})
    
    if epoch_bleu > best_bleu:
      best_bleu = epoch_bleu
      model.save_pretrained(f"version_{args.version}/epoch_{epoch}")

# model.save_pretrained(f"version_{args.version}/final")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data_dir", default='./data/dataset5/')
  # tokenize
  parser.add_argument("-m", "--max_length_token", default=100)
  parser.add_argument("-v", "--vocab_size", default=600)
  # dataset
  parser.add_argument("-b", "--batch_size", default=16)
  # training
  parser.add_argument("-i", "--version", default=5)
  parser.add_argument("-e", "--num_epoch", default=5, type=int)
  parser.add_argument("-r", "--report_step", default=100)
  args = parser.parse_args()

  wandb.login()
  wandb.init(project="GOME", entity='jiwo-o', name='140k')
  train(args)