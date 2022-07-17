import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel

import argparse
import os
from tqdm import tqdm

from tokenizer import tokenizer
from dataset import prepare_dataset

from datasets import load_metric
from transformers import AdamW

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", default='./data')
# tokenize
parser.add_argument("-m", "--max_length_token", default=100)
parser.add_argument("-v", "--vocab_size", default=600)
# dataset
parser.add_argument("-b", "--batch_size", default=16)
# training
parser.add_argument("-i", "--version", default=5)
parser.add_argument("-e", "--num_epoch", default=5)
parser.add_argument("-r", "--report_step", default=100)
args = parser.parse_args()

# Set up environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

train_dataset, valid_dataset, eval_dataset, tokenizer = prepare_dataset(data_dir = args.data_dir, max_length_token=args.max_length_token, vocab_size=args.vocab_size)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
validate_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-stage1")
model.to(device)

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

cer_metric = load_metric("cer")

def compute_cer(pred_ids, label_ids):
    pred_str = tokenizer.decode_batch(pred_ids.tolist(), skip_special_tokens=True)
    label_ids[label_ids == -100] = tokenizer.token_to_id("[PAD]")
    label_str = tokenizer.decode_batch(label_ids.tolist(), skip_special_tokens=True)
    
    # Filter out empty label strings
#    for l in enumerate(label_str):
#        if not l:
#            label_str.pop(l)
#            pred_str.pop(l)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(args.num_epoch):  # loop over the dataset multiple times
   # train
   model.train()
   train_loss = 0.0
   for i, batch in enumerate(tqdm(train_dataloader)):
      # get the inputs
      for k,v in batch.items():
        batch[k] = v.to(device)

      # forward + backward + optimize
      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss.item()
      if i % args.report_step == 0: print(f"Loss: {loss.item()}") 

   print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
   model.save_pretrained(f"version_{args.version}/epoch_{epoch}")
    

model.save_pretrained(f"version_{args.version}/final")
