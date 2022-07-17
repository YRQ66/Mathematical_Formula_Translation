from os.path import join
from preprocess import preprocess_df

# Create and train word level tokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

def tokenizer(formulas_file, data_dir, max_length = None, vocab_size = None):

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.enable_padding(length=max_length)
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                        vocab_size=vocab_size,
                        show_progress=True,
                        )

    files = [formulas_file]
    tokenizer.train(files, trainer)
    tokenizer.save(join(data_dir, "tokenizer-wordlevel.json"))
    return tokenizer

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default='./data')
    # tokenize
    parser.add_argument("-m", "--max_length_token", default=100)
    parser.add_argument("-v", "--vocab_size", default=600)
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

    tokenizer = tokenizer(formulas_file = formulas_file, data_dir = args.data_dir, max_length = args.max_length_token, vocab_size=args.vocab_size)

    # Sanity check of tokenizer
    i = 5

    print(tokenizer.encode(train_df.loc[i, 'text']).tokens )
    print(tokenizer.encode(train_df.loc[i, 'text']).ids )
    print(tokenizer.token_to_id("[PAD]") )
