import argparse
import sentencepiece as spm
from transformers import BertTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.processors import TemplateProcessing

def train_bpe_tokenizer(dataset_path, output_path):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(files=[dataset_path], trainer=trainer)
    tokenizer.save(output_path)

def train_wordpiece_tokenizer(dataset_path, output_path):
    tokenizer = Tokenizer(WordPiece())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(files=[dataset_path], trainer=trainer)
    tokenizer.save(output_path)

def train_unigram_tokenizer(dataset_path, output_path):
    tokenizer = Tokenizer(Unigram())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = UnigramTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(files=[dataset_path], trainer=trainer)
    tokenizer.save(output_path)

def train_sentencepiece_tokenizer(dataset_path, output_prefix):
    spm.SentencePieceTrainer.train(input=dataset_path, model_prefix=output_prefix, vocab_size=32000, model_type='bpe', user_defined_symbols=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

def train_bert_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return tokenizer

def load_bpe_tokenizer(tokenizer_path):
    return Tokenizer.from_file(tokenizer_path)

def load_wordpiece_tokenizer(tokenizer_path):
    return Tokenizer.from_file(tokenizer_path)

def load_unigram_tokenizer(tokenizer_path):
    return Tokenizer.from_file(tokenizer_path)

def load_sentencepiece_tokenizer(tokenizer_path):
    return spm.SentencePieceProcessor(model_file=tokenizer_path)

def load_bert_tokenizer(tokenizer_path):
    return BertTokenizerFast.from_pretrained(tokenizer_path)

def main():
    parser = argparse.ArgumentParser(description="Train and use tokenization models")
    parser.add_argument("--model", type=str, required=True, choices=["bpe", "wordpiece", "unigram", "sentencepiece", "bert"], help="The tokenization model to train or use")
    parser.add_argument("--dataset", type=str, help="Path to the dataset file")
    parser.add_argument("--output", type=str, help="Path to save the trained tokenizer")
    parser.add_argument("--tokenizer", type=str, help="Path to the trained tokenizer")
    parser.add_argument("--text", type=str, help="Text to tokenize")
    args = parser.parse_args()

    if args.dataset and args.output:
        if args.model == "bpe":
            train_bpe_tokenizer(args.dataset, args.output)
        elif args.model == "wordpiece":
            train_wordpiece_tokenizer(args.dataset, args.output)
        elif args.model == "unigram":
            train_unigram_tokenizer(args.dataset, args.output)
        elif args.model == "sentencepiece":
            train_sentencepiece_tokenizer(args.dataset, args.output)
        elif args.model == "bert":
            tokenizer = train_bert_tokenizer()
            tokenizer.save_pretrained(args.output)
    elif args.tokenizer and args.text:
        if args.model == "bpe":
            tokenizer = load_bpe_tokenizer(args.tokenizer)
        elif args.model == "wordpiece":
            tokenizer = load_wordpiece_tokenizer(args.tokenizer)
        elif args.model == "unigram":
            tokenizer = load_unigram_tokenizer(args.tokenizer)
        elif args.model == "sentencepiece":
            tokenizer = load_sentencepiece_tokenizer(args.tokenizer)
        elif args.model == "bert":
            tokenizer = load_bert_tokenizer(args.tokenizer)

        if args.model in ["bpe", "wordpiece", "unigram"]:
            output = tokenizer.encode(args.text).tokens
        elif args.model == "sentencepiece":
            output = tokenizer.encode_as_pieces(args.text)
        elif args.model == "bert":
            output = tokenizer.encode(args.text)

        print(output)

if __name__ == "__main__":
    main()
