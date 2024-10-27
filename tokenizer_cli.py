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

def main():
    parser = argparse.ArgumentParser(description="Train tokenization models")
    parser.add_argument("--model", type=str, required=True, choices=["bpe", "wordpiece", "unigram", "sentencepiece", "bert"], help="The tokenization model to train")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the trained tokenizer")
    args = parser.parse_args()

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

if __name__ == "__main__":
    main()
