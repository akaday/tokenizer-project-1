from flask import Flask, request, jsonify
import sentencepiece as spm
from transformers import BertTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
import tokenizer  # Import the C++ tokenizer module

app = Flask(__name__)

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

@app.route('/train', methods=['POST'])
def train_tokenizer():
    data = request.json
    model = data.get('model')
    dataset = data.get('dataset')
    output = data.get('output')

    if model == "bpe":
        train_bpe_tokenizer(dataset, output)
    elif model == "wordpiece":
        train_wordpiece_tokenizer(dataset, output)
    elif model == "unigram":
        train_unigram_tokenizer(dataset, output)
    elif model == "sentencepiece":
        train_sentencepiece_tokenizer(dataset, output)
    elif model == "bert":
        tokenizer = train_bert_tokenizer()
        tokenizer.save_pretrained(output)

    return jsonify({"message": "Tokenizer trained successfully!"})

@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.json
    model = data.get('model')
    text = data.get('text')
    tokenizer_path = data.get('tokenizer_path')

    if model == "bpe":
        tokenizer = load_bpe_tokenizer(tokenizer_path)
    elif model == "wordpiece":
        tokenizer = load_wordpiece_tokenizer(tokenizer_path)
    elif model == "unigram":
        tokenizer = load_unigram_tokenizer(tokenizer_path)
    elif model == "sentencepiece":
        tokenizer = load_sentencepiece_tokenizer(tokenizer_path)
    elif model == "bert":
        tokenizer = load_bert_tokenizer(tokenizer_path)
    elif model == "cpp":
        output = tokenizer.tokenize(text)
        return jsonify({"tokens": output})

    if model in ["bpe", "wordpiece", "unigram"]:
        output = tokenizer.encode(text).tokens
    elif model == "sentencepiece":
        output = tokenizer.encode_as_pieces(text)
    elif model == "bert":
        output = tokenizer.encode(text)

    return jsonify({"tokens": output})

if __name__ == '__main__':
    app.run(debug=True)
