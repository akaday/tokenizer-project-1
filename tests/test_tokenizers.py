import unittest
import subprocess
import requests
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
import sentencepiece as spm
from transformers import BertTokenizerFast

class TestTokenizers(unittest.TestCase):

    def setUp(self):
        self.dataset_path = "test_dataset.txt"
        with open(self.dataset_path, "w") as f:
            f.write("This is a test dataset for tokenizers.\n" * 100)

    def tearDown(self):
        os.remove(self.dataset_path)

    def test_bpe_tokenizer(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.train(files=[self.dataset_path], trainer=trainer)
        tokenizer.save("tokenizer_bpe.json")
        loaded_tokenizer = Tokenizer.from_file("tokenizer_bpe.json")
        output = loaded_tokenizer.encode("This is a test.").tokens
        self.assertIsInstance(output, list)
        os.remove("tokenizer_bpe.json")

    def test_wordpiece_tokenizer(self):
        tokenizer = Tokenizer(WordPiece())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.train(files=[self.dataset_path], trainer=trainer)
        tokenizer.save("tokenizer_wp.json")
        loaded_tokenizer = Tokenizer.from_file("tokenizer_wp.json")
        output = loaded_tokenizer.encode("This is a test.").tokens
        self.assertIsInstance(output, list)
        os.remove("tokenizer_wp.json")

    def test_unigram_tokenizer(self):
        tokenizer = Tokenizer(Unigram())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = UnigramTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer.train(files=[self.dataset_path], trainer=trainer)
        tokenizer.save("tokenizer_unigram.json")
        loaded_tokenizer = Tokenizer.from_file("tokenizer_unigram.json")
        output = loaded_tokenizer.encode("This is a test.").tokens
        self.assertIsInstance(output, list)
        os.remove("tokenizer_unigram.json")

    def test_sentencepiece_tokenizer(self):
        spm.SentencePieceTrainer.train(input=self.dataset_path, model_prefix='tokenizer_sp', vocab_size=32000, model_type='bpe', user_defined_symbols=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        tokenizer = spm.SentencePieceProcessor(model_file='tokenizer_sp.model')
        output = tokenizer.encode_as_pieces("This is a test.")
        self.assertIsInstance(output, list)
        os.remove("tokenizer_sp.model")
        os.remove("tokenizer_sp.vocab")

    def test_bert_tokenizer(self):
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        output = tokenizer.encode("This is a test.")
        self.assertIsInstance(output, list)

    def test_cli_bpe_tokenizer(self):
        subprocess.run(["python", "tokenizer_cli.py", "--model", "bpe", "--dataset", self.dataset_path, "--output", "tokenizer_bpe.json"])
        output = subprocess.check_output(["python", "tokenizer_cli.py", "--model", "bpe", "--tokenizer", "tokenizer_bpe.json", "--text", "This is a test."])
        self.assertIn("[CLS]", output.decode())
        os.remove("tokenizer_bpe.json")

    def test_web_bpe_tokenizer(self):
        requests.post("http://localhost:5000/train", json={"model": "bpe", "dataset": self.dataset_path, "output": "tokenizer_bpe.json"})
        response = requests.post("http://localhost:5000/tokenize", json={"model": "bpe", "text": "This is a test.", "tokenizer_path": "tokenizer_bpe.json"})
        self.assertIn("[CLS]", response.json()["tokens"])
        os.remove("tokenizer_bpe.json")

if __name__ == "__main__":
    unittest.main()
