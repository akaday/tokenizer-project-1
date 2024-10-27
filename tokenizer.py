from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.processors import TemplateProcessing
import sentencepiece as spm
from transformers import BertTokenizer, BertTokenizerFast

# Initialiser le tokenizer BPE
tokenizer_bpe = Tokenizer(BPE())
tokenizer_bpe.pre_tokenizer = Whitespace()

# Entraîner le tokenizer BPE
trainer_bpe = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer_bpe.train(files=["path/to/your/dataset.txt"], trainer=trainer_bpe)

# Sauvegarder le tokenizer BPE
tokenizer_bpe.save("tokenizer_bpe.json")

# Initialiser le tokenizer WordPiece
tokenizer_wp = Tokenizer(WordPiece())
tokenizer_wp.pre_tokenizer = Whitespace()

# Entraîner le tokenizer WordPiece
trainer_wp = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer_wp.train(files=["path/to/your/dataset.txt"], trainer=trainer_wp)

# Sauvegarder le tokenizer WordPiece
tokenizer_wp.save("tokenizer_wp.json")

# Initialiser le tokenizer Unigram
tokenizer_unigram = Tokenizer(Unigram())
tokenizer_unigram.pre_tokenizer = Whitespace()

# Entraîner le tokenizer Unigram
trainer_unigram = UnigramTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer_unigram.train(files=["path/to/your/dataset.txt"], trainer=trainer_unigram)

# Sauvegarder le tokenizer Unigram
tokenizer_unigram.save("tokenizer_unigram.json")

# Initialiser et entraîner le tokenizer SentencePiece
spm.SentencePieceTrainer.train(input='path/to/your/dataset.txt', model_prefix='tokenizer_sp', vocab_size=32000, model_type='bpe', user_defined_symbols=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# Charger le tokenizer SentencePiece
tokenizer_sp = spm.SentencePieceProcessor(model_file='tokenizer_sp.model')

# Initialiser et entraîner le tokenizer BERT
tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenisation multi-langues
normalizer = normalizers.Sequence([NFD(), StripAccents()])
tokenizer_bpe.normalizer = normalizer
tokenizer_wp.normalizer = normalizer
tokenizer_unigram.normalizer = normalizer

# Post-traitement pour BERT
tokenizer_bert._tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer_bert.convert_tokens_to_ids("[CLS]")),
        ("[SEP]", tokenizer_bert.convert_tokens_to_ids("[SEP]")),
    ],
)
