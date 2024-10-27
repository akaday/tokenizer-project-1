# Tokenizer Project

## Description
Ce projet est conçu pour fournir une implémentation rapide et efficace de tokenizers pour la recherche et la production. Il utilise des modèles de pointe pour le traitement du langage naturel (NLP).

## Fonctionnalités
- **Tokenisation rapide** : Capable de tokeniser un gigaoctet de texte en moins de 20 secondes.
- **Support de plusieurs modèles** : Inclut Byte-Pair Encoding (BPE), WordPiece, Unigram, SentencePiece, et BERT Tokenizer.
- **Personnalisation facile** : Permet de personnaliser la pré-tokenisation et la normalisation.
- **Compatibilité multi-langages** : Fournit des bindings pour Python, Node.js, et plus encore.

## Installation
Pour installer ce projet, vous pouvez cloner le dépôt et installer les dépendances nécessaires :
```bash
git clone https://github.com/akaday/tokenizer-project.git
cd tokenizer-project
npm install
```

## Utilisation
Voici des instructions détaillées pour utiliser les nouveaux modèles de tokenisation :

### Tokenizer BPE
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialiser le tokenizer BPE
tokenizer_bpe = Tokenizer(BPE())
tokenizer_bpe.pre_tokenizer = Whitespace()

# Entraîner le tokenizer BPE
trainer_bpe = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer_bpe.train(files=["path/to/your/dataset.txt"], trainer=trainer_bpe)

# Sauvegarder le tokenizer BPE
tokenizer_bpe.save("tokenizer_bpe.json")
```

### Tokenizer WordPiece
```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialiser le tokenizer WordPiece
tokenizer_wp = Tokenizer(WordPiece())
tokenizer_wp.pre_tokenizer = Whitespace()

# Entraîner le tokenizer WordPiece
trainer_wp = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer_wp.train(files=["path/to/your/dataset.txt"], trainer=trainer_wp)

# Sauvegarder le tokenizer WordPiece
tokenizer_wp.save("tokenizer_wp.json")
```

### Tokenizer Unigram
```python
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialiser le tokenizer Unigram
tokenizer_unigram = Tokenizer(Unigram())
tokenizer_unigram.pre_tokenizer = Whitespace()

# Entraîner le tokenizer Unigram
trainer_unigram = UnigramTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer_unigram.train(files=["path/to/your/dataset.txt"], trainer=trainer_unigram)

# Sauvegarder le tokenizer Unigram
tokenizer_unigram.save("tokenizer_unigram.json")
```

### Tokenizer SentencePiece
```python
import sentencepiece as spm

# Initialiser et entraîner le tokenizer SentencePiece
spm.SentencePieceTrainer.train(input='path/to/your/dataset.txt', model_prefix='tokenizer_sp', vocab_size=32000, model_type='bpe', user_defined_symbols=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# Charger le tokenizer SentencePiece
tokenizer_sp = spm.SentencePieceProcessor(model_file='tokenizer_sp.model')
```

### Tokenizer BERT
```python
from transformers import BertTokenizerFast

# Initialiser et entraîner le tokenizer BERT
tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-uncased')
```

## Exemples et Tutoriels
Voici quelques exemples et tutoriels pour vous aider à démarrer avec les nouveaux modèles de tokenisation :

### Exemple d'utilisation du Tokenizer BPE
```python
# Charger le tokenizer BPE
tokenizer_bpe = Tokenizer.from_file("tokenizer_bpe.json")

# Tokeniser une phrase
output = tokenizer_bpe.encode("Ceci est un exemple de phrase.")
print(output.tokens)
```

### Exemple d'utilisation du Tokenizer SentencePiece
```python
# Charger le tokenizer SentencePiece
tokenizer_sp = spm.SentencePieceProcessor(model_file='tokenizer_sp.model')

# Tokeniser une phrase
output = tokenizer_sp.encode_as_pieces("Ceci est un exemple de phrase.")
print(output)
```

### Exemple d'utilisation du Tokenizer BERT
```python
from transformers import BertTokenizerFast

# Initialiser le tokenizer BERT
tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokeniser une phrase
output = tokenizer_bert.encode("Ceci est un exemple de phrase.")
print(output)
```

## Utilisation des interfaces CLI et Web avec les nouveaux tokenizers

### Interface CLI
Vous pouvez utiliser les nouveaux tokenizers via l'interface en ligne de commande (CLI). Voici quelques exemples :

#### Tokenizer BPE
```bash
python tokenizer_cli.py --model bpe --dataset path/to/your/dataset.txt --output tokenizer_bpe.json
```

#### Tokenizer WordPiece
```bash
python tokenizer_cli.py --model wordpiece --dataset path/to/your/dataset.txt --output tokenizer_wp.json
```

#### Tokenizer Unigram
```bash
python tokenizer_cli.py --model unigram --dataset path/to/your/dataset.txt --output tokenizer_unigram.json
```

#### Tokenizer SentencePiece
```bash
python tokenizer_cli.py --model sentencepiece --dataset path/to/your/dataset.txt --output tokenizer_sp
```

#### Tokenizer BERT
```bash
python tokenizer_cli.py --model bert --dataset path/to/your/dataset.txt --output tokenizer_bert
```

### Interface Web
Vous pouvez également utiliser les nouveaux tokenizers via l'interface web. Voici quelques exemples d'utilisation des endpoints :

#### Entraîner un tokenizer
```bash
curl -X POST http://localhost:5000/train -H "Content-Type: application/json" -d '{
  "model": "bpe",
  "dataset": "path/to/your/dataset.txt",
  "output": "tokenizer_bpe.json"
}'
```

#### Tokeniser un texte
```bash
curl -X POST http://localhost:5000/tokenize -H "Content-Type: application/json" -d '{
  "model": "bpe",
  "text": "Ceci est un exemple de phrase."
}'
```

### Running the Server
To run the server, navigate to the project directory and execute the following command:
```bash
node server.js
```

### Using the Web Interface
1. Open your web browser and navigate to `http://localhost:3000`.
2. Enter the text you want to tokenize in the textarea.
3. Click the "Tokenize" button to see the tokenization results displayed in the result div.
