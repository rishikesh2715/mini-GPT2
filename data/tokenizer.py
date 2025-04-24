import json
import re
from collections import defaultdict
import time

class GPT2Tokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []
        self.unk_token = "[UNK]"
        self.unk_token_id = None

    def get_stats(self, word_freqs):
        """Calculate pair frequencies across all words."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def merge_pair(self, pair, word_freqs):
        """Merge the most frequent pair in all words."""
        new_word_freqs = defaultdict(int)
        pair_str = ''.join(pair)
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(pair_str)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] += freq
        return new_word_freqs

    def train(self, text):
        """Train the BPE tokenizer on the input text."""
        start_time = time.time()
        print("Initializing word frequencies...")
        # Split into words and convert to character tuples
        words = re.findall(r'\S+|\s', text)
        word_freqs = defaultdict(int)
        for word in words:
            word_freqs[tuple(word)] += 1

        # Initialize base vocabulary with unique characters and [UNK]
        print("Building base vocabulary...")
        chars = set()
        for word in word_freqs:
            for char in word:
                chars.add(char)
        self.token_to_id = {char: i for i, char in enumerate(sorted(chars))}
        self.token_to_id[self.unk_token] = len(self.token_to_id)
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        # Perform BPE merges
        num_merges = self.vocab_size - len(self.token_to_id)
        print(f"Performing {num_merges} merges...")
        for merge_idx in range(num_merges):
            pairs = self.get_stats(word_freqs)
            if not pairs:
                print("No more pairs to merge.")
                break
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self.merge_pair(best_pair, word_freqs)
            new_token = ''.join(best_pair)
            new_id = len(self.token_to_id)
            self.token_to_id[new_token] = new_id
            self.id_to_token[new_id] = new_token
            self.merges.append(best_pair)
            if merge_idx % 100 == 0:
                print(f"Merge {merge_idx}/{num_merges}, time elapsed: {time.time() - start_time:.2f}s")
        
        print(f"Training completed in {time.time() - start_time:.2f}s")

    def encode(self, text):
        """Encode text using learned BPE merges."""
        if not self.token_to_id:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        start_time = time.time()
        print("Encoding text...")
        words = re.findall(r'\S+|\s', text)
        encoded = []
        for word_idx, word in enumerate(words):
            tokens = list(word)
            # Apply merges in order
            for pair in self.merges:
                i = 0
                while i < len(tokens) - 1:
                    if (tokens[i], tokens[i + 1]) == pair:
                        tokens[i] = ''.join(pair)
                        tokens.pop(i + 1)
                    else:
                        i += 1
            # Convert tokens to IDs
            for token in tokens:
                encoded.append(self.token_to_id.get(token, self.unk_token_id))
            if word_idx % 10000 == 0:
                print(f"Encoded {word_idx}/{len(words)} words, time elapsed: {time.time() - start_time:.2f}s")
        
        print(f"Encoding completed in {time.time() - start_time:.2f}s")
        return encoded

    def decode(self, tokens):
        """Decode token IDs back to text."""
        return ''.join(self.id_to_token.get(id, self.unk_token) for id in tokens)

    def save_vocabulary(self, filepath):
        """Save vocabulary and merges to a file."""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'merges': [list(pair) for pair in self.merges],
            'unk_token': self.unk_token
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def load_vocabulary(self, filepath):
        """Load vocabulary and merges from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {int(i): t for t, i in self.token_to_id.items()}
        self.merges = [tuple(pair) for pair in vocab_data['merges']]
        self.unk_token = vocab_data['unk_token']
        self.unk_token_id = self.token_to_id[self.unk_token]
