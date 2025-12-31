# =========================
# GPT-2 style byte-level BPE
# =========================
from collections import Counter
from typing import List, Dict, Tuple

# ----- Byte <-> Unicode mapping (GPT-2 style) -----

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) \
       + list(range(ord("¡"), ord("¬")+1)) \
       + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

BYTE_TO_UNICODE: Dict[int, str] = bytes_to_unicode()
UNICODE_TO_BYTE: Dict[str, int] = {v: k for k, v in BYTE_TO_UNICODE.items()}


def text_to_gpt_chars(text: str) -> str:
    """UTF-8 text -> GPT-style 'alphabet string' (1 char per byte)."""
    b = text.encode('utf-8')
    return "".join(BYTE_TO_UNICODE[byte] for byte in b)


def gpt_chars_to_text(s: str) -> str:
    """GPT-style 'alphabet string' -> UTF-8 text."""
    b = bytes([UNICODE_TO_BYTE[c] for c in s])
    return b.decode("utf-8", errors='replace')


class GPT2BPE:
    def __init__(self, vocab_size: int = 50257, max_merges: int = 50000):
        """
        vocab_size: desired final vocab size (including base 256 byte tokens)
        max_merges: safety cap on number of BPE merge steps
        """
        self.vocab_size = vocab_size
        self.max_merges = max_merges

        # vocab maps token string -> integer id.
        # token strings are *GPT-char strings* (possibly multi-char).
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # BPE merge rules: list of pairs of token-strings, in order
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        self.trained = False

    def _init_base_vocab(self):
        """
        Initialize vocab with 256 byte tokens, each represented as a
        single GPT-char string.
        """
        for b in range(0, 256):
            ch = BYTE_TO_UNICODE[b]
            self.token_to_id[ch] = b
            self.id_to_token[b] = ch

    # ----- BPE training helpers -----

    def _get_pair_stats(self, corpus_tokens: List[List[str]]) -> Counter:
        """
        Count frequency of adjacent token pairs over the whole corpus.

        corpus_tokens: list of sentences, each sentence is a list of token strings.
        Example sentence: ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
        """
        pair_counts = Counter()
        for sent in corpus_tokens:
            for i in range(0, len(sent) - 1):
                pair_counts[(sent[i], sent[i+1])] += 1
        return pair_counts

    def _merge_pair_in_corpus(
        self,
        corpus_tokens: List[List[str]],
        pair: Tuple[str, str],
        new_token: str,
    ) -> List[List[str]]:
        """
        Replace all occurrences of 'pair' with 'new_token' in corpus_tokens.
        Return a new corpus_tokens list.
        """
        new_corpus_tokens = []
        for sent in corpus_tokens:
            cur = []
            i = 0
            while i < len(sent) - 1:
                if (sent[i], sent[i+1]) == pair:
                    cur.append(new_token)
                    i += 2
                else:
                    cur.append(sent[i])
                    i += 1
            if i == len(sent) - 1: cur.append(sent[-1])
            new_corpus_tokens.append(cur)
        return new_corpus_tokens

    # ----- Training -----

    def train(self, texts: List[str]):
        """
        Train BPE merges from a list of raw text strings.
        Steps:
          1) Initialize base vocab (256 byte tokens).
          2) Convert each text -> GPT-chars string -> list of chars.
          3) Iteratively:
             - count pair stats
             - pick most frequent pair
             - create new token (concatenation of the pair)
             - update vocab and corpus tokenization
             - record merge in self.merges
        """
        # 1) base vocab
        self._init_base_vocab()

        # 2) corpus as lists of "tokens" (initially single GPT-chars)
        corpus_tokens: List[List[str]] = []
        for text in texts:
            g = text_to_gpt_chars(text)
            corpus_tokens.append(list(g))

        # 3) BPE loop
        next_id = len(self.token_to_id)
        while len(self.token_to_id) < self.vocab_size and len(self.merges) < self.max_merges:
            pair_counts = self._get_pair_stats(corpus_tokens)
            if not pair_counts:
                break

            top_pair, freq = pair_counts.most_common(1)[0]
            if freq < 2:
                break

            new_token = top_pair[0] + top_pair[1]
            self.token_to_id[new_token] = next_id
            self.id_to_token[next_id] = new_token
            self.merges.append(top_pair)
            next_id += 1

            corpus_tokens = self._merge_pair_in_corpus(corpus_tokens, top_pair, new_token)

        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.trained = True

    # ----- Encoding / decoding -----

    def _bpe_tokenize_chars(self, s: str) -> List[str]:
        """
        Apply learned BPE merges to a GPT-char string, producing a list of token strings.
        This is the greedy "merge best pairs first" procedure.
        """
        tokens = list(s)
        while True:
            best_pair = None
            best_rank = float('inf')

            # find best-ranked pair among adjacent pairs
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                rank = self.merge_ranks.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            # no merge applies → stop
            if best_rank == float('inf'):
                break

            # merge ALL occurrences of best_pair
            new_tokens = []
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            # handle final token
            if i == len(tokens) - 1:
                new_tokens.append(tokens[-1])

            tokens = new_tokens

        return tokens


    def encode(self, text: str) -> List[int]:
        """
        Text -> list of token IDs.
        """
        if not self.trained:
            raise RuntimeError("Call train() first")

        g = text_to_gpt_chars(text)
        token_strs = self._bpe_tokenize_chars(g)
        return [self.token_to_id[x] for x in token_strs]

    def decode(self, ids: List[int]) -> str:
        """
        List of token IDs -> text.
        """
        tokens = [self.id_to_token[x] for x in ids]
        char_str = "".join(tokens)
        return gpt_chars_to_text(char_str)