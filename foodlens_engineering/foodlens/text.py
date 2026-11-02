import re
from .schema import normalize_name

SPECIAL = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", normalize_name(text), flags=re.UNICODE)


def detokenize(tokens):
    text = " ".join(tokens)
    # Opening brackets keep the preceding word separator, but close onto content.
    text = re.sub(r"\s+([^\w\s(\[{])", r"\1", text)
    text = re.sub(r"([\(\[{])\s+", r"\1", text)
    text = re.sub(r"(?<=[\-'’/])\s+(?=\w)", "", text)
    return normalize_name(text)


class Vocabulary:
    def __init__(self, tokens=None):
        self.tokens = list(tokens or SPECIAL)
        if len(set(self.tokens)) != len(self.tokens) or self.tokens[:4] != list(SPECIAL):
            raise ValueError("vocabulary must retain unique PAD/BOS/EOS/UNK IDs")
        self.index = {word: i for i, word in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def extend_training(self, texts, *, split):
        if split != "train":
            raise ValueError("vocabulary may only be fit on training text")
        additions = sorted({token for text in texts for token in tokenize(text)} - set(self.index))
        self.tokens.extend(additions)
        self.index = {word: i for i, word in enumerate(self.tokens)}

    def add_sep(self):
        if "<SEP>" not in self.index:
            self.tokens.append("<SEP>")
            self.index["<SEP>"] = len(self.tokens) - 1

    def encode(self, text):
        return [self.index.get(token, 3) for token in tokenize(text)]
