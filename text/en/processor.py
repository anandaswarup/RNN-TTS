"""Text processor for English"""

import re
from itertools import islice

from text.en.cleaners import clean_text

_pad = "_PAD_"
_unk = "_UNK_"
_bos = "_BOS_"
_eos = "_EOS_"
_wb = "#"
_punctuation = "!,.?"
_english_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_cmudict_symbols = [
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
_arpabet = ["@" + s for s in _cmudict_symbols]

# Get list of symbols to be used for text processing and create mapping
symbols = [_pad, _bos, _eos, _wb, _unk] + list(_punctuation) + list(_english_characters) + _arpabet

symbol_to_id = {symb: index for index, symb in enumerate(symbols)}

# Regular expression for tokenizing text
tokenizer_pattern = re.compile(r"[\w\{\}']+|[.,!?]")


def tokenize_text(text):
    return tokenizer_pattern.findall(text)


def load_cmudict():
    """Load the CMU pronunciation dictionary
    """
    with open("text/en/cmudict-0.7b.txt", encoding="ISO-8859-1") as file_reader:
        cmudict = (line.strip().split("  ") for line in islice(file_reader, 126, 133905))

        cmudict = {word: pronunciation for word, pronunciation in cmudict}

    return cmudict


def parse_text(text, cmudict):
    """Parse the text to get the sequence of phonemes for words in the CMUDict. For OOV words backoff to character
    sequence instead of phoneme sequence
    """
    text_seq = []

    # Normalize the text
    text = tokenize_text(clean_text(text))

    # Get the sequence of phonemes for words in the text while explicitly marking word boundaries. Incase of OOV words
    # backoff to using character sequence
    for word in text:
        if word.upper() in cmudict:
            text_seq.append(" ".join(["@" + s for s in cmudict[word.upper()].split(" ")]))
        else:
            text_seq.append(" ".join(char for char in word))

        if word not in _punctuation:
            text_seq.append(_wb)

    text_seq = [word.split(" ") for word in text_seq]
    text_seq = [char for word in text_seq for char in word]

    # Insert _bos and _eos symbols
    text_seq.insert(0, _bos)
    text_seq.append(_eos)

    return text_seq


def text_to_sequence(text, cmudict):
    """Convert the text to a sequence of IDs corresponding to the symbols present in the text
    """
    text_seq = parse_text(text, cmudict)
    id_seq = [symbol_to_id[s] if s in symbol_to_id else symbol_to_id[_unk] for s in text_seq]

    return id_seq
