"""Text processor for English"""

import re

from text.en.cleaners import clean_text

_pad = "_PAD_"
_unk = "_UNK_"
_bos = "_BOS_"
_eos = "_EOS_"
_wb = "#"
_punctuation = "!'\"(),.:;?-_"
_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

symbols = [_pad, _unk, _bos, _eos, _wb] + list(_punctuation) + list(_alphabet)

_symbol_to_id = {sym: idx for idx, sym in enumerate(symbols)}


def text_to_sequence(text):
    """Convert text to a sequence of IDs corresponding to symbols present in the text
    """
    # Clean and normalize text
    text = clean_text(text)

    # Insert word boundaries and remove whitespaces between words
    text = re.sub(r"(\w+\b)", rf"\1{_wb}", text)
    text = re.sub(r"\s+", "", text)

    # Ensure that text ends with only . ? or !
    text = re.sub(r"[^?.!]$", r".", text)

    text_seq = [_symbol_to_id[sym] if sym in _symbol_to_id else _symbol_to_id[_unk] for sym in text]

    return text_seq
