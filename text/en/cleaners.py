"""Pipeline for English text including text normalization and abbreviations expansion"""

import re

from text.en.normalization.abbreviations import abbreviations_en
from text.en.normalization.number_norm import normalize_numbers
from text.en.normalization.time_norm import expand_time_english

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()


def expand_abbreviations(text):
    for regex, replacement in abbreviations_en:
        text = re.sub(regex, replacement, text)

    return text


def remove_aux_symbols(text):
    text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
    
    return text


def replace_symbols(text):
    text = text.replace(";", ",")
    text = text.replace("-", " ")
    text = text.replace(":", ",")
    text = text.replace("&", " and ")

    return text


def cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion
    """
    text = lowercase(text)
    text = expand_time_english(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    
    return text
