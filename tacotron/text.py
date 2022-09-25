""" adapted from https://github.com/keithito/tacotron """

import re
from itertools import islice

import importlib_resources

# fmt: off
PUNCTUATION = ['!', ',', '.', '?']
SYMBOLS = [
    '_', '~', ' ', *PUNCTUATION, '@', 'r', 's', 't', 'n', 'l', 'k', 'x', 'd', 'a',
     'f', 'i', 'A:', 'i@', 'm', 'O', 'p', 'b', 'u@', 'v', '@i', 'N', '{', 'E', 'u',
      'h_', 'y', '9', '9y', 'w', 'j', 'S', '2:', '@u', 'g', 'z', 'Z'
]
# fmt: on

symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}
id_to_symbol = {i: s for i, s in enumerate(SYMBOLS)}

abbreviations = [
    (re.compile(fr"\b{abbreviation}\.", re.IGNORECASE), replacement.upper())
    for abbreviation, replacement in [
        ("mrs", "missis"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
        ("etc", "etcetera"),
    ]
]
parentheses_pattern = re.compile(r"(?<=[.,!?] )[\(\[]|[\)\]](?=[.,!?])|^[\(\[]|[\)\]]$")
dash_pattern = re.compile(r"(?<=[.,!?] )-- ")
alt_entry_pattern = re.compile(r"(?<=\w)\((\d)\)")
tokenizer_pattern = re.compile(r"[\w\{\}']+|[.,!?]")


def expand_abbreviations(text):
    for pattern, replacement in abbreviations:
        text = pattern.sub(replacement, text)
    return text


def format_alt_entry(text):
    return alt_entry_pattern.sub(r"{\1}", text)


def replace_symbols(text):
    # replace semi-colons and colons with commas
    text = text.replace(";", ",")
    text = text.replace(":", ",")

    # replace dashes with commas
    text = dash_pattern.sub("", text)
    text = text.replace(" --", ",")
    text = text.replace(" - ", ", ")

    # split hyphenated words
    text = text.replace("-", " ")

    # use {#} to indicate alternate pronunciations
    text = format_alt_entry(text)

    # replace parentheses with commas
    text = parentheses_pattern.sub("", text)
    text = text.replace(")", ",")
    text = text.replace(" (", ", ")
    text = text.replace("]", ",")
    text = text.replace(" [", ", ")
    return text


def clean(text):
    text = text.lower()
    text = expand_abbreviations(text)
    #text = replace_symbols(text) temporarily removed as test data already formatted
    return text


def tokenize(text):
    return tokenizer_pattern.findall(text)


def load_afrdict():
    """Loads the Afr(local) Pronouncing Dictionary"""

    dict_ref = "afr_za_dict.txt"
    dict_file = open(dict_ref, 'r')
    dict_list = dict_file.readlines()
    dict_file.close()
    afrdict = {}
    for i in range(0, len(dict_list)):
        dict_list[i] = dict_list[i].strip().split()
        entry = " ".join([str(word) for word in dict_list[i][1:]])
        afrdict[str(dict_list[i][0])] = entry
    return afrdict

def parse_text(text, afrdict):
    words = tokenize(clean(text))

    # check if any words are not in the dictionary
    stripped = (word for word in words if word not in PUNCTUATION)
    out_of_vocab = set(word for word in stripped if word not in afrdict)
    if out_of_vocab:
        out_of_vocab_list = ", ".join(out_of_vocab)
        raise KeyError(
            f"Please add {out_of_vocab_list} to the pronunciation dictionary."
        )
    words = (afrdict[word] if word not in PUNCTUATION else word for word in words)
    words = (word.split(" ") for word in words)
    words = (x for word in words for x in (word, [" "]))
    symbols = list(symbol for word in words for symbol in word)
    symbols.append("~")
    return symbols


def text_to_id(text, afrdict):
    """
    Converts text to a sequence of symbol ids.

    Parameters:
        text (string): The input text.
        afrdict (dict): The pronuniation dictionary used for
            grapheme-to-phone conversion

    Returns:
        Tensor: The sequence of symbol ids.
    """
    symbols = parse_text(text, afrdict)
    return [symbol_to_id[symbol] for symbol in symbols]
