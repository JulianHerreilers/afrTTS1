""" adapted from https://github.com/keithito/tacotron """

import re
from itertools import islice

import importlib_resources

# fmt: off
PUNCTUATION = ['!', ',', '.', '?']
SYMBOLS = [
    '_', '~', ' ', *PUNCTUATION, '@', 'r', 's', 't', 'n', 'l', 'k', 'x', 'd', 'a',
     'f', 'i', 'A:', 'i@', 'm', 'O', 'p', 'b', 'u@', 'v', '@i', 'N', '{', 'E', 'u',
      'h\\', 'y', '9', '9y', 'w', 'j', 'S', '2:', '@u', 'g', 'z', 'Z'
]
# fmt: on

# SYMBOLS = [
#     '_', '~', ' ', *PUNCTUATION, 'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0',
#     'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2', 'AO', 'AO0', 'AO1', 'AO2', 'AW',
#     'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH',
#     'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2',
#     'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1', 'IY2', 'JH',
#     'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0', 'OY1',
#     'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
#     'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
# ]

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

    dict_ref = "rcrl_apd.1.4.1.txt"
    dict_file = open(dict_ref, 'r')
    dict_list = dict_file.readlines()
    dict_file.close()
    afrdict = {}
    for i in range(0, 24174):
        dict_list[i] = dict_list[i].strip().split()
        afrdict[dict_list[i][0]] = dict_list[i][1:]
    return afrdict
    # dict_ref = "rcrl_apd.1.4.1.txt"
    # with open(dict_ref, 'r') as file:
    # #with open(dict_ref, encoding="ISO-8859-1") as file:
    #     afrdict = (line.strip().split("  ") for line in islice(file, 0, 24174))
    # file.close()
    #     # afrdict = {
    #     #     format_alt_entry(word): pronunciation for word, pronunciation in afrdict
    #     # }
    # return afrdict


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
    #words = (word.split(" ") for word in words)
    #print(words)
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
