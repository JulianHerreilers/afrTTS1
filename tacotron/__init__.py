__version__ = "0.1.1"

from .dataset import BucketBatchSampler, TTSDataset, pad_collate
from .model import Tacotron
from .text import load_afrdict, symbol_to_id, text_to_id

__all__ = [
    Tacotron,
    TTSDataset,
    BucketBatchSampler,
    load_afrdict,
    text_to_id,
    symbol_to_id,
    pad_collate,
]
