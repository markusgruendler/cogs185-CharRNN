# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import random
import time
import math
import torch

# Reading and un-unicode-encoding data

all_characters = string.printable
n_characters = len(all_characters)

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

# Turning a string into a tensor

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Reading device argument from CLI

def resolve_device(requested: str = "cpu") -> torch.device:
    """
    Resolve and return a valid torch.device based on availability.

    Parameters:
        requested (str): 'cpu', 'cuda', or 'mps'

    Returns:
        torch.device: A usable device (may fallback to 'cpu')
    """
    requested = requested.lower()
    device_checks = {
        'cuda': lambda: torch.cuda.is_available(),
        'mps':  lambda: torch.backends.mps.is_available(),
        'cpu':  lambda: True
    }

    if requested not in device_checks:
        raise ValueError(f"Unsupported device: '{requested}'")

    if device_checks[requested]():
        device = torch.device(requested)
    else:
        print(f"Requested device '{requested}' not available. Falling back to CPU.")
        device = torch.device("cpu")

    # print(f"Using device: {device}")
    return device
