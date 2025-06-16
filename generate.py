#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse

from helpers import *
from model import *

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, device='cpu'):
    hidden = decoder.init_hidden(1)

    # safe device sending
    if isinstance(hidden, tuple):
        hidden = tuple(h.to(device) for h in hidden)
    else:
        hidden = hidden.to(device)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0)).to(device)

    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char

        inp = Variable(char_tensor(predicted_char).unsqueeze(0)).to(device)

    return predicted

# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help="Device to use for generation: 'cpu', 'cuda', or 'mps'")

    args = argparser.parse_args()

    device = resolve_device(args.device)  # From helpers.py
    decoder = torch.load(args.filename, weights_only=False)
    decoder.to(device)

    # ignore filename argument and specifically pass device object instead of str to generate() call
    del args.filename
    args.device = device

    print(generate(decoder, **vars(args)))

