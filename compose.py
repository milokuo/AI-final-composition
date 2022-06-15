from tokenize import String
import numpy as np
import pandas as pd
from train import LSTM, sample, seq2csv

import torch
from torch import nn
import torch.nn.functional as F
import plac


@plac.annotations(
    input=("Name of input model", "option", "i", str),
    output=("Name of output music", "option", "o", str),
)

def main(input="default_model", output="mymusic"):
    checkpoint = torch.load(open(f'models/{input}.pth', 'rb'), map_location='cpu')    
    net = LSTM(checkpoint['tokens'], hidden_size=checkpoint['hidden_size'], layers_size=checkpoint['layers_size'])
    net.load_state_dict(checkpoint['state_dict'])

    fname = output    # File save name
    prime = "A4" # Prime for the RNN
    top_k = 3            # Take top k prediction to randomly choose from
    compose_len = 1500   # Length of sequence to compose

    channel = [0]        # MIDI Channels

    seqs = {}
    idx_retry = 0
    while True:
        assert max(channel) <= 15
        try:
            for i in range(len(channel)):
                seq = sample(net, compose_len, prime=prime, top_k=top_k)
                seq = " ".join(seq.split()[:-1])
                seqs[i+1] = seq
            seq2csv(seqs, fname, channel)
            print(seqs)
            break
        except:
            idx_retry += 1
            print(f"Retry music composing... [{idx_retry}]")
            if idx_retry == 10:
                print("Music composition failed. Try to train the model longer")            
                break



if __name__ == '__main__':
    plac.call(main)