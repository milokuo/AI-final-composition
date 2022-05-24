import os
from tqdm import tqdm
from utils import csv2seq
import numpy as np

csv_source = "dataset"
model_output = "models"


def main():
    if not os.path.exists(model_output):
        os.makedirs(model_output)

    note = {}
    for fname in tqdm(os.listdir(csv_source)):
        note = csv2seq(csv_source, fname, note)
    text = " ".join([track for music in note.values() for track in music.values()])

    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    encoded = np.array([char2int[ch] for ch in text])