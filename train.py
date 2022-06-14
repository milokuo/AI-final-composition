import os
from venv import create
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import pickle
import time
import math
from os import system
import sys
import random

csv_source = "dataset"
model_output = "models"

drop_prob = 0.2
learning_rate = 0.001
first_note_dur = 512


class LSTM(nn.Module):
    def __init__(self, tokens, hidden_size=50, layers_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.chars = tokens

        self.layers_size = layers_size
        self.lstm = nn.LSTM(len(self.chars), hidden_size, layers_size,
                            dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hidden_size, len(self.chars))
        self.dropout = nn.Dropout(drop_prob)
        self.lr = learning_rate
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

    
    def forward(self,x, hidden):
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.hidden_size)    
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.layers_size, batch_size, self.hidden_size).zero_(),
                      weight.new(self.layers_size, batch_size, self.hidden_size).zero_())
        return hidden

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s:4.1f}s'

def train(net, data, model_output, batch_size=128, seq_l=25, clip=5):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print_every_n = 500
    save_every_n = 1000
    n_chars = len(net.chars)
    counter = epoch = 0   
    loss_history = []
    start = time.time()    
    while True:
        epoch += 1
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_l):
            counter += 1
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            h = tuple([each.data for each in h])
            net.zero_grad()
            
            output, h = net(inputs, h)
            loss = criterion(output, targets.type(torch.int64).view(batch_size*seq_l))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            if counter % print_every_n == 0:
                print(f"Epoch: {epoch:5} | Step: {counter:6} | Loss: {loss.item():.4f} | Elapsed Time: {time_since(start)}")

            if counter % save_every_n == 0:
                print(f"Epoch: {epoch:5} | Step: {counter:6} | Loss: {loss.item():.4f} | Elapsed Time: {time_since(start)}")
                print(" --- Save checkpoint ---")
                checkpoint = {
                    'hidden_size': net.hidden_size,
                    'layers_size': net.layers_size,
                    'state_dict': net.state_dict(),
                    'tokens': net.chars,
                    'loss_history': loss_history
                }
                torch.save(checkpoint, open(f"{model_output}/epoch_{epoch}.pth", 'wb'))                
        loss_history.append(loss.item())

def predict(net, char, h=None, top_k=None):
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    h = tuple([each.data for each in h])
    out, h = net(inputs, h)

    p = F.softmax(out, dim=1).data

    if top_k is None:
        top_ch = np.arange(len(net.chars))
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
    elif top_k == 1: 
        char = p.argmax().item()
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())  
    return net.int2char[char], h

def sample(net, size, prime='The', top_k=None):
    net.cpu()
    net.eval()
    
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)
    chars.append(char)
    
    
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)
    
    output = createMelody(chars)
    return ' '.join(output)


def createMelody(chars):
    global melody_table
    melody_table = pickle.load(open("melody table/melody_table.pkl", "rb"))
    text = ''.join(chars).split(' ')
    curDur = first_note_dur
    for i in range(len(text)):
        text[i] = text[i] + f"-{curDur}-{curDur}"
        curDur = predict_melody(str(curDur))
        while curDur > 4096:
            curDur = predict_melody(str(curDur))
    return text


def csv2seq(line, filename):
    
    division = int(line[0][-1])
    scale = 1024/division
    df = pd.DataFrame(line, columns=["track", "time", "tipe", "channel", "note", "velocity"])
    df = df.loc[df.tipe.isin(["Note_on_c", "Note_off_c"])]

    df.time = df.time.apply(lambda x: round(int(x)*scale))
    df.track = df.track.apply(int)
    df.note = df.note.apply(lambda x: midi2note[int(x)])
    df.velocity = df.velocity.apply(int)

    df.tipe[(df.tipe=="Note_on_c") & (df.velocity==0)] = "Note_off_c"
    
    df.drop(["channel", "velocity"], axis=1, inplace=True)
    filename = filename.strip(".csv")

    note[filename] = {}
    for track in df.track.unique():
        df_on = df.loc[(df.tipe=="Note_on_c") & (df.track==track)]
        df_off = df.loc[(df.tipe=="Note_off_c") & (df.track==track)]
        df_on.durr = [df_off[(df_off.note==note) & (df_off.time > time)].iloc[0, 1] for time, note in zip(df_on.time.values, df_on.note.values)] - df_on.time
        df_on["next"] = df_on.time.diff().shift(-1).fillna(0)
        learn_melody(df_on.durr.apply(lambda x: str(int(x))), df_on.next.apply(lambda x: str(int(x))))
        df_on.note = df_on.note
        note[filename][track-1] = " ".join(df_on.note.values)
    return note

def seq2csv(seq, fname, channel):
    out = []
    out.append(f"0, 0, Header, 1, {len(seq.keys())}, 1024")
    
    for idx, track in enumerate(seq.keys()):
        out.append(f"{idx+1}, 0, Start_track")
        out.append(f'{idx+1}, 0, Title_t, "{idx+1}"')
        df = pd.DataFrame([item.split("-") for item in seq[track].split()], columns=["note", "durr", "next"])
        df.note = df.note.apply(lambda x: note2midi[x])
        df.durr = df.durr.apply(int)
        df.next = df.next.apply(int)
        df["time_on"] = df.next.cumsum().shift(1).fillna(0).apply(int)
        df["time_off"] = df.time_on + df.durr
        df["track"] = idx+1
        df["tipe_on"], df["tipe_off"] = "Note_on_c", "Note_off_c"
        df["channel"] = channel[idx]
        df["velocity_on"], df["velocity_off"] = 65, 0     
        
        out1 = df[["track", "time_on", "tipe_on", "channel", "note", "velocity_on"]]
        out2 = df[["track", "time_off", "tipe_off", "channel", "note", "velocity_off"]]
        out1.columns = ["track", "time", "tipe", "channel", "note", "velocity"]
        out2.columns = ["track", "time", "tipe", "channel", "note", "velocity"]
        df = out1.append(out2).sort_values("time")
        end_time = df.time.iloc[-1] + 1

        out.extend((df.track.apply(str) + ", " + df.time.apply(str) + ", " + df.tipe + ", " + df.channel.apply(str) + ", " + df.note.apply(str) + ", " + df.velocity.apply(str)).values)
        out.append(f"{idx+1}, {end_time}, End_track")
    out.append(f"0, 0, End_of_file")
    with open(f"{fname}.csv", "w") as f:
        for item in out:
            f.write(f"{item}\n")

def init_converter():
    df = pd.read_csv("conversion/conversion.csv")
    df.note = df.note.apply(lambda x: x.split("/")[0])
    midi2note = dict(zip(df.midi, df.note))
    midi2note[60] = "C4"
    midi2note[69] = "A4"
    note2midi = dict(zip(midi2note.values(), midi2note.keys()))

    pickle.dump(midi2note, open("conversion/midi2note.pkl", "wb"))
    pickle.dump(note2midi, open("conversion/note2midi.pkl", "wb"))            

init_converter()
#Call this function if it's the first time execution
midi2note = pickle.load(open("conversion/midi2note.pkl", "rb"))
note2midi = pickle.load(open("conversion/note2midi.pkl", "rb"))


def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


def get_batches(arr, batch_size, seq_length):
    batch_size_total = batch_size * seq_length
    n_batches = len(arr)//batch_size_total
    
    arr = arr[:n_batches * batch_size_total]
    arr = arr.reshape((batch_size, -1))
    
    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n+seq_length]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y   

melody_table = dict()

def learn_melody(durr, next):
    np.set_printoptions(threshold=sys.maxsize)
    n = np.array(next)
    for i in range(len(n) - 1):
        if n[i] not in melody_table:
            melody_table[n[i]] = {}
        if n[i+1] not in melody_table[n[i]]:
            melody_table[n[i]][n[i+1]] = 0
        melody_table[n[i]][n[i+1]] += 1

def save_melody():
    pickle.dump(melody_table, open("melody table/melody_table.pkl", "wb"))


def predict_melody(prev):
    next_note = np.array([int(k) for k in [*melody_table[prev].keys()]])
    frequency = np.array([*melody_table[prev].values()])
    output = random.choices(next_note, weights = frequency)
    return output[0]



if __name__ == '__main__':
    if not os.path.exists(model_output):
        os.makedirs(model_output)

    note = {}
    for fname in tqdm(os.listdir(csv_source)):
        with open(f"{csv_source}/{fname}", "r") as f:
            line = [line.strip("\n").split(", ") for line in f if len(line.split(", "))==6]
            note = csv2seq(line, fname)
    text = " ".join([track for music in note.values() for track in music.values()])
    text = [note2midi[x] for x in text.split(' ')]
    text = ' '.join([midi2note[x] for x in text if x >= 60])
    save_melody()
    tokens = tuple(set(text))
    int2char = dict(enumerate(tokens))
    char2int = {ch: ii for ii, ch in int2char.items()}
    encoded = np.array([char2int[ch] for ch in text])

    net = LSTM(tokens)
    train(net, encoded, model_output)