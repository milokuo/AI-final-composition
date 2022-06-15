# AI composition

## Table of Contents
- [Overview](#overview)
- [Prerequisite](#prerequisite)
- [Usage](#usage)
- [Result](#result)

## Overview
The project implemented the AI music generation through LSTM and Markov. The LSTM network was used to predict the pitch of notes, while the Markov model was used to predict the rhythm.

## Prerequisite
- Python == 3.7.9
- Pytorch == 1.10.0
- numpy
- pandas
- pickle
- tqdm
- plac


## Usage
To train the model:
```python train.py```\
Make sure you do have some music sheet in csv form in your dataset folder. You could convert the midi files of your sheet into csv through [MIDICSV](https://www.fourmilab.ch/webtools/midicsv/).

To compose a song:
```python compose.py -i [-model name] -o [-output]```\
The filename extension should not be included.\
The output would be a csv file. You could convert it into midi file by MIDICSV similarly.

## Result
You could find a demo of a song from this AI through the link : [AI composition - LSTM with Markov](https://www.youtube.com/watch?v=JTDrYezXNmk)
