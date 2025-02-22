import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# Loading the eminem song lyric dataset available here: https://www.kaggle.com/datasets/aditya2803/eminem-lyrics/data
# Run these commands to download and unzip the dataset:
#   1. curl -L -o ~/data/eminem-lyrics.zip https://www.kaggle.com/api/v1/datasets/download/aditya2803/eminem-lyrics
#   2. unzip ~/data/eminem-lyrics.zip

# Read the data into a DataFrame
df = pd.read_csv('data/Eminem_Lyrics.csv', sep='\t', comment='#', encoding = "ISO-8859-1")

# Preprocessing

# 1. Remove '\x82', '\x85', '\x91', '\x92', '\x93', '\x94', '\x96', '\x97' from lyrics text
df.Lyrics = df.Lyrics.str.replace('\x82', '')
df.Lyrics = df.Lyrics.str.replace('\x85', '')
df.Lyrics = df.Lyrics.str.replace('\x91', '')
df.Lyrics = df.Lyrics.str.replace('\x92', '')
df.Lyrics = df.Lyrics.str.replace('\x93', '')
df.Lyrics = df.Lyrics.str.replace('\x94', '')
df.Lyrics = df.Lyrics.str.replace('\x96', '')
df.Lyrics = df.Lyrics.str.replace('\x97', '')

# 2. Creating the character set to represent characters as integers based on their index
#   in the sorted character set
all_text = '\n'.join(df.Lyrics)
chars = sorted(list(set(all_text)))

encoder = {}
decoder = {}
for i, char in enumerate(chars):
    encoder[char] = i
    decoder[i] = char

# Lambda function that encodes a string to a list of ints
encode = lambda s: [encoder.get(c, '-1') for c in s]

# Lambda function that decodes a list of ints back to a string
decode = lambda l: ''.join([decoder.get(i, '') for i in l])

# 3. Encoding all the lyrics
all_text_encoded = torch.tensor(encode(all_text))

# 4. Creating train/ validation split
split_index = int(len(all_text_encoded) * 0.9)
train_data = all_text_encoded[:split_index]
val_data = all_text_encoded[split_index:]
