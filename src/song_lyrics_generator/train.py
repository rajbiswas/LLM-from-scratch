import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from matplotlib import pyplot as plt

from models import BigramLM
from utils import get_bigram_mini_batch_samples

# Loading the eminem song lyric dataset available here: https://www.kaggle.com/datasets/aditya2803/eminem-lyrics/data
# Run these commands to download and unzip the dataset:
#   1. curl -L -o ~/data/eminem-lyrics.zip https://www.kaggle.com/api/v1/datasets/download/aditya2803/eminem-lyrics
#   2. unzip ~/data/eminem-lyrics.zip

# Setting global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
BLOCK_SIZE = 8
NUM_EPOCHS = 10000
EVAL_INTERVAL = 500
EVAL_ITERS = 200

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

# 5. Creating mini-batch samples
X_train, y_train = get_bigram_mini_batch_samples(train_data, BATCH_SIZE, BLOCK_SIZE)

# 6. Initializing the model
vocab_size= len(chars)
embedding_size = len(chars)
model = BigramLM(vocab_size, embedding_size)

# 7. Train the model
def get_loss(model, eval_iters):
    """ Compute average loss over multple batches of train and validation data
    Args:
        model (BigramLM): Model for which the loss has to be computed
        eval_iters (int): Number of batches to draw from the data
    """
    # Setting model to eval mode
    model.eval()

    train_losses = []
    val_losses = []

    # Generate samples from train_data eval_iters times and average the loss
    for i in range(eval_iters):
        train_x, train_y = get_bigram_mini_batch_samples(train_data, batch_size, block_size)
        _, train_loss = model(train_x, train_y)

        val_x, val_y = get_bigram_mini_batch_samples(val_data, batch_size, block_size)
        _, val_loss = model(val_x, val_y)
    
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
    
    # Calculate average loss
    train_loss = sum(train_losses) / len(train_losses)
    val_loss = sum(val_losses) / len(val_losses)

    # Setting model back to train mode
    model.train()

    return train_loss, val_loss

def train_model(model, optimizer):
    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        x, y = get_bigram_mini_batch_samples(train_data, batch_size, block_size)
        y_pred, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % EVAL_INTERVAL == 0:
            train_loss, val_loss = get_loss(model, EVAL_ITERS)
            print(f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss}")
            train_losses.append(loss.item())
            val_losses.append(val_loss)

    plt.plot(train_losses)
    plt.plot(val_losses)