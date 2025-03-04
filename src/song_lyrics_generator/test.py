from model import BigramLM
from train import vocab_size,embedding_size,train_data, BATCH_SIZE, BLOCK_SIZE
from utils import get_bigram_mini_batch_samples
import torch

# Initialize Bigram Language Model
model = BigramLM(vocab_size, embedding_size)
x,y = get_bigram_mini_batch_samples(train_data, BATCH_SIZE, BLOCK_SIZE)

# Test model forward pass
logits, loss = model(x, y)
print(logits.shape)
print(loss)

# Test model generate function
print(model.generate_song('I', 200))