import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLM(nn.Module):
    """
    Class defining the bi-gram language model for next character prediction.
    An embedding is being learnt for each character based on which the next
    character will be generated
    """

    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
    
    def get_loss(logits, y):
        """
        logits dim:
        y dim:
        """
    
    def forward(self, mini_batch):
        """
        Mini batch dim: (batch_size, block_size)
        logits dim: (batch_size, block_size, emb_size)
        """
        logits = self.embed(mini_batch)