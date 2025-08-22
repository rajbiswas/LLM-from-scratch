import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLM(nn.Module):
    """
    Class defining the bi-gram language model for next character prediction.
    An embedding is being learnt for each character based on which the next
    character will be generated
    """

    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)

    def get_batch_loss(self, mini_batch_logits, y):
        """Calculate the cross-entropy loss for the model's predictions.

        Args:
            mini_batch_logits (torch.Tensor): Predicted logits from the model.
                Shape: (batch_size, block_size, emb_size)
            y (torch.Tensor): Target labels/tokens.
                Shape: (batch_size, block_size)

        Returns:
            torch.Tensor: The computed cross-entropy loss.
                Shape: scalar value averaged over the batch

        Notes:
            The function reshapes the input tensors to match PyTorch's CrossEntropyLoss
            expectations:
            - Inputs are reshaped to (batch_size * block_size, num_classes)
            - Targets are reshaped to (batch_size * block_size,)
            where num_classes = emb_size = vocab_size
        """
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(mini_batch_logits.view(-1, mini_batch_logits.shape[-1]), y.view(-1))
        return loss
    
    def forward(self, mini_batch, y=None):
        """Forward pass of the model that generates predictions and optionally computes loss.

        Args:
            mini_batch (torch.Tensor): Input tensor containing token indices.
                Shape: (batch_size, block_size)
            y (torch.Tensor, optional): Target labels for computing loss.
                Shape: (batch_size, block_size). Defaults to None.

        Returns:
            tuple:
                - y_pred (torch.Tensor): Predicted token indices sampled from the output distribution.
                    Shape: (batch_size * block_size,)
                - loss (torch.Tensor or None): Cross-entropy loss if y is provided, None otherwise.
                    Shape: scalar value

        Notes:
            The function performs the following steps:
            1. Embeds the input tokens
            2. Applies softmax to get probability distribution
            3. Samples from the distribution to generate predictions
            4. Computes loss if target labels are provided

        TODO:
            Current implementation only produces predictions based on the last character
            in each block. Need to modify the forward pass to:
            1. Support batch predictions for all individual characters in a block
            2. Make it compatible with mini-batch training by considering the entire
            sequence context
        """
        # Get logits from embedding layer
        mini_batch_logits = self.embed(mini_batch)
        
        # Convert logits to probabilities
        y_prob = F.softmax(mini_batch_logits, dim=-1)  # Shape: (batch_size, block_size, vocab_size)
        y_prob = y_prob.view(-1, y_prob.shape[-1])
        
        # Sample from probability distribution
        y_pred = torch.multinomial(y_prob, num_samples=1).squeeze(-1)  # Shape: (batch_size * block_size,)

        # Compute loss during training, return None during inference
        loss = None
        if y is not None:
            loss = self.get_batch_loss(mini_batch_logits, y)

        return y_pred, loss
    
    def generate_song(self, initial_char, song_length, encode_fn=None, decode_fn=None):
        """Generate a song sequence starting from an initial character.

        Args:
            initial_char (str): The first character to start the song generation.
            song_length (int): The desired length of the generated song in characters.
            encode_fn (callable): Function to encode characters to indices.
            decode_fn (callable): Function to decode indices back to characters.

        Returns:
            str: The generated song text.

        Notes:
            The function generates text character by character using the trained model.
            It maintains a context of the last character to predict the next one,
            building the song sequence iteratively.
        """
        if encode_fn is None or decode_fn is None:
            raise ValueError("encode_fn and decode_fn must be provided")
            
        song = []
        
        # Get the device the model is on
        device = next(self.parameters()).device
        
        # Encode the initial character to its corresponding index
        previous_char = encode_fn(initial_char)
        
        # Generate characters one at a time
        for _ in range(song_length):
            # Create input tensor with zeros, shape: (1, 8)
            # Only the last position will contain the previous character
            input = torch.zeros(1, 8, dtype=torch.int64, device=device)
            input[0, -1] = previous_char
            
            # Get model prediction and ignore the loss
            y_pred, loss = self(input, None)
            
            # Use the last predicted character as input for next iteration
            previous_char = y_pred[-1]
            
            # Add the predicted character to our song sequence
            song.append(previous_char.item())
        
        # Convert the list of character indices back to a string
        return decode_fn(song)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledSelfAttention(nn.Module):
    def __init__(self, emb_size, head_size, dropout):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_size = head_size
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.to(self.device)
        B, T, C = x.shape
        k = self.key(x)           # B, T, head_size
        q = self.query(x)       # B, T, head_size

        # Compute affinity matrix containing affinity of each token wrt to other tokens (T, T)
        # which the dot product between query and key for each token wrt other token (overall euivalent to matrix multiplication)
        affinity_matrix = q @ k.transpose(-2, -1)     # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
        # Scale the affinity matrix for stability
        affinity_matrix *= (self.head_size ** -0.5)
        
        # Apply lower triangle mask math trick to get running sum of affinity matrix
        lower_tri_mask = torch.tril(torch.ones(T, T)).to(self.device)    # (T, T)
        affinity_matrix = affinity_matrix.masked_fill(lower_tri_mask == 0, float('-inf'))   # (B, T, T)

        k_q_affinity = F.softmax(affinity_matrix, dim=-1)      # (B, T, T)
        k_q_affinity = self.dropout(k_q_affinity)

        v = self.value(x)       # B, T, head_size

        attention = k_q_affinity @ v        # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)

        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, head_size, dropout):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.heads = nn.ModuleList([ScaledSelfAttention(emb_size, head_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):   # (B, T, C)
        x = x.to(self.device)
        out = torch.cat([h(x) for h in self.heads], dim=-1)     # (B, T, head_size * num_heads)
        out = self.projection(out)      # (B, T, C)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, emb_size, dropout):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = x.to(self.device)
        return self.net(x)          # B, T, C -> B, T, C * 4 -> B, T, C


class AttentionBlock(nn.Module):
    def __init__(self, emb_size, num_heads, head_size, dropout):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_norm1 = nn.LayerNorm(emb_size)
        self.attention = MultiHeadAttention(emb_size, num_heads, head_size, dropout)
        self.layer_norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = FeedForward(emb_size, dropout)
    
    def forward(self, x):
        x = x.to(self.device)
        x = x + self.attention(self.layer_norm1(x))         # B, T, C
        x = self.layer_norm2(x)                             # B, T, C
        x = x + self.feed_forward(x)                        # B, T, C
        return x                                            # B, T, C
        



class GPT(nn.Module):
    def __init__(self, block_size, vocab_size, emb_size, n_blocks, num_heads, head_size, dropout):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, emb_size)
        self.pos_embed = nn.Embedding(block_size, emb_size)
        self.blocks = nn.Sequential(*[AttentionBlock(emb_size, num_heads, head_size, dropout) for _ in range(n_blocks)])
        self.layer_norm_final = nn.LayerNorm(emb_size)
        self.linear_final = nn.Linear(emb_size, vocab_size)
    
    def get_loss(self, mini_batch_logits, y):
        """Calculate the cross-entropy loss for the model's predictions.

        Args:
            mini_batch_logits (torch.Tensor): Predicted logits from the model.
                Shape: (batch_size, block_size, emb_size)
            y (torch.Tensor): Target labels/tokens.
                Shape: (batch_size, block_size)

        Returns:
            torch.Tensor: The computed cross-entropy loss.
                Shape: scalar value averaged over the batch

        Notes:
            The function reshapes the input tensors to match PyTorch's CrossEntropyLoss
            expectations:
            - Inputs are reshaped to (batch_size * block_size, num_classes)
            - Targets are reshaped to (batch_size * block_size,)
            where num_classes = emb_size = vocab_size
        """
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(mini_batch_logits.view(-1, mini_batch_logits.shape[-1]), y.view(-1))
        return loss
    
    def forward(self, mini_batch, y = None):
        mini_batch = mini_batch.to(self.device)
        B, T = mini_batch.shape

        if y is not None:
            y = y.to(self.device)

        # Get logits from embedding layer
        token_embd = self.token_embed(mini_batch)           # B, T, C
        token_positions = torch.arange(T, device=self.device)    # T
        pos_embd = self.pos_embed(token_positions)          # B, T, C

        x = token_embd + pos_embd                           # B, T, C

        # Add multi-head attention below
        x = self.blocks(x)                                  # B, T, C
        x = self.layer_norm_final(x)                        # B, T, C

        # Pass through final linear layer to get logits
        mini_batch_logits = self.linear_final(x)                       # B, T, vocab_size

        # Convert logits to probabilities
        y_prob = F.softmax(mini_batch_logits, dim=-1)  # Shape: (batch_size, block_size, vocab_size)
        y_prob = y_prob.view(-1, y_prob.shape[-1])                      # Reshaping to (batch_size * block_size, vocab_size)

        # Sample from probability distribution
        y_pred = torch.multinomial(y_prob, num_samples=1).squeeze(-1)  # Shape: (batch_size * block_size,)

        loss = None
        if y is not None:
            loss = self.get_loss(mini_batch_logits, y)

        return y_pred, loss
    
    def generate_song(self, initial_char, song_length):
        """Generate a song sequence starting from an initial character.

        Args:
            initial_char (str): The first character to start the song generation.
            song_length (int): The desired length of the generated song in characters.

        Returns:
            str: The generated song text.

        Notes:
            The function generates text character by character using the trained model.
            It maintains a context of the last character to predict the next one,
            building the song sequence iteratively.
        """

        # TODO: Add logic to pass initial string instead of one character

        # Encode the initial character to its corresponding index
        song = encode(initial_char).to(self.device)

        # Generate characters one at a time
        for _ in range(song_length):
            # Create input tensor with zeros, shape: (1, 8)
            # Only the last position will contain the previous character
            # input = torch.zeros(1, 8, dtype=torch.int64)
            song_window_encoded = song[-self.block_size:]
            input = song_window_encoded.reshape(1, -1)

            # Get model prediction and ignore the loss
            y_pred, loss = self(input, None)

            # Use the last predicted character as input for next iteration
            next_char_encoded = y_pred[-1]
            song = torch.cat((song, next_char_encoded.unsqueeze(0)))

        # Convert the list of character indices back to a string
        return decode(song)