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
