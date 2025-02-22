import torch

def get_bigram_mini_batch_samples(data, batch_size, block_size):
    """
    Function to generate a mini-batch samples from the provided data.

    X has 'batch_size' samples where each sample is of size 'block_size' (input layer size of the model)
    X dim - (batch_size, block_size)

    y is the target to predict where each target is X shifted by one such that we are predicting the next token
    y dim - (batch_size, block_size)
    """
    start_idxs = torch.randint(0, len(data) - block_size, (batch_size,))

    X = torch.stack([data[idx : idx + block_size] for idx in start_idxs])
    y = torch.stack([data[idx + 1 : idx + block_size + 1] for idx in start_idxs])

    return X, y