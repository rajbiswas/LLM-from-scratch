# Training loop
from matplotlib import pyplot as plt

# Function to get validation loss during training
def get_loss(model, train_data, val_data, eval_iters):
  """
  val_data: vector of token indices

  """
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

  model.train()

  return train_loss, val_loss

def train_model(model, optimizer):
  model.train()
  train_losses = []
  val_losses = []

  for epoch in tqdm(range(num_epochs)):
      x, y = get_bigram_mini_batch_samples(train_data, batch_size, block_size)
      y_pred, loss = model(x, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if epoch % eval_interval == 0:
          train_loss, val_loss = get_loss(model, train_data, val_data, eval_iters)
          print(f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss}")
          train_losses.append(loss.item())
          val_losses.append(val_loss)

  plt.plot(train_losses)
  plt.plot(val_losses)

batch_size = 64
block_size = 256
num_epochs = 5000
eval_interval = 100
eval_iters = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT(
    block_size = 256,
    vocab_size = len(chars),
    emb_size = 384,
    n_blocks = 6,
    num_heads = 6,
    head_size = 64,
    dropout = 0.2
)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

import time
start_time = time.time()
train_model(model, optimizer)
end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")