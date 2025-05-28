from tqdm import tqdm
from matplotlib import pyplot as plt
from song_lyrics_generator.utils import get_bigram_mini_batch_samples


class Trainer:
    def __init__(self, model, optimizer, device, num_epochs, eval_interval, eval_iterations, train_data, val_data, batch_size, block_size):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.eval_interval = eval_interval
        self.eval_iterations = eval_iterations
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.block_size = block_size

    def get_loss(self):
        """ Compute average loss over multple batches of train and validation data
        Args:
            model (BigramLM): Model for which the loss has to be computed
            eval_iters (int): Number of batches to draw from the data
        """
        # Setting model to eval mode
        self.model.eval()

        train_losses = []
        val_losses = []

        # Generate samples from train_data eval_iters times and average the loss
        for i in range(self.eval_iterations):
            train_x, train_y = get_bigram_mini_batch_samples(self.train_data, self.batch_size, self.block_size)
            _, train_loss = self.model(train_x.to(self.device), train_y.to(self.device))

            val_x, val_y = get_bigram_mini_batch_samples(self.val_data, self.batch_size, self.block_size)
            _, val_loss = self.model(val_x.to(self.device), val_y.to(self.device))

            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

        # Calculate average loss
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)

        # Setting model back to train mode
        self.model.train()

        return train_loss, val_loss

    def train_model(self):
        train_losses = []
        val_losses = []

        for epoch in tqdm(range(self.num_epochs)):
            x, y = get_bigram_mini_batch_samples(self.train_data, self.batch_size, self.block_size)
            y_pred, loss = self.model(x.to(self.device), y.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % self.eval_interval == 0:
                train_loss, val_loss = self.get_loss()
                print(f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss}")
                train_losses.append(loss.item())
                val_losses.append(val_loss)

        plt.plot(train_losses)
        plt.plot(val_losses)