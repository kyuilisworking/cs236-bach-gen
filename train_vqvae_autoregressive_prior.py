import torch
from torch.utils.data import DataLoader, TensorDataset
from codebase.models.vqvae import AutoregressiveLSTM
from codebase import utils as ut
from codebase.load_data import get_data_loader_input_target


# Assuming AutoregressiveLSTM is defined as in your snippet
def train(model, dataloader, optimizer, epochs):
    model.train()
    step = 0
    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs.to(ut.get_device())
            targets.to(ut.get_device())
            loss = model.calculate_loss(inputs, targets)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:  # Print loss every 100 batches
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
        # Save model periodically
        if step % 10 == 0:
            ut.save_model_by_name(model, step)
            # torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

        step += 1


# Main script
if __name__ == "__main__":
    file_path = "./data/training_data/vqvae_lstm_train_400.pkl"
    dataloader = get_data_loader_input_target(file_path=file_path, batch_size=256)

    config_file_path = "./config.json"  # Replace with your JSON file path
    config = ut.read_model_config(config_file_path)

    hidden_dim = config["lstm_hidden_dim"]  # your hidden dimension
    num_layers = config["lstm_num_layers"]  # your number of LSTM layers
    num_embeddings = config["num_embeddings"]  # your number of embeddings

    model = AutoregressiveLSTM(num_embeddings, hidden_dim, num_layers)
    model.to(ut.get_device())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 1000

    train(model, dataloader, optimizer, epochs)
