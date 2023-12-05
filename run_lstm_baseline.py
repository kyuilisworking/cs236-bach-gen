import torch
import torch.nn as nn
import torch.optim as optim
from codebase.load_data import get_data_loader
from codebase.models.lstm_baseline import LstmBaseline
import codebase.utils as ut

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Model parameters
input_dim = 129
hidden_dim = 512
num_layers = 2

# Create the model
model = LstmBaseline(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
model.to(device)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the DataLoader
dataloader = get_data_loader(
    "data/training_data/32_note_sequences_filtered.pkl", batch_size=128, shuffle=True
)

# Training loop...
num_epochs = 1000
iter_save = 100
step = 0
for epoch in range(num_epochs):
    for i, x in enumerate(dataloader):
        x = x.to(device)
        optimizer.zero_grad()

        # Forward pass
        loss = model.loss(x)
        loss.backward()
        optimizer.step()

        step += 1
        if step % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(dataloader)}], Loss: {loss}"
            )
        if step % iter_save == 0:
            ut.save_model_by_name(model, step)
