#%% 
# import necessary libraries
import time
import json
import torch
import sys
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

sys.path.append('/Users/sydneymarder/Desktop/homework #8')

from model.metric import Metrics
from model.dynamic_model import DenseModel
from data_loader.function_dataset import FunctionDataset

# %%
# Function to load configurations
def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

# Load 4 different configurations for different experiments
configs = {
    "Basic Configuration": load_config("configs/config.json"),
    "Optimal Configuration": load_config("configs/optimal.json"),
    "Overfit Configuration": load_config("configs/overfit.json"),
    "Underfit Configuration": load_config("configs/underfit.json")
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Run the experiments
for config_name, config in configs.items():
    print(f"Running experiment: {config_name}")

    # Initialize TensorBoard writer
    log_dir = f'runs/{config_name.replace(" ", "_")}'
    writer = SummaryWriter(log_dir)

    # Initialize model
    model = DenseModel(
        hidden_layers=config["hidden_layers"],
        neurons_per_layer=config["neurons_per_layer"],
        activation_hidden=config["activation_hidden"],
        activation_output=config["activation_output"],
    ).to(device)

    # Initialize dataset and DataLoader
    train_dataset = FunctionDataset(n_samples=1000, function="linear")
    val_dataset = FunctionDataset(n_samples=200, function="linear")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Log model graph
    print(model)
    dummy_input = torch.ones(1, 1).to(device)
    writer.add_graph(model, dummy_input)

    # Initialize metrics and optimizer
    metrics = Metrics(tolerance=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Training loop
    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        # Training phase
        model.train()
        total_train_loss, total_train_accuracy = 0.0, 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            predictions = model(x)
            loss = metrics.mse_loss(predictions, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_accuracy += metrics.accuracy(predictions, y).item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = total_train_accuracy / len(train_loader)

        # Log training metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_train_accuracy, epoch)

        # Validation phase
        model.eval()
        total_val_loss, total_val_accuracy = 0.0, 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                predictions = model(x)
                loss = metrics.mse_loss(predictions, y)
                total_val_loss += loss.item()
                total_val_accuracy += metrics.accuracy(predictions, y).item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = total_val_accuracy / len(val_loader)

        # Log validation metrics
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', avg_val_accuracy, epoch)
        writer.add_scalar('Time/epoch', time.time() - start_time, epoch)

        # Print log
        metrics.log(epoch, avg_train_loss, avg_val_loss, avg_train_accuracy, avg_val_accuracy)

    # Close TensorBoard writer
    writer.close()
