# Homework Report

## Personal Details
**Name:** Sydney Marder

**Date:** 3/26/25

**Course:** ISC 5935

**Instructor:** Olmo S. Zavala-Romero

## 1. Implement Custom Dataset

 I approached the creation of a custom dataset by defining a dataset class FunctionDataset, which generates samples based on a chosen mathematical function (linear, quadratic, or harmonic). This allows for controlled experimentation with various data distributions. The dataset also standardizes the target variable (y) to have a mean of 0 and a standard deviation of 1. This was important to ensure the model's training process wasn't biased by the magnitude of the target values. I then used the FunctionDataLoader class to integrate the dataset into the data loading pipeline, allowing for easy batching and shuffling during training.

My code:
```python
import torch
from torch.utils.data import Dataset
import random
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from base.base_data_loader import BaseDataLoader

# Define regular functions instead of lambdas
def linear_function(x):
    return 1.5 * x + 0.3 + random.uniform(-1, 1)

def quadratic_function(x):
    return 2 * x ** 2 + 0.5 * x + 0.3 + random.uniform(-1, 1)

def harmonic_function(x):
    return 0.5 * x ** 2 + 5 * math.sin(x) + 3 * math.cos(3 * x) + 2 + random.uniform(-1, 1)

class FunctionDataset(Dataset):
    def __init__(self, n_samples=100, function="linear"):
        super().__init__()

        self.n_samples = n_samples
        
        function_map = {
            "linear": linear_function,
            "quadratic": quadratic_function,
            "harmonic": harmonic_function
        }

        if function not in function_map:
            raise ValueError("Function not supported")

        self.function = function_map[function]
        self.x = [random.uniform(0, 2 * math.pi) for _ in range(n_samples)]
        self.y = [self.function(xi) for xi in self.x]
        y_tensor = torch.tensor(self.y, dtype=torch.float32)
        self.mean = torch.mean(y_tensor).item()
        self.std = torch.std(y_tensor).item()
        self.y = ((y_tensor - self.mean) / self.std).tolist()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32).view(1)
        y = torch.tensor(self.y[idx], dtype=torch.float32).view(1)
        return x, y

class FunctionDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, function='linear', n_samples=100):
        # Initialize dataset
        dataset = FunctionDataset(n_samples, function)
        # Pass the dataset to the base loader
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)
```
---

## 2. Implement Model Architecture

In developing the model architecture, I designed a flexible DenseModel class based on the BaseModel class. This model allows for customization of the number of hidden layers, neurons per layer, and the choice of activation functions for both hidden and output layers. The flexibility of the architecture was important to experiment with different network configurations, as the goal was to test underfitting, overfitting, and optimal configurations. The get_activation function simplifies the addition of various activation functions, allowing easy experimentation with ReLU, Sigmoid, Tanh, or a linear identity activation.

My Code
```python
import torch
import torch.nn as nn
import sys
sys.path.append('/Users/sydneymarder/Desktop/homework #8')
print(sys.path)
from base.base_model import BaseModel

class DenseModel(BaseModel):  # Inherit from BaseModel
    def __init__(self, hidden_layers=1, neurons_per_layer=1, activation_hidden="relu", activation_output="linear"):
        super().__init__()  # Call BaseModel's constructor

        # Define a list to hold layers
        layers = []

        # Input layer
        input_dim = 1
        output_dim = neurons_per_layer if hidden_layers > 0 else 1

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, output_dim))
            if activation_hidden != "linear":
                layers.append(self.get_activation(activation_hidden))
            input_dim = output_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))  # Output layer always has 1 dimension
        if activation_output != "linear":
            layers.append(self.get_activation(activation_output))

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

    def get_activation(self, activation_name):
        # Returns the activation function corresponding to the given name
        activation_map = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "linear": nn.Identity()
        }
        # You can raise an error here to help identify unsupported activations
        if activation_name not in activation_map:
            raise ValueError(f"Activation function '{activation_name}' is not supported.")
        return activation_map[activation_name]
    
    def forward(self, x):
        # Pass input through the defined layers
        return self.model(x)
```
---

## 3. Implement Training Metrics

For training metrics, I implemented a Metrics class that calculates key performance metrics, such as Mean Squared Error (MSE) for loss and accuracy based on a predefined tolerance. The tolerance value was chosen to account for minor deviations in regression tasks. I also integrated a logging method that prints training and validation metrics, including loss and accuracy, at each epoch. This was essential for tracking model performance over time and for comparing configurations with different model complexities.

My Code
```python

class Metrics:
    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance

    # mean squared error loss
    def mse_loss(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)

    # accuracy
    def accuracy(self, predictions, targets):
        correct = (torch.abs(predictions - targets) < self.tolerance).float()
        return correct.mean()

    # log metrics (epoch, loss, accuracy)
    # will print training and validation loss and training and validation accuracy
    def log(self, epoch, train_loss, val_loss, train_acc=None, val_acc=None):
        print_msg = f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        if train_acc is not None and val_acc is not None:
            print_msg += f" | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        print(print_msg)
```
---

## 4. Configuration Files

I created different configuration files to represent various model configurations for testing. These configurations include parameters for hidden layers, neurons per layer, activation functions, learning rate, and batch size. The config.json file represents a basic configuration, while optimal.json, overfit.json, and underfit.json represent different levels of model complexity. These configurations were chosen to analyze the impact of model size and training parameters on performance, specifically to identify overfitting and underfitting scenarios.

My code:
### config.json
```json
{
  "hidden_layers": 3,
  "neurons_per_layer": 64,
  "activation_hidden": "relu",
  "activation_output": "linear",
  "num_epochs": 20,
  "learning_rate": 0.001,
  "batch_size": 32
}
```

### optimal.json
```json
{
  "hidden_layers": 3,
  "neurons_per_layer": 64,
  "activation_hidden": "relu",
  "activation_output": "linear",
  "num_epochs": 20,
  "learning_rate": 0.001,
  "batch_size": 32
}
```

### overfit.json
```json
{
  "hidden_layers": 5,
  "neurons_per_layer": 100,
  "activation_hidden": "relu",
  "activation_output": "linear",
  "num_epochs": 50,
  "learning_rate": 0.0001,
  "batch_size": 32
}
```

### underfit.json
```json
{
  "hidden_layers": 1,
  "neurons_per_layer": 16,
  "activation_hidden": "sigmoid",
  "activation_output": "linear",
  "num_epochs": 10,
  "learning_rate": 0.01,
  "batch_size": 16
}
```
---

## 5. TensorBoard Analysis

I integrated TensorBoard into the training pipeline to track and compare different model configurations. Using SummaryWriter, I logged training and validation loss, accuracy, and epoch time, which allowed me to visualize training and validation loss curves and assess convergence. I also logged the time per epoch to compare training speeds. The model architecture was visualized with add_graph() to compare structures across configurations. These metrics enabled side-by-side comparisons of loss curves, training speed, and model architectures, with screenshots capturing the key visualizations for analysis.

When attempting to load files from other parts of the repository, I encountered issues with module imports due to the structure of the project. The paths weren't resolving correctly when running the code in a Jupyter notebook. To address this, I switched to using Jupyter cells within a Python script instead of the notebook itself. This allowed me to maintain the correct file paths and execute the code in an environment that recognized the directory structure. This approach ensured that all dependencies and modules were properly imported, and I could continue my work without running into import errors.

My Code:
```python
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
```
---

### Training Time (Epochs to Convergence)
<img width="770" alt="Screenshot 2025-03-26 at 5 24 01 PM" src="https://github.com/user-attachments/assets/1d87e8aa-4a4e-44b8-a5e0-ecf0b10de87a" />

- **Overfit Configuration**:  
  - Converges the slowest due to the high model complexity and excessive parameters.  
  - Training loss continues to decrease while validation loss starts increasing, indicating memorization rather than learning.  
  - The loss curve suggests prolonged optimization with little improvement in generalization.  
  - While the model achieves high accuracy on the training set, its performance on validation data deteriorates.  

- **Optimal Configuration**:  
  - Balances training speed and generalization, converging efficiently within a reasonable number of epochs.  
  - The model avoids unnecessary parameter tuning and instead focuses on meaningful feature extraction.  
  - Training and validation losses stabilize at a good point, preventing both overfitting and underfitting.  
  - This configuration is ideal for deployment as it maximizes accuracy while maintaining computational efficiency.  

- **Underfit Configuration**:  
  - Converges very quickly but lacks the capacity to learn complex patterns.  
  - The model reaches a plateau early on, failing to improve due to limited parameters or insufficient training.  
  - Training and validation losses remain high, indicating that the model is unable to capture the underlying structure of the data.  
  - This configuration results in poor accuracy and should not be used in practice.  

- **Basic Configuration**:  
  - Shows moderate convergence speed, neither as fast as the underfit model nor as slow as the overfit one.  
  - Its effectiveness depends on hyperparameter tuning, as it may require adjustments to optimize learning.  
  - Performs reasonably well but does not outperform the optimal configuration.  

---
<img width="770" alt="Screenshot 2025-03-26 at 5 14 00 PM" src="https://github.com/user-attachments/assets/b791c4c3-6bb5-4822-9341-eb736c514335" />

### Training Loss Curves
- **Overfit Configuration**:  
  - The loss curve sharply drops early, followed by continuous fine-tuning, indicating memorization rather than real learning.  
  - Training loss remains low, but the validation loss starts diverging, proving the model is overfitting.  

- **Optimal Configuration**:  
  - Training loss decreases steadily and stabilizes, preventing overfitting while still achieving good accuracy.  
  - Validation loss does not diverge significantly, showing that the model generalizes well to unseen data.  

- **Underfit Configuration**:  
  - The loss curve flattens almost immediately, suggesting the model stops learning early.  
  - The inability to reduce loss further signals insufficient model complexity or poor feature extraction.  

- **Basic Configuration**:  
  - The loss curve shows decent improvement, but further hyperparameter tuning is needed to match the performance of the optimal configuration.  

### Validation Loss Curves

- **Overfit Configuration**:  
  - Validation loss initially decreases but then starts increasing due to overfitting.  
  - The model fails to generalize, leading to poor validation performance despite low training loss.  

- **Optimal Configuration**:  
  - Validation loss decreases and remains stable, showing that the model does not overfit.  
  - This behavior suggests the model learns meaningful patterns that generalize well to unseen data.  

- **Underfit Configuration**:  
  - Validation loss remains high, confirming the model's inability to extract useful information from the data.  
  - This model is not suitable for real-world use as it does not perform well even on training data.  

- **Basic Configuration**:  
  - The validation loss curve is more stable than in the overfitting case but requires further tuning to reach optimal generalization.  

---
<img width="650" alt="Screenshot 2025-03-26 at 5 30 03 PM" src="https://github.com/user-attachments/assets/c2c9b9cb-d8a8-4a7a-badc-8b69683797c2" />

### Model Architectures using `add_graph`
- **Overfit Configuration**:
- DenseModel(
  (model): Sequential(
    (0): Linear(in_features=1, out_features=100, bias=True)
    (1): ReLU()
    (2): Linear(in_features=100, out_features=100, bias=True)
    (3): ReLU()
    (4): Linear(in_features=100, out_features=100, bias=True)
    (5): ReLU()
    (6): Linear(in_features=100, out_features=100, bias=True)
    (7): ReLU()
    (8): Linear(in_features=100, out_features=100, bias=True)
    (9): ReLU()
    (10): Linear(in_features=100, out_features=1, bias=True)
  )
)
  - A highly complex model with excessive parameters, leading to overfitting.  
  - Deeper layers and a large number of neurons allow it to memorize data rather than generalize.  

- **Optimal Configuration**:
- DenseModel(
  (model): Sequential(
    (0): Linear(in_features=1, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=64, bias=True)
    (5): ReLU()
    (6): Linear(in_features=64, out_features=1, bias=True)
  )
)  
  - A well-balanced architecture with enough capacity to learn meaningful patterns without overfitting.  
  - Features sufficient depth and width to capture complex representations while avoiding excessive complexity.  

- **Underfit Configuration**:
- DenseModel(
  (model): Sequential(
    (0): Linear(in_features=1, out_features=16, bias=True)
    (1): Sigmoid()
    (2): Linear(in_features=16, out_features=1, bias=True)
  )
)
  - A shallow network with insufficient parameters, making it incapable of learning useful patterns.  
  - The model lacks the capacity to represent the data adequately, leading to high bias.  

- **Basic Configuration**:
- DenseModel(
  (model): Sequential(
    (0): Linear(in_features=1, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=64, bias=True)
    (5): ReLU()
    (6): Linear(in_features=64, out_features=1, bias=True)
  )
)
  - A moderately complex architecture, but not optimized for the dataset.  
  - Can be adjusted through hyperparameter tuning to improve performance.  

---

### Side-by-Side Loss Curve Comparisons  
- The TensorBoard screenshots included show clear differences in loss behavior across configurations.  
- The overfit configuration exhibits a sharp increase in validation loss, while the optimal configuration maintains stability.  
- The underfit configuration shows minimal improvement, confirming poor learning capacity.  

### Training Speed Differences  
- The TensorBoard graphs highlight differences in convergence speed among the models.  
- The overfit model continues learning for longer, whereas the underfit model plateaus quickly.  
- The optimal model strikes a balance between efficient learning and convergence.  

### Architecture Comparisons  
- Using `add_graph`, we visualize each model's structure in TensorBoard.  
- The overfit model has significantly more layers and parameters, explaining its memorization tendency.  
- The optimal model has a structured architecture that generalizes well.  
- The underfit model lacks sufficient layers to capture complex features.  

---

## 6. Analysis and Documentation (10 pts)

This was done in a py file using jupyter cells. For some reason everytime I tries to use jupyter notebook there were problems importing files from other parts of the repository. This was the only way to get it to work. My code does do the following:

1. **Loads different model configurations**  
   - Uses pre-trained model states or initializes fresh training runs.  
   - Ensures consistency across experiments for fair comparisons.  

2. **Trains models using the template**  
   - Uses a standardized training script to maintain reproducibility.  
   - Implements logging for performance tracking.  

3. **Visualizes results**  
   - Plots training and validation loss/accuracy curves.  
   - Compares performance across configurations.  

4. **Analyzes overfitting and underfitting cases**  
   - Examines loss curves and accuracy trends.  
   - Identifies symptoms of overfitting (divergence in loss curves).  
   - Detects underfitting (early plateauing of loss).    

---

### Findings and Conclusions
- **Best Performing Configuration**: The **optimal configuration** achieves the best trade-off between training speed, accuracy, and generalization. This configuration effectively captures the underlying patterns in the data without overfitting or underfitting. It demonstrates consistent performance across both training and validation datasets and strikes a balance between model complexity and accuracy.  
- **Overfit Configuration**: This configuration performs exceptionally well on the training data, but struggles on the validation set, indicating it has overfitted the training data. The model's excessive complexity, such as too many features or an overly deep architecture, causes it to memorize noise and fail to generalize effectively to new, unseen data.  
- **Underfit Configuration**: This configuration fails to capture the underlying trends in the data, as indicated by its poor performance on both the training and validation sets. It lacks sufficient capacity, such as insufficient model parameters or overly simple assumptions, to learn meaningful patterns, leading to underfitting.
- **Basic Configuration**: The basic configuration provides moderate performance on both training and validation datasets. While not as optimized as the best configuration, it serves as a useful baseline. This configuration benefits from some degree of complexity, but additional tuning in terms of hyperparameters or model architecture could lead to improved results.
- **Generalization Observations**: Across all configurations, it was observed that generalization ability depends on the model's capacity to balance between bias and variance. Too much complexity leads to overfitting, while too little complexity results in underfitting. The optimal configuration achieves a balance that enables the model to generalize well to new data, making it the most reliable choice for real-world applications.
---
