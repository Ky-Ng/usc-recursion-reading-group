#  %%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torch import nn as nn

import torch

# %%

# %%
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image to tensor [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean, std
])

train_dataset = datasets.MNIST(
    root='./data',           # Where to save/load data
    train=True,              # We want training data
    download=True,           # Download if not already present
    transform=transform      # Apply our transformations
)

# Download and load the test dataset
test_dataset = datasets.MNIST(
    root='./data',
    train=False,             # We want test data
    download=True,
    transform=transform
)
# %%
train_dataset[0][0].shape
# %%
def plot_mnist_grid(dataset, grid_size=5, figsize=(10, 10)):
    """
    Plot MNIST digits in a grid.
    
    Parameters:
    - dataset: The MNIST dataset
    - grid_size: Size of the square grid (grid_size x grid_size images)
    - figsize: Figure size for the plot
    
    First Principle: We're creating a spatial arrangement to see multiple
    examples at once. This helps us understand the variability in the data.
    """
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    fig.suptitle('MNIST Digit Samples', fontsize=16, fontweight='bold')
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            image, label = dataset[idx]
            
            # Remove the channel dimension and convert to numpy
            # Shape: (1, 28, 28) -> (28, 28)
            image_np = image.squeeze().numpy()
            
            axes[i, j].imshow(image_np, cmap='gray')
            axes[i, j].set_title(f'Label: {label}', fontsize=10)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    return fig

# Create and display the grid
fig1 = plot_mnist_grid(train_dataset, grid_size=5, figsize=(12, 12))

# %% 
# Create the Neural Network
sample_data_img, sample_data_label = train_dataset[0]
# sample_data_img = train_dataset[0][0]
# sample_data_label = train_dataset[0][1]

channels, height, width = sample_data_img.shape
print(f"Sample Image for {sample_data_label} has {channels} channel, {height} height, {width} width")

layer_1 = nn.Linear(in_features=height*width, out_features=512)
layer_2 = nn.Linear(in_features=512, out_features=764)
layer_3 = nn.Linear(in_features=764, out_features=764)
layer_4 = nn.Linear(in_features=764, out_features=10)

model = nn.Sequential(
    nn.Flatten(),
    layer_1,
    nn.ReLU(),
    layer_2,
    nn.ReLU(),
    layer_3,
    nn.ReLU(),
    layer_4,
    nn.ReLU()
)
# %%
model(sample_data_img)

# %%
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,    # Shuffle training data for better learning
)

# %%
model_final = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_final.parameters(), lr=0.001)

# Training loop
EPOCHS = 2000
for epoch in range(EPOCHS):
    total_loss = 0  # ← Initialize here!
    correct = 0      # ← Initialize here!
    total = 0        # ← Initialize here!
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        optimizer.zero_grad()
        output = model_final(data)  # Model handles flattening
        
        # Compute loss
        loss = loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = output.argmax()
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Print progress
        if batch_idx % 500 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, ' + 
                  f'Loss: {loss.item():.4f}, ' +
                  f'Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / (batch_idx + 1)
    accuracy = 100. * correct / total
    print(f'\nEpoch {epoch+1} Summary: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%')
# %%

for img, label in train_dataset:
    output = model(img)
    predicted_label = torch.argmax(output)
    print(f"Predicted Label is {predicted_label}, actual {label}")
    break

# %%