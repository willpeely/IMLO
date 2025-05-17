import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedular
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CNN import CNN

# Configuration variables
device = torch.device("cpu")
batch_size = 64
epochs = 10
model_path = "model.pth"

# Defining the training transform
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Adding random horiziontal flip
    transforms.RandomCrop(32, padding=4),   # Adding random crop
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Getting the CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initializing our model
model = CNN().to(device)

# Defining our loss function
loss_function = nn.CrossEntropyLoss()

# Defining our optimizer function
optimizer_function = optim.Adam(model.parameters(), lr=0.001)

# Defining our schedular function
schedular_function = schedular.StepLR(step_size=2, gamma=0.5, optimizer=optimizer_function)

# Used to train the data
def train():
    for epoch in range(epochs): # Iterate through each epoch

        total_loss = 0.0    # Used to keep track of the loss for the epoch

        for images, labels in train_loader: # Iterate through each image and corresponding label

            images, labels = images.to(device), labels.to(device)

            optimizer_function.zero_grad()  # Zero the gradient of the optimizer function

            outputs = model(images) # Get the outputs caluclated 

            loss = loss_function(outputs, labels)   # Caluclate the loss of the epoch

            loss.backward() #Backpropagate 

            optimizer_function.step()   # Update the weights 

            total_loss += loss.item()   # Add to the total loss

        print(f"Epoch - {epoch+1}/{epochs},  Loss - {total_loss/len(train_loader):.6f}")

        schedular_function.step()   # Step the schedular function

# Train the neural network
train() 

# Save the neural network
torch.save(model.state_dict(), model_path)  
