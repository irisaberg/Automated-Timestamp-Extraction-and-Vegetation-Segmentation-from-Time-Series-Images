# -*- coding: utf-8 -*-
"""
The script:
1. Loads cropped digit images from disk
2. Trains a CNN to classify digits (0–9)
3. Evaluates training accuracy
4. Saves the trained model to disk



How to use:
- Place digit images in class-labeled folders (0–9)
- Update the dataset path if needed
- Run the script to train and save the model

"""



import torch
import torch.nn as nn
from torchvision import datasets, transforms



# Define relevant variables
batch_size = 64
num_classes = 10
learning_rate = 0.0005
num_epochs = 20

# Select GPU if avaliable, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training augmentation: grayscale, resize, sharpness
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.RandomAdjustSharpness(1.5),
    transforms.ToTensor()
])

# Test transform: no augmentation
test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Load dataset using class-folder structure
trainset = datasets.ImageFolder(
    #Directory containing folders (0-9)
    root='All_digits_cut/All_digits', 
    transform=train_transforms
)



train_loader = torch.utils.data.DataLoader(dataset = trainset,
                                           batch_size = batch_size,
                                           shuffle = True)


# Creating a CNN class
class DigitCNN(nn.Module):
#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes):
        super(DigitCNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

model = DigitCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# Training loop
for epoch in range(num_epochs):
# Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    
# Accuracy check
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the train images: {} %'.format( 100 * correct / total))
    
    
# Save trained model
torch.save(model.state_dict(), "digit_cnn.pth")
print("\nTraining complete. Model saved as digit_cnn.pth.")



