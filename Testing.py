# -*- coding: utf-8 -*-
"""

This script: 
1. Loads images containing timestamps 
2. Extracts the timestamp strip from each image 
3. Crops individual digits from the strip 
4. Uses a trained CNN to classify each digit 
5. Reconstructs the timestamp 
6. Renames the image files using the extracted date and time


How to use:
1. Make sure that the trained model (digit_cnn.pth) is located in the same folder as this script.
2. Set the directory path containing the images with timestamps.
3. Run the script
4. Images will be renamed with extracted date and time.




Expected timestamp format in image:
YYYY/MM/DD HH:MM:SS

"""

import imageio.v2 as imageio
import cv2 
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os 

#Number of digit classes 0-9
num_classes = 10

#Timestamp strip extraction
def getStrip(img):
    
    #Crop the region where the timestamp is located
    strip = img[-20:, 575:-401, :]
    
    #Convert to grayscale
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    
    #Threshold to create binary image
    binary = (gray > 230).astype(np.uint8)
    return binary


#Digit cropping
def cropDigit(strip):
    
    #Total number of characters in the timestamp string
    nbrChars = 19
    
    #Convert binary image to PIL format
    strip_img = Image.fromarray(strip * 255)  
    
    #Calculating width and height for each character region
    digit_w = strip_img.width / nbrChars
    digit_h = strip_img.height
    
    #Position of digits
    digit_positions = [0,1,2,3,5,6,8,9,11,12,14,15,17,18]  # skip / and :
    
    #Crop out each digit and save each digit in a list
    digits = []
    for i in digit_positions:
        x1 = int(i * digit_w)
        y1 = 0
        x2 = int(x1 + digit_w)
        y2 = int(digit_h)
        digits.append(strip_img.crop((x1, y1, x2, y2)))
    return digits
    

#CNN model definition
class DigitCNN(nn.Module):

    def __init__(self, num_classes):
        super(DigitCNN, self).__init__()
        
        #Convolutional layers
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        #Fully connected layers
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

# Load trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitCNN(num_classes).to(device)
model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
model.eval()



#Digit preprocessing
digit_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


#Digit prediction
def predict_digit(img):
    with torch.no_grad():
        x = digit_transform(img).unsqueeze(0).to(device)
        out = model(x)
        return out.argmax(1).item()


#Extracts and predicts the full timestamp from an image
def predict_timestamp(image_path):
    img = imageio.imread(image_path)
    strip = getStrip(img)
    digit_imgs = cropDigit(strip)

    predictions = [predict_digit(d) for d in digit_imgs]
    return predictions



if __name__ == "__main__":

    #The directory containing images with embedded timestamps
    directory = 'YOUR PATH'  
    
    #Loop through all files in the directory 
    for entry in os.scandir(directory):  
        if entry.is_file():
            
            #Predict timestamp digits
            result = predict_timestamp(entry.path)
            
            #print("Predicted timestamp digits:", result)
            
            #Convert digit list to string
            timestamp_str = "".join(str(d) for d in result)
            
            #Split into date and time
            date = timestamp_str[:8] 
            time = timestamp_str[8:] 

            # Build new filename (same directory)
            name, ext = os.path.splitext(entry.name)
            new_name = f"{name}_{date}_{time}{ext}"
            new_path = os.path.join(directory, new_name)
    
            # Rename file
            os.rename(entry.path, new_path)
            #print(f"Renamed to: {new_name}")
            
            
            
    print('')
    print('')
    
    print('---------------------------')
    print('All files have been renamed')
    print('---------------------------')

