import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score
import pandas as pd
from PIL import Image
import os
import numpy as np
from tqdm import tqdm


# Global variables
#------------------------------------------------------------------------------------------------------------

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Define the path to the labels and images
csv_file = os.path.join(parent_dir, 'data', 'sample', 'labels.csv')
img_dir = os.path.join(parent_dir, 'data', 'sample', 'images')

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to save the model
model_path = os.path.join(parent_dir, 'models', 'CNN_multilabel.pth')

# Define the input shape and number of classes
input_shape = (1, 224, 224)
num_classes = 14  # Adjust this based on the number of diseases you're classifying

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

# Define the dataset class

class ChestXRayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Ensure label columns are numeric
        label_columns = self.data.columns[1:num_classes+1]
        self.data[label_columns] = self.data[label_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Open as grayscale
        
        # Downscale the image to 224x224
        image = image.resize((224, 224), Image.LANCZOS)
        
        if self.transform:
            image = self.transform(image)
        
        # Convert image to tensor and normalize
        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0
        
        # Get labels (assuming columns 1 to num_classes are the disease labels)
        labels = self.data.iloc[idx, 1:num_classes+1].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, labels

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

# Define the CNN model

class ChestXRayCNN(nn.Module):
    def __init__(self):
        super(ChestXRayCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

# Function to predict diseases
def predict_diseases(model, image_path):
    # Load the model from the file
    model = ChestXRayCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert('L')
        image = image.resize((224, 224), Image.LANCZOS)  # Downscale the image
        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        image = image.to(device)
        output = model(image)
    return output.cpu().numpy()[0]

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def train():

    # Initialize the model
    model = ChestXRayCNN()

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # Create dataset and data loader

    dataset = ChestXRayDataset(csv_file, img_dir)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training loop
    num_epochs = 3
    model.to(device)
    print("Using device:", device)

    for epoch in tqdm(range(num_epochs)):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_images, batch_labels in train_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == batch_labels).float().sum()
            train_total += batch_labels.numel()
        
        train_loss /= len(train_loader)
        train_accuracy = (train_correct / train_total) * 100

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch_labels).float().sum()
                val_total += batch_labels.numel()
        
        val_loss /= len(val_loader)
        val_accuracy = (val_correct / val_total) * 100

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    print("Training completed.")

    # Final validation accuracy using sklearn for multi-label classification
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            outputs = model(batch_images)
            predicted = (outputs > 0.5).float()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    final_accuracy = accuracy_score(all_labels.flatten(), all_predictions.flatten())
    print(f"Final Validation Accuracy: {final_accuracy * 100:.2f}%")

    # Save the model to a file
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def main():
    train()

if __name__ == "__main__":
    main()