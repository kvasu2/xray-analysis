import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score
import pandas as pd
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# Define the number of epochs
num_epochs = 1

# Define the path to save the model
model_path = os.path.join(parent_dir, 'models', 'CNN_multilabel.pth')

# Define the input shape and number of classes
input_shape = (1, 224, 224)
num_classes = 15  # Adjust this based on the number of diseases you're classifying

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
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # Changed to output 1 channel
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(28 * 28, 64)  # Adjusted input size
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.feature_maps = None

    def forward(self, x):
        self.feature_maps = self.features(x)
        x = self.feature_maps.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        fc2_output = self.sigmoid(self.fc2(x))
        return fc2_output, x

def get_gradcam(model, image, target_class):
    model.eval()
    image = image.requires_grad_()
    
    # Forward pass
    output,_ = model(image)
    model.feature_maps.retain_grad()  # Retain gradients for feature maps
    
    # Get the score for the target class
    score = output[0][target_class]
    
    # Backward pass
    score.backward()
    
    # Get the gradients and feature maps
    gradients = model.feature_maps.grad[0, 0]  # Shape: [28, 28]
    feature_maps = model.feature_maps[0, 0]    # Shape: [28, 28]
    
    # Create heatmap
    heatmap = gradients * feature_maps
    
    # Apply ReLU to the heatmap
    heatmap = F.relu(heatmap)
    
    # Normalize the heatmap
    heatmap = heatmap / torch.max(heatmap)
    
    return heatmap.detach().cpu().numpy()

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def predict_diseases(model, image_path):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move model to device
    
    image = Image.open(image_path).convert('L')
    image = image.resize((224, 224), Image.LANCZOS)  # Downscale the image
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    
    # Get predictions
    with torch.no_grad():
        output,_ = model(image_tensor)
    
    predictions = output.cpu().numpy()[0]
    
    # Get the class with the highest prediction
    target_class = np.argmax(predictions)
    
    # Generate Grad-CAM heatmap
    heatmap = get_gradcam(model, image_tensor, target_class)
    
    # Superimpose the heatmap on original image
    img = np.array(image)
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = Image.fromarray(heatmap_resized).resize((img.shape[1], img.shape[0]))
    heatmap_resized = np.asarray(heatmap_resized)
    
    superimposed_img = heatmap_resized * 0.4 + img
    superimposed_img = superimposed_img / np.max(superimposed_img)
    
    # # Display the original image and the superimposed image
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax1.imshow(img, cmap='gray')
    # ax1.set_title('Original Image')
    # ax1.axis('off')
    # ax2.imshow(superimposed_img, cmap='jet')
    # ax2.set_title('Grad-CAM Heatmap')
    # ax2.axis('off')
    # plt.tight_layout()
    # plt.show()
    
    return predictions, superimposed_img
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def visualize_cam(model, image_path, target_class):

    model.eval()
    model.to(device)

    image = Image.open(image_path).convert('L')
    image = image.resize((224, 224), Image.LANCZOS)  # Downscale the image
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    # Generate the CAM
    heatmap = get_gradcam(model, image_tensor, target_class)
    
    # Superimpose the heatmap on original image
    img = np.array(image)
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = Image.fromarray(heatmap_resized).resize((img.shape[1], img.shape[0]))
    heatmap_resized = np.asarray(heatmap_resized)
    
    superimposed_img = heatmap_resized * 0.4 + img
    superimposed_img = superimposed_img / np.max(superimposed_img)
    
    # Display the original image and the superimposed image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(superimposed_img, cmap='jet')
    ax2.set_title('Grad-CAM Heatmap')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()


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

    # Example usage of predict_diseases
    # image_path = os.path.join(parent_dir, 'data', 'sample', 'images', '00000013_005.png')
    # predictions, heatmap = predict_diseases(ChestXRayCNN(), image_path)
    # print("Predictions:", predictions)

if __name__ == "__main__":
    main()