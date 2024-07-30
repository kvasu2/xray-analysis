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
from torchvision.models import resnet18


# Global variables
#------------------------------------------------------------------------------------------------------------

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Define the path to the labels and images
csv_file = os.path.join(parent_dir, 'data', 'sample' ,'labels.csv')
img_dir = os.path.join(parent_dir, 'data','sample')

# csv_file = os.path.join(parent_dir, 'data' ,'labels.csv')
# img_dir = os.path.join(parent_dir, 'data')

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the number of epochs
num_epochs = 1

# Define the path to save the model
model_path = os.path.join(parent_dir, 'models', 'RESNET18_diseasedetection.pth')

# Define the input shape and number of classes
input_shape = (1, 224, 224)
num_classes = 1  # Adjust this based on the number of diseases you're classifying

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

        file_dict = {}
        subdirectories = [name for name in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, name))]
        for subdir in subdirectories:
            subdir_path = os.path.join(img_dir, subdir, 'images')
            if os.path.exists(subdir_path):
                file_list = [file for file in os.listdir(subdir_path) if file.endswith('.png')]
                for file in file_list:
                    file_dict[file] = subdir
        self.file_dict = file_dict

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.file_dict[self.data.iloc[idx, 0]], 'images' ,self.data.iloc[idx, 0])
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


class ResNet18ForChestXRay(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18ForChestXRay, self).__init__()
        self.resnet = resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x), x

def get_gradcam(model, image, target_class):
    model.eval()
    image = image.requires_grad_()
    
    # Forward pass
    output,_ = model(image)
    
    # Get the score for the target class
    score = output[0][target_class]
    
    # Backward pass
    model.zero_grad()
    score.backward(retain_graph=True)
    
    # Get the gradients from the last convolutional layer
    gradients = model.resnet.layer4[-1].conv2.weight.grad
    
    # Get the feature maps from the last convolutional layer
    feature_maps = model.resnet.layer4[-1].conv2(model.resnet.layer4[-1].conv1(model.resnet.layer4[-1-1](model.resnet.layer3(model.resnet.layer2(model.resnet.layer1(model.resnet.maxpool(model.resnet.relu(model.resnet.bn1(model.resnet.conv1(image))))))))))

    # Resize gradients to match feature maps
    gradients = F.interpolate(gradients, size=feature_maps.shape[2:], mode='bilinear', align_corners=False)
    
    # Create heatmap
    heatmap = torch.mean(gradients, dim=[0, 1]) * feature_maps.squeeze()
    
    # Apply ReLU to the heatmap
    heatmap = F.relu(heatmap)
    
    # Normalize the heatmap
    heatmap = heatmap / torch.max(heatmap)
    
    return heatmap.detach().cpu().numpy()



#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

def predict_diseases(model, image_path):
    model.eval()
    model.to(device)
    
    image = Image.open(image_path).convert('L')
    image = image.resize((224, 224), Image.LANCZOS)
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output,_ = model(image_tensor)
        predictions = output.cpu().numpy()[0]
    
    target_class = np.argmax(predictions)
    heatmap = get_gradcam(model, image_tensor, target_class)
    
    # Ensure heatmap is 2D
    if len(heatmap.shape) > 2:
        heatmap = np.mean(heatmap, axis=0)
    
    # Normalize heatmap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    
    # Convert heatmap to RGB
    # heatmap_rgb = (heatmap * 255).astype(np.uint8)
    # heatmap_rgb = np.stack([heatmap_rgb] * 3, axis=-1)

    cmap = plt.get_cmap('hot')
    heatmap_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    
    # Resize heatmap to match original image size
    heatmap_resized = Image.fromarray(heatmap_rgb).resize((image.width, image.height))
    heatmap_resized = np.array(heatmap_resized)
    
    # Convert original image to RGB for superimposing
    img_rgb = np.array(image.convert('RGB'))
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap_resized * 0.4 + img_rgb * 0.6
    superimposed_img = superimposed_img.astype(np.uint8)
    
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
    ax2.imshow(superimposed_img, cmap='hot')
    ax2.set_title('Grad-CAM Heatmap')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------


def train():
    # Initialize the model
    model = ResNet18ForChestXRay(num_classes)

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
            outputs,_ = model(batch_images)
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
                
                outputs,_ = model(batch_images)
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