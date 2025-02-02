#!/usr/bin/env python3
import time
import os 
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from clearml import Task, Logger
from torchvision.models import resnet18, ResNet18_Weights


SMOKING = 1
NO_SMOKING = 0

def get_label_by_filename(file_name: str):
    return NO_SMOKING if file_name.startswith("notsmoking_") else SMOKING


class SmokingDataset(datasets.VisionDataset):
    def __init__(self, data_dir, include_transform=True):
        self.data_dir = data_dir
        self.transform = None if not include_transform else self.set_transform()
        self.file_list = os.listdir(data_dir)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        
        label = NO_SMOKING if file_name.startswith("notsmoking_") else SMOKING

        image = Image.open(file_path).convert("RGB")
        
        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)
        
        return image, label, file_name

    def set_transform(self):
        return transforms.Compose([
            transforms.Resize((150, 150)),  # שינוי גודל
            transforms.RandomHorizontalFlip(p=0.5),  # היפוך אופקי עם הסתברות של 50%
            transforms.RandomRotation(degrees=30),  # סיבוב עד 30 מעלות לכל כיוון
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # שינוי צבעים קל
            transforms.ToTensor(),  # המרת תמונה ל-Tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # נרמול לערכים בין -1 ל-1
        ])


class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super(BinaryClassificationModel, self).__init__()
        self.conv = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),  
            
            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            # Fourth convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        # Fully connected layers
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9 * 9, 128),  
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),  
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fully_connected(x)
        return x


def test_model(model, device, validation_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for index, (images, labels, file_names) in enumerate(validation_loader):
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            outputs = model(images).squeeze()
            predicted = (outputs > 0.5).float()

            # Print the corresponding file names
            mismatched_indices = (predicted != labels).nonzero(as_tuple=True)[0]

            if len(mismatched_indices) > 0:
                print("Mismatched predictions found in files:")
                for idx in mismatched_indices:
                    print(f"----- Mismatched prediction: ----- \nlabel: {labels[idx]}, \npredict: {predicted[idx]},\noutput: {outputs[idx]}, filename: {file_names[idx]}\n---------")

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            
            
    
    return f'{100 * correct / total:.2f}'

def train_epoch(model, device, train_loader):
    loss_function = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    running_loss = 0.0
    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device, dtype=torch.float32)

        # Forward pass
        outputs = model(images).squeeze()
        loss = loss_function(outputs, labels)
        
        # Bpadding=1ackward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss

    
def main():
    task = Task.init(project_name='Ex4', task_name=f'Trainning Custom CNN - V2 {time.time()}')
    # Prepare the tranning dataset
    train_dataset: datasets.ImageFolder = SmokingDataset('student_318411840_v2/Training/Training/smoking')
    validate_dataset: datasets.ImageFolder = SmokingDataset('student_318411840_v2/Validation/Validation/smoking')
    test_dataset: datasets.ImageFolder = SmokingDataset('student_318411840_v2/Testing/Testing/smoking')
 
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validate_dataset, batch_size=2, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 0

    model_path = "models_pth/model.pth"

    print("Model file found. Loading the model...")
    model = BinaryClassificationModel()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        
    # Define the model, Loss Function and Optimizer
    model: nn.Module = BinaryClassificationModel()
   

    # Train the Model
    model.to(device) 

    for epoch in range(num_epochs):
        print(f"\n======= Epoch [{epoch+1}/{num_epochs}] =======")
        running_loss = train_epoch(model, device, train_loader)
        accuracy = test_model(model, device, validation_loader)
        
        print(f"Loss: {running_loss/len(train_loader):.4f} \nValidation Accuracy: {accuracy}%")
        
        Logger.current_logger().report_scalar(
            title="Loss VS Epoch", series="Loss", iteration=epoch, value=running_loss
        )
        Logger.current_logger().report_scalar(
                title="Accuracy VS Epoch", series="Accuracy", iteration=epoch, value=accuracy
        )
            
    torch.save(model.state_dict(), model_path)


    # accuracy = test_model(model, device, test_loader)
    # print(f"\nValidation Accuracy on the testing set: {accuracy}%")





if __name__ == '__main__':
    main()
   