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
            transforms.Resize((250, 250)),  # Resize images to 128x128
            transforms.RandomRotation(20), 
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class ResNet18ClassificationModel(nn.Module):
    def __init__(self):
        super(ResNet18ClassificationModel, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        num_features = self.resnet.fc.in_features 
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1),  
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.resnet(x)



if __name__ == '__main__':
    task = Task.init(project_name='Ex4', task_name=f'Fine Tuning CNN {time.time()}')
    # Prepare the tranning dataset
    train_dataset: datasets.ImageFolder = SmokingDataset('student_318411840/Training/Training/smoking')
    validate_dataset: datasets.ImageFolder = SmokingDataset('student_318411840/Validation/Validation/smoking')

 
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validate_dataset, batch_size=20, shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 4

    model_path = "ft_models_pth/model.pth"

    model = ResNet18ClassificationModel()
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path, weights_only=True))
        
    loss_function = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the Model
    model.to(device) 

    for epoch in range(num_epochs):
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
        
        Logger.current_logger().report_scalar(
            title="Loss VS Epoch", series="Loss", iteration=epoch, value=running_loss
        )

            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), model_path)

    # Validate the Model for binary classification
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, file_names in validation_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            outputs = model(images).squeeze()
            predicted = (outputs > 0.5).float()

            print("labels:\n", labels)
            print("predicted:\n", predicted)
            print("outputs:\n", outputs)
            print("file_names:\n", file_names)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total:.2f}%')
