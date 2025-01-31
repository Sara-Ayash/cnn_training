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
        
        # התאמת השכבה האחרונה לסיווג בינארי
        num_features = self.resnet.fc.in_features  # מספר הפיצ'רים שיוצאים מהשכבה האחרונה
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1),  # שכבה אחת עם נוירון אחד
            nn.Sigmoid()  # סיגמואיד להסתברות בינארית
        )

    def forward(self, x):
        return self.resnet(x)


class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super(BinaryClassificationModel, self).__init__()
        self.conv = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Output: 250x250
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Down-sample: 125x125
            
            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: 125x125
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # Down-sample: 63x63
            
            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 63x63
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Down-sample: 32x32
            
            # Fourth convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: 32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Down-sample: 16x16
        )
        
        # Fully connected layers
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 15 * 15, 128),  # Adjust input size for 16x16 feature maps
            nn.ReLU(),
            nn.Linear(128, 1),  # Single output for binary classification
            nn.Sigmoid()  # Sigmoid activation for probabilities
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
        for images, labels, file_names in validation_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            outputs = model(images).squeeze()
            predicted = (outputs > 0.5).float()

            # print("labels:\n", labels)
            # print("predicted:\n", predicted)
            # print("outputs:\n", outputs)
            # print("file_names:\n", file_names)

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
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
    task = Task.init(project_name='Ex4', task_name=f'Trainning Custom CNN - test on Testing image{time.time()}')
    # Prepare the tranning dataset
    train_dataset: datasets.ImageFolder = SmokingDataset('student_318411840/Training/Training/smoking')
    validate_dataset: datasets.ImageFolder = SmokingDataset('student_318411840/Validation/Validation/smoking')
 
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validate_dataset, batch_size=20, shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 10

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
            title="Accuracy VS Epoch", series="Loss", iteration=epoch, value=accuracy
        )
            
    torch.save(model.state_dict(), model_path)


    # accuracy = test_model(model, device, validation_loader)





if __name__ == '__main__':
    main()
   