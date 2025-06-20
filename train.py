import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import os
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets
from sklearn.metrics import classification_report
from configuration import CNNModel, transform

parser = argparse.ArgumentParser(description="Training settings")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs for training")
args = parser.parse_args()
epochs = args.epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=epochs):
    model.train()
    epoch_losses = []
    best_loss = float("inf")
    epochs_no_improve = 0
    start_time = time.time()
    print("Training...")

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        text = f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {time.time() - epoch_start:.2f}s"
        print(text)
        epoch_losses.append(text)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "logs/model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 3:
                text = "Early stopping triggered"
                print(text)
                epoch_losses.append(text)
                break

    os.makedirs("logs", exist_ok=True)
    torch.save(model.state_dict(), "logs/model.pth")
    return epoch_losses

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_preds, digits=4)
    print("Classification Report:", report)
    return accuracy, all_labels, all_preds

def start_training():
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    if not os.path.exists("dataset") or len([file for file in os.listdir("dataset") if file != "readme.md"]) == 0:
        os.makedirs("dataset", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        torch.save(model.state_dict(), "logs/model.pth")
        print("Model saved to \"logs/model.pth\"")
        return

    dataset = datasets.ImageFolder(root="./dataset/", transform=transform)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    class_sample_count = np.bincount(train_labels)
    class_weights = 1. / class_sample_count
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    with open("logs/train.logs", "w") as f:
        print("Number of images in training set:", len(train_dataset))
        f.write(f"Number of images in training set: {len(train_dataset)}\n")
        print("Number of images in test set:", len(test_dataset))
        f.write(f"Number of images in test set: {len(test_dataset)}\n")
        losses = train_model(model, criterion, optimizer, train_loader, val_loader)
        for loss in losses:
            f.write(f"{loss}\n")
        print("Model saved to \"logs/model.pth\"")
        all_labels, all_preds = evaluate_model(model, test_loader)
        report = classification_report(all_labels, all_preds, digits=4)
        f.write("Classification Report:\n")
        f.write(report)

if __name__ == "__main__":
    start_training()
    input("Press Enter to exit...")
