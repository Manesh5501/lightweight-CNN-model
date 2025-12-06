# main.py

import argparse
from model import LightweightCNN   # import your CNN class
from utils import load_data        # helper for dataset
import torch

def train():
    # Load dataset
    train_loader, test_loader = load_data()

    # Initialize model
    model = LightweightCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Save model
    torch.save(model.state_dict(), "cnn_model.pth")

def inference(image_path):
    model = LightweightCNN()
    model.load_state_dict(torch.load("cnn_model.pth"))
    model.eval()
    # Add code to preprocess image and predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], default="train")
    parser.add_argument("--image", type=str, help="Path to image for inference")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        inference(args.image)
