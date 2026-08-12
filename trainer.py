import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader

from src.data_loader import DeepfakeDataset
from src.model_classifier import WavLMFakeDetector


def train(dataset_path: str, batch_size: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = DeepfakeDataset(dataset_path)
    if len(dataset) == 0:
        raise ValueError(
            f"No audio files found in dataset path: {dataset_path}. "
            "Expected subfolders like 'real_audio' and 'fake_audio' with .wav files."
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = WavLMFakeDetector()

    # The Dual-GPU Activation
    # If PyTorch detects more than 1 GPU, wrap the model in the DataParallel Manager
    if torch.cuda.device_count() > 1:
        print(f"Geometry shift: Activating {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Now send the (potentially wrapped) model to the hardware
    model = model.to(device)

    criterion = nn.BCELoss()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "models/Brain_V5_WavLM.pth")
    print("Model saved to models/Brain_V5_WavLM.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Train WavLM SSL deepfake detector head")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the In-The-Wild dataset root (contains real_audio/ and fake_audio/).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
    )