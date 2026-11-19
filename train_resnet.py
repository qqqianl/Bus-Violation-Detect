"""Train a ResNet18 classifier on an external ImageFolder dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def make_loader(root: Path, train: bool, batch_size: int):
    operations = [transforms.Resize((224, 224))]
    if train:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform = transforms.Compose(operations)
    dataset = datasets.ImageFolder(root / ("train" if train else "val"), transform=transform)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(1).cpu()
        correct += int((predictions == labels).sum())
        total += len(labels)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="contains train/ and val/ ImageFolder trees")
    parser.add_argument("--output", type=Path, default=Path("weights/resnet18_best.pth"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    train_set, train_loader = make_loader(args.data_root, True, args.batch_size)
    val_set, val_loader = make_loader(args.data_root, False, args.batch_size)
    if train_set.classes != val_set.classes:
        raise ValueError("train and val class folders must match")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_set.classes))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            loss = criterion(model(images.to(device)), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        score = accuracy(model, val_loader, device)
        print(f"epoch={epoch + 1} val_accuracy={score:.4f}")
        if score > best:
            best = score
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": train_set.classes}, args.output)


if __name__ == "__main__":
    main()

