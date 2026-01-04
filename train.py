import os
import glob
import numpy as np
import pandas as pd
import kagglehub

import torch
import torch.nn as torch_nn
import torch.optim as torch_optim

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def download_dataset():
    return kagglehub.dataset_download(
        "iamsouravbanerjee/animal-image-dataset-90-different-animals"
    )


def build_image_dataframe(root_directory):
    records = []

    for label_name in os.listdir(root_directory):
        label_directory = os.path.join(root_directory, label_name)
        if not os.path.isdir(label_directory):
            continue

        for filename in os.listdir(label_directory):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                records.append({
                    "filepath": os.path.join(label_directory, filename),
                    "label": label_name,
                    "filename": filename
                })

    return pd.DataFrame(records)


class ImageDataFrameDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["filepath"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_index = row["label_index"]
        return image, label_index


class AnimalClassificationModel(torch_nn.Module):
    def __init__(self, inner_layer_size=256, dropout_rate=0.2, number_of_classes=90):
        super().__init__()

        self.backbone = models.resnet18(weights="IMAGENET1K_V1")

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        input_features = self.backbone.fc.in_features
        self.backbone.fc = torch_nn.Identity()

        self.inner_layer = torch_nn.Linear(input_features, inner_layer_size)
        self.activation = torch_nn.ReLU()
        self.dropout = torch_nn.Dropout(dropout_rate)
        self.output_layer = torch_nn.Linear(inner_layer_size, number_of_classes)

    def forward(self, inputs):
        features = self.backbone(inputs)
        features = self.inner_layer(features)
        features = self.activation(features)
        features = self.dropout(features)
        outputs = self.output_layer(features)
        return outputs


def create_model_and_optimizer(device, learning_rate, inner_layer_size, dropout_rate):
    model = AnimalClassificationModel(
        inner_layer_size=inner_layer_size,
        dropout_rate=dropout_rate,
        number_of_classes=90
    ).to(device)

    optimizer = torch_optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=learning_rate
    )

    return model, optimizer


def train_one_epoch(model, data_loader, optimizer, loss_function, device):
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    average_loss = running_loss / len(data_loader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def validate(model, data_loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            total_loss += loss.item()
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def main():
    dataset_path = download_dataset()
    image_root = os.path.join(dataset_path, "animals", "animals")

    dataframe = build_image_dataframe(image_root)

    label_encoder = LabelEncoder()
    dataframe["label_index"] = label_encoder.fit_transform(dataframe["label"])

    training_dataframe, test_dataframe = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = 224
    batch_size = 20
    epochs = 60

    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=27),
        transforms.RandomResizedCrop(size=image_size, scale=(0.85, 1.0)),
        transforms.ColorJitter(contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    validation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    training_dataset = ImageDataFrameDataset(training_dataframe, training_transform)
    test_dataset = ImageDataFrameDataset(test_dataframe, validation_transform)

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model, optimizer = create_model_and_optimizer(
        device=device,
        learning_rate=0.0005,
        inner_layer_size=256,
        dropout_rate=0.2
    )

    loss_function = torch_nn.CrossEntropyLoss()

    best_validation_accuracy = 0.0
    checkpoint_template = "Model_Files/animals_classification_{epoch:02d}_{accuracy:.3f}.pth"
    os.makedirs("Model_Files", exist_ok=True)

    for epoch in range(epochs):
        training_loss, training_accuracy = train_one_epoch(
            model, training_loader, optimizer, loss_function, device
        )

        validation_loss, validation_accuracy = validate(
            model, test_loader, loss_function, device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {training_loss:.4f}, Train Accuracy: {training_accuracy:.4f} | "
            f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            checkpoint_path = checkpoint_template.format(
                epoch=epoch + 1,
                accuracy=validation_accuracy
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print("Training finished")

    checkpoint_files = glob.glob("Model_Files/animals_classification_*.pth")
    best_checkpoint = max(checkpoint_files, key=os.path.getctime)

    print(f"Loading best model: {best_checkpoint}")

    model.load_state_dict(torch.load(best_checkpoint))
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    onnx_output_path = "animals_classification_latest.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print(f"Model exported to {onnx_output_path}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
