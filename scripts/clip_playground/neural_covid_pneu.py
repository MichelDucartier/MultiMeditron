from random import choices, seed
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim

from load_from_clip import load_model, preprocess_dataset, make

# Specify here the CLIP-based models you want to test
# The first element in the tuple is used for naming the model in the logs and when saving the weights of the classification neural networl
# So write whatever you want that makes you tell the models apart

# The second element in the tuple is either a reference to a Hugging Face model or the local path to the model/checkpoint
clips = [
    ("full_us", "../clip-splade-US/checkpoint-468/"),
    ("train_us", "../clip-splade-US-train/checkpoint-270/"),
    ("train_us_more", "../clip-splade-US-train/checkpoint-531/"),
    ("standard_clip", "openai/clip-vit-base-patch32"),
    ("train_us_1e", "../clip-splade-US-train-1e/"),
    ("train_us_fast", "../clip-splade-US-train-fast/")
]

# Specify here whether you want the script to save the weights of the classification neural network
# The script will save one file per CLIP-based model
save_nn = True

# Logs
with open("logs_neural_covid_pneu.txt", "w"):
    pass

def printNew(*args):
    print(*args)
    
    with open("logs_neural_covid_pneu.txt", "a") as f:
        f.write(" ".join(str(x) for x in args) + "\n")

# Get the data from COVID-US
printNew("Loading dataset…")
dataset_path = "/mloscratch/homes/nemo/training/US/COVID-US/"
with open(os.path.join(dataset_path, "COVID-US-test.jsonl"), "r") as f:
    lines = [json.loads(line) for line in f if "healthy" in line or "COVID" in line or "pneumonia" in line]
printNew(len(lines), "examples")

chosen = lines

#preprocessing for the model
printNew("Processing dataset…")
chosen_img = preprocess_dataset(chosen, dataset_path)

for name_clip, path_model in clips:
    seed(42)
    np.random.seed(42)

    #compute the image embeddings
    embeds_path = os.path.join(dataset_path, f"images-{name_clip}-covid-pneu.pt")
    if not os.path.exists(embeds_path):
        # Convert textual labels to numeric
        label_to_idx = {"healthy": 0, "covid-19": 1, "pneumonia": 2}
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        labels = ["healthy" if "healthy" in line["text"] else "covid-19" if "COVID" in line["text"] else "pneumonia" for line in chosen]
        labels_numeric = np.array([label_to_idx[label] for label in labels])
        printNew(sum(x == "healthy" for x in labels), sum(x == "covid-19" for x in labels), sum(x == "pneumonia" for x in labels), len(labels))

        # Get image embeddings
        printNew("Loading CLIP model…")
        model = load_model(path_model)
        model.eval()
        printNew("Making embeds…")
        image_embeds = make(model, chosen_img)

        X = torch.tensor(image_embeds, dtype=torch.float32)
        y = torch.tensor(labels_numeric, dtype=torch.long)

        torch.save(X, embeds_path)
        torch.save(y, embeds_path+"-y.pt")
    else:
        printNew("Loading embeddings…")
        X = torch.load(embeds_path)
        y = torch.load(embeds_path+"-y.pt")

    # Balance the dataset so that labels are balanced
    def balance_classes(X_train, y_train):
        X_train_np = X_train.numpy()
        y_train_np = y_train.numpy()

        # Separate minority and majority class indices
        healthy_indices = np.where(y_train_np == 0)[0]
        covid_indices = np.where(y_train_np == 1)[0]
        pneu_indices = np.where(y_train_np == 2)[0]

        # Randomly sample from majority class to match minority class count
        covid_sampled_indices = np.random.choice(covid_indices, size=len(healthy_indices), replace=False)
        pneu_sampled_indices = np.random.choice(pneu_indices, size=len(healthy_indices), replace=False)

        # Combine the indices and shuffle
        balanced_indices = np.concatenate([healthy_indices, covid_sampled_indices, pneu_sampled_indices])
        np.random.shuffle(balanced_indices)

        # Create balanced datasets
        X_balanced = torch.tensor(X_train_np[balanced_indices], dtype=torch.float32)
        y_balanced = torch.tensor(y_train_np[balanced_indices], dtype=torch.long)
        return X_balanced, y_balanced

    X, y = balance_classes(X, y)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(X))
    test_size = len(X) - train_size
    X_train, X_test = torch.utils.data.random_split(X, [train_size, test_size])
    y_train, y_test = torch.utils.data.random_split(y, [train_size, test_size])

    # Define a simple neural network
    class SimpleNN(nn.Module):
        def __init__(self, input_size, num_classes):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x

    # Initialize the neural network
    input_size = 512
    num_classes = 3
    model = SimpleNN(input_size, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.dataset)
        loss = criterion(outputs, y_train.dataset)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            printNew(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.dataset)
        y_pred_classes = torch.argmax(y_pred, axis=1)
        accuracy = (y_pred_classes == y_test.dataset).float().mean()
        printNew("CLIP name:", name_clip)
        printNew(f"Accuracy: {accuracy:.2f}")
        printNew("Classification Report:")
        printNew(classification_report(y_test.dataset.numpy(), y_pred_classes.numpy(), target_names=["healthy", "covid-19", "pneumonia"], labels=[0, 1, 2]))

    # Save the trained model (optional)
    if save_nn:
        torch.save(model.state_dict(), f'models/{name_clip}-covid-pneu.pth')

    # Load the model later with:
    # model.load_state_dict(torch.load(f'models/{name_clip}-covid-pneu.pth'))
