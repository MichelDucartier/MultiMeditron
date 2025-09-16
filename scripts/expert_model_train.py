import argparse
import os
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torch
import clip
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--data_url', type=str, required=True, help='Base directory for images')
parser.add_argument('--output_dir', type=str, default="output", help='Output directory for the model')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train the model')
parser.add_argument('--save_model', type=bool, default=False, help='Save the model')

args = parser.parse_args()

dataset = load_dataset(args.data_url)

# Prepare the model and the dataset, as it depends on the ResNet size
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess(input):
    processing = processor(images=input['images'], text=input['labels'], return_tensors="pt", padding='max_length', truncation=True)
    return { 'images': processing['pixel_values'], 'labels': processing['input_ids'] }

train_test_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

#strain_dataset = train_dataset.map(preprocess, batched=False)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()

class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract the image and text data from the dataset
        image = self.dataset[idx]['images']  # Assuming the images are in the 'images' field
        label = self.dataset[idx]['labels']  # Assuming the text is in the 'labels' field
        
        # Use the processor to preprocess the image and text
        # The processor handles resizing and normalization of images, and tokenizes the text
        processed = processor(images=image, text=label, return_tensors="pt", padding='max_length', truncation=True)
        
        # Extract the pixel values and input IDs from the processed output
        # The squeeze(0) removes the batch dimension added by the processor
        pixel_values = processed['pixel_values'].squeeze(0)  # Shape: [3, height, width]
        input_ids = processed['input_ids'].squeeze(0)  # Shape: [max_length]
        
        return pixel_values, input_ids

# Initialize the dataset
train_dataset = ImageTextDataset(train_dataset)
eval_dataset = ImageTextDataset(eval_dataset)
# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
# Ensure the model is on the correct device
model.to(device)

# Training loop
for epoch in range(args.num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    total_loss = 0.0
    count = 0
    for images, titles in pbar:
        count += 1
        optimizer.zero_grad()

        # Move data to the same device as the model
        images = images.to(device)  # images should be 4D: [batch_size, 3, height, width]
        titles = titles.to(device)  # titles should be 2D: [batch_size, sequence_length]

        # Forward pass through the CLIP model
        # The model expects pixel_values for the images and input_ids for the text
        outputs = model(pixel_values=images, input_ids=titles)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        # Compute ground truth
        ground_truth = torch.arange(images.shape[0], device=device, dtype=torch.long)

        # Calculate the loss for both images and text
        loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        # Backpropagation
        loss.backward()

        # Optimizer step
        optimizer.step()
        total_loss += loss.item()

        # Update progress bar
        pbar.set_description(f"Epoch {epoch+1}, Loss: {(total_loss / count):.4f}")
    
    # Evaluate the model
    model.eval()
    eval_loss = 0.0
    count = 0
    for image, title in eval_dataloader:
        count += 1
        image = image.to(device)
        title = title.to(device)
        outputs = model(pixel_values=image, input_ids=title)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text
        ground_truth = torch.arange(image.shape[0], device=device, dtype=torch.long)
        loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        eval_loss += loss.item()
    print(f"Validation Loss: {(eval_loss / count):.4f}")
    
    model.train()

if args.save_model:
    for param in model.parameters():
        param.data = param.data.contiguous()
    model.save_pretrained(args.output_dir)