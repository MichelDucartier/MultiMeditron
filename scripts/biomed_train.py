import argparse
import os
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torch
import torchvision
import clip
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--data_url', type=str, required=True, help='Base directory for images')
parser.add_argument('--output_dir', type=str, default="output", help='Output directory for the model')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train the model')
parser.add_argument('--save_model', type=bool, default=False, help='Save the model')

args = parser.parse_args()

print("Loading dataset…")
url = args.data_url
if ".jsonl"in url:
    jsonl = os.path.basename(url)
    url = os.path.dirname(url)
    dataset = load_dataset(url, data_files=jsonl)
else:
    dataset = load_dataset(args.data_url)
print("Done!")

# Prepare the model and the dataset, as it depends on the ResNet size
print("Preparing model…")
_, processor = create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', clean_up_tokenization_spaces=True)

#check whether there is already a local copy of the model in the output folder
#it is assumed to be the "better" version of the model, since it is already fine-tuned to some extent
model_path = os.path.join(args.output_dir, "model.pth")
if os.path.exists(model_path):
    model = torch.load(model_path)
else:
    model = _

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Done!")

print("Preprocessing the data…")
def preprocess(input_data):
    ret = {"images": [os.path.join(url, x["value"]) for x in input_data["modalities"] if x["type"] == "image"], "text": input_data["text"]}
    return ret

train_test_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

train_dataset = train_dataset.map(preprocess, batched=False)
eval_dataset = eval_dataset.map(preprocess, batched=False)
print("Done!")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()

# Prepare the dataset
class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract the image and text data from the dataset
        
        images = processor(Image.open(self.dataset[idx]['images'][0])) #takes only the first image
        text = tokenizer(self.dataset[idx]['text'], context_length=256).squeeze(0).to(device)
        
        return images, text
    
# Initialize the dataset
train_dataset = ImageTextDataset(train_dataset)
eval_dataset = ImageTextDataset(eval_dataset)
# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
# Ensure the model is on the correct device
model.to(device)

print("Preparation of the dataset completed! Starting the training!")

# Training loop
for epoch in range(args.num_epochs):
    print("Epoch", epoch+1)
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
        image_features, text_features, logit_scale = model(images, titles)
        
        logits = (logit_scale * image_features @ text_features.T).float().to(device)

        # Compute ground truth
        ground_truth = torch.arange(images.shape[0], device=device, dtype=torch.long)

        # Calculate the loss for both images and text
        loss = loss_img(logits, ground_truth)

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
        image_features, text_features, logit_scale = model(image, title)
        
        logits = (logit_scale * image_features @ text_features.T).float().to(device)
        ground_truth = torch.arange(image.shape[0], device=device, dtype=torch.long)
        loss = loss_img(logits, ground_truth)
        eval_loss += loss.item()
    print(f"Validation Loss: {(eval_loss / count):.4f}")
    
    model.train()

#saving the model in the output folder
if args.save_model:
    for param in model.parameters():
        param.data = param.data.contiguous()
        
    torch.save(model, model_path)