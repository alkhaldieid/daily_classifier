import os
import shutil
from PIL import Image
import torch
from torchvision import transforms, models
from tqdm import tqdm

# --- Configuration ---
model_path = "/alkhaldieid/home/repos/daily_classifier/models/best_model.pth"  # Path to saved model
input_folder = os.getcwd()            # Use current working directory
output_root = os.path.join(input_folder, "classified")  # Where to copy classified images
class_names = ['civil', 'electric', 'hvac', 'garden', 'cleaning', 'mech']
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# --- Define transform ---
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Create output directories ---
for cls in class_names:
    os.makedirs(os.path.join(output_root, cls), exist_ok=True)

# --- Process images in input_folder ---
for filename in tqdm(os.listdir(input_folder), desc="Classifying images"):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, pred_idx = torch.max(output, 1)
            predicted_label = class_names[pred_idx.item()]

        destination = os.path.join(output_root, predicted_label, filename)
        shutil.copy2(image_path, destination)

print("✅ Classification complete. Images copied into 'classified/' folder.")
