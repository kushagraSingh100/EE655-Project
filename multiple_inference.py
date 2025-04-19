from PIL import Image
from torchvision import transforms
import torch

import torch
from model.MyNet import MyNet
import os

# Initialize model and load weights
model = MyNet()  # include any args your model needs
model.load_state_dict(torch.load('MyNet.pth'))
model.eval()
model.cuda() if torch.cuda.is_available() else model.cpu()

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((352, 352)),  # or your training size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_path = 'EORSSD/test-images/0023.jpg'
img = Image.open(img_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0)

input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor
model.eval()
with torch.no_grad():
    pred,_ = model(input_tensor)
    pred = torch.sigmoid(pred)
    pred = pred.squeeze().cpu().numpy()

# Save or view prediction
pred_img = Image.fromarray((pred * 255).astype('uint8'))
pred_img.save('prediction.png')

# model = MyNet()
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parameters: {total_params}")
# print(f"Trainable parameters: {trainable_params}")
