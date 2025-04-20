import torch
from model.MyNet import MyNet
from data import test_dataset
import os

# Initialize model and load weights
model = MyNet()  
model.load_state_dict(torch.load('MyNet.pth'))
model.eval()
model.cuda() if torch.cuda.is_available() else model.cpu()


# Directory paths
image_root = 'examples/test-images'  
gt_root = 'examples/test-labels' 
testsize = 352  

test_data = test_dataset(image_root, gt_root, testsize)

for i in range(test_data.size):
    image, gt, name = test_data.load_data()
    image = image.cuda() if torch.cuda.is_available() else image
    with torch.no_grad():
        prediction,_ = model(image)
        prediction = torch.sigmoid(prediction) 
        pred_np = prediction.squeeze().cpu().numpy()  

    from PIL import Image
    pred_img = Image.fromarray((pred_np * 255).astype('uint8'))
    pred_img.save(os.path.join('examples/results/', name))  
