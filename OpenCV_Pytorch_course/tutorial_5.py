# Import torch and torchvision modules
from torchvision import models # To load any classification model.
from PIL import Image, ImageDraw, ImageFont # To read images from disk.
from torchvision import transforms # To apply PyTorch transformations

import os
import requests # To download file.
import cv2 # For annotating images.
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt # To visualize images.
from zipfile import ZipFile
from urllib.request import urlretrieve

def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)

URL = r"https://www.dropbox.com/s/8srx6xdjt9me3do/TF-Keras-Bootcamp-NB07-assets.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), "PyTorch-Bootcamp-NB07-assets.zip")

# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)


# Print the models available in torchvision
dir(models)


# Specify image transformations.
transform = transforms.Compose([
                  transforms.Resize(256),     #Resize the image to 256×256 pixels.
                  transforms.CenterCrop(224), #Crop the image to 224×224 pixels about the center.
                  transforms.ToTensor(),      #Convert the image to PyTorch Tensor data type.
                  transforms.Normalize(
                  mean=[0.485, 0.456, 0.406], #Normalize the image with imagenet mean and std.
                  std=[0.229, 0.224, 0.225]
                  )])



# Download imagenet classes text file.
# !wget -q  'https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt' -O'imagenet_classes.txt'


# Load resnet18 model
model = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1) #models.resnet18(weights = "DEFAULT")

# Load alexnet model
#model = models.alexnet(weights = "DEFAULT")

#Load vgg16 model
#model = models.vgg16(weights = "DEFAULT")

# Put our model in eval mode to do inference
model.eval()
print(model)

# Read the image.
img = Image.open("images/baseball-player.png")
plt.imshow(img)
plt.axis('off')
plt.show()


# Pytorch process image data in batches thus even one image is processed , tensor has to have batch dimmension
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0) #Add batch dimension [C,H,W] --> [B,C,H,W]

# Carry out inference
out = model(batch_t)
print(out.shape) # [B,num_classes]

# Load labels
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]


# Get top k predictions
_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
[(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]


def visualize_predictions(img, class_name, conf):
    """
    Function to visualize results:
    :param img: PIL Image
    :param class_name: Class name string
    :param conf: Prediction confidence string
    """
    bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_h, img_w = bgr_img.shape[:2]

    # Define font scale and thickness based on image height
    font_scale = max(0.003 * img_h, 0.5)
    thickness = max(1, int(img_h / 200))

    text = f"{class_name}, {conf}%"

    # Calculate text size to center it
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (img_w - text_w) // 2
    text_y = (img_h + text_h) // 10

    cv2.putText(
        img=bgr_img,
        org=(text_x, text_y),
        text=text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        color=(0, 0, 255),
        fontScale=font_scale,
        thickness=thickness,
        lineType=cv2.LINE_AA
    )

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.show()

class_name = classes[indices[0][0]]
conf = f"{percentage[indices[0][0]].item():.1f}"
visualize_predictions(img, class_name, conf)


def prediction(img_path, model):
    model.eval()
    img = Image.open(img_path)
    img_t = transform(img).unsqueeze(0)
    out = model(img_t)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
    class_name = classes[indices[0][0]].split(',')[0]
    conf = f"{percentage[indices[0][0]].item():.1f}"

    return img, class_name, conf


for img_path in os.listdir("images"):
    img_path = os.path.join("images", img_path)
    img, class_name, conf = prediction(img_path, model)

    visualize_predictions(img, class_name, conf)