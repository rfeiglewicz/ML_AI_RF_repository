# Fine-tuning pretrained models for image classification

import torch  # PyTorch core library for deep learning
import torch.nn as nn  # Neural network module in PyTorch
from torch.optim import Adam  # Adam optimizer for training models
import torch.nn.functional as F  # Functional interface for PyTorch layers
from torch.utils.data import DataLoader, Dataset  # Utilities for handling datasets and data loading
from torchinfo import summary  # Prints model architecture summary
from torch.utils.tensorboard import SummaryWriter  # Logging utility for TensorBoard

import torchvision  # PyTorch's library for image processing
from torchvision import datasets  # Datasets module for common image datasets
from torchvision.transforms import v2 as transforms  # Image transformation utilities
from torchvision.ops import Conv2dNormActivation  # Convolution layers with normalization and activation

from dataclasses import dataclass  # Data structure utility for structured storage
from sklearn.metrics import confusion_matrix  # Computes confusion matrix for classification
from sklearn.model_selection import train_test_split  # Splits dataset into training and testing sets
import seaborn as sn  # Data visualization library, often used for heatmaps

import matplotlib.pyplot as plt  # Visualization library for creating plots
import time  # Provides time-related functions
import numpy as np  # Library for numerical computing and array operations
import random  # Standard library module for generating random numbers
import warnings  # Handles warning messages
import os  # Provides functionalities for interacting with the operating system
from tqdm import tqdm  # Displays progress bars for loops

import pandas as pd  # Library for data manipulation and analysis
import random  # Included twice, can be removed if unnecessary
# Ensures inline plotting in Jupyter notebooks
# %matplotlib inline
# warnings.filterwarnings("ignore")  # Suppresses warning messages

#Set seed for reproducibilty
def set_seed(seed):
    random.seed(seed)  # Set the seed for Python's built-in random module
    np.random.seed(seed)  # Set the seed for NumPy's random number generator
    torch.manual_seed(seed)  # Set the seed for PyTorch's random number generator

    if torch.cuda.is_available():  # Check if a CUDA-enabled GPU is available
       torch.cuda.manual_seed(seed)  # Set the seed for CUDA (single GPU)
       torch.cuda.manual_seed_all(seed)  # Set the seed for CUDA (multiple GPUs)
       torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CuDNN
       torch.backends.cudnn.benchmark = True  # Enable benchmark mode for CuDNN (can improve performance)

set_seed(42)  # Set the random seed to 42 for reproducibility

# Download and unzip dataset with monkey species
# !wget -q "https://www.dropbox.com/s/45jdd8padeyjq6t/10_Monkey_Species.zip?dl=1" -O "10_Monkey_Species.zip"
# !unzip -q "10_Monkey_Species.zip"


@dataclass(frozen=True)  # Creating an immutable dataclass for training configuration
class TrainingConfig:
      ''' Configuration for Training '''
      batch_size: int = 32  # Number of samples per training batch
      num_epochs: int = 20  # Number of epochs (full passes through dataset)
      learning_rate: float = 1e-4  # Learning rate for optimizer

      log_interval: int = 1  # Interval at which to log training progress
      test_interval: int = 1  # Interval at which to test model performance
      data_root: int = "./"  # Path to dataset storage
      num_workers: int = 5  # Number of worker processes for data loading
      device: str = "cuda"  # Default device for training (CUDA if available)

# Creating an instance of the training configuration
train_config = TrainingConfig()

# Determining whether to use a GPU (if available) or CPU
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print("Available Device: ", DEVICE)  # Print the selected device


# Defining the path to the training dataset
train_root = os.path.join("10_Monkey_Species", "training", "training")  # Path to training data directory

# Defining the path to the validation dataset
val_root = os.path.join(train_config.data_root, "10_Monkey_Species", "validation", "validation")  # Path to validation data directory


# Reading the monkey species labels from a CSV file
# The file is stored inside the '10_Monkey_Species' directory
# The separator is a comma, and it has no header row initially

df = pd.read_csv(os.path.join("10_Monkey_Species", "monkey_labels.txt"), sep=",", header=None)

# Assigning meaningful column names to the DataFrame
df.columns = ["Label", "Latin Name", "Common Name", "Train Images", "Validation Images"]

# Cleaning up the 'Latin Name' column by replacing tab characters with spaces
df['Latin Name'] = df['Latin Name'].str.replace("\t", " ")

# Displaying all rows except the first one

print(df[1:])


# Mean and standard deviation values computed from the Monkey Species dataset
mean = [0.4368, 0.4336, 0.3294]  # Mean pixel values for normalization (RGB channels)
std = [0.2457, 0.2413, 0.2447]  # Standard deviation pixel values for normalization (RGB channels)

# Defining the target image size for model input
img_size = (224, 224)  # Resizing images to 224x224 pixels

# Creating a preprocessing pipeline for images
preprocess = transforms.Compose(
    [
        transforms.Resize(img_size, antialias=True),  # Resize images while preserving aspect ratio
        transforms.ToTensor()  # Convert images to PyTorch tensors (scales pixel values to [0,1])
    ]
)

# Define common transformations used for both training and validation datasets
common_transforms = transforms.Compose(
    [
        preprocess, # Apply the standard preprocessing pipeline (resize, convert to tensor)
        transforms.Normalize(mean=mean, std=std)
    ]
)

# Define additional transformations for training data augmentation
train_transforms = transforms.Compose(
    [
        preprocess, # Apply the same preprocessing steps as common_transforms
        transforms.RandomHorizontalFlip(), # Randomly flip images horizontally with a probability of 0.5
        transforms.RandomErasing(p=0.4), # Randomly erase parts of the image with a probability of 0.4

        # Apply random affine transformations with controlled randomness
        transforms.RandomApply([
            transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
        ], p=0.1),  # Apply affine transformations with a probability of 0.1

        transforms.Normalize(mean=mean, std=std) # Normalize images using dataset-specific mean and std
    ]
)

# Apply augmentations to the training dataset
train_data = datasets.ImageFolder(root = train_root, transform = train_transforms)

# The validation dataset should have only common transforms like Resize, ToTensor and Normalize
val_data = datasets.ImageFolder(root=val_root, transform =  common_transforms)
print(train_data.classes)

print(train_data.class_to_idx)

print(len(train_data))


# Dataloader is a function to convert loaded images to batches
# Creating a DataLoader for training data
train_loader = DataLoader(
    train_data, # Dataset to load
    shuffle=True, # Shuffle data at every epoch to improve generalization
    batch_size=train_config.batch_size, # Number of samples per batch
    num_workers=train_config.num_workers # Number of subprocesses for data loading
)

# Creating a DataLoader for validation data
val_loader = DataLoader(
    val_data,  # Dataset to load
    shuffle=False,  # No shuffling needed for validation data
    batch_size=train_config.batch_size,  # Number of samples per batch
    num_workers=train_config.num_workers  # Number of subprocesses for data loading
)

#Map class names to class ids.
class_mapping = {

    0: "mantled_howler",
    1: "patas_monkey",
    2: "bald_uakari",
    3: "japanese_macaque",
    4: "pygmy_marmoset",
    5: "white_headed_capuchin",
    6: "silvery_marmoset",
    7: "common_squirrel_monkey",
    8: "black_headed_night_monkey",
    9: "nilgiri_langur"
}


import matplotlib.pyplot as plt  # Importing Matplotlib for visualization
import numpy as np  # Importing NumPy for numerical operations

def visualize_images(dataloader, num_images=20):
    """
    Function to visualize a batch of images from the DataLoader.

    Parameters:
    dataloader (DataLoader): The PyTorch DataLoader containing image data.
    num_images (int): Number of images to visualize.
    """
    fig = plt.figure(figsize=(10, 10))  # Create a figure with a specified size

    # Retrieve the first batch from the DataLoader
    images, labels = next(iter(dataloader))

    num_rows = 4  # Define number of rows in the grid
    num_cols = int(np.ceil((num_images / num_rows)))  # Compute number of columns

    for idx in range(min(num_images, len(images))):  # Loop through the images up to num_images
        image, label = images[idx], labels[idx]  # Extract image and label

        ax = fig.add_subplot(num_rows, num_cols, idx + 1, xticks=[], yticks=[])  # Create subplot

        image = image.permute(1, 2, 0)  # Reorder dimensions for visualization (C, H, W) → (H, W, C)

        # Normalize the image to [0,1] for display
        image = (image - image.min()) / (image.max() - image.min())
        ax.imshow(image, cmap="gray")  # Display the image
        ax.set_title(f"{label.item()}: {class_mapping[label.item()]}")  # Set title with label info

    fig.tight_layout()  # Adjust layout for better spacing
    plt.show()  # Display the figure

# Call the function to visualize a sample of images from the training dataset
visualize_images(train_loader, num_images=16)


from torchvision.models import mobilenet_v3_small

# Load the model
mobilenetv3_model = mobilenet_v3_small(weights="DEFAULT")

# Number of feature layers
num_feature_layers = len(mobilenetv3_model.features)
print(f"Number of feature layers: {num_feature_layers}")

# Number of classifier layers
num_classifier_layers = len(mobilenetv3_model.classifier)
print(f"Number of classifier layers: {num_classifier_layers}")

#Set requires_grad to True for all model parameters to allow training
# Freeze the earlier layers to retain pretrained low-level features
for param in mobilenetv3_model.features[:10].parameters():  # Adjust as needed
    param.requires_grad = False

# Check the final classifier
print(mobilenetv3_model.classifier[3])

# Replace the final linear layer of the classifier to match the number of classes in the dataset
mobilenetv3_model.classifier[3] = nn.Linear(in_features = 1024, out_features = 10, bias = True)

# Check the final classifier after modifications
print(mobilenetv3_model.classifier[3])

model = mobilenetv3_model


# Initializing the optimizer with model parameters and learning rate from the training configuration
optimizer = Adam(model.parameters(), lr=train_config.learning_rate)

# Selecting the appropriate device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Define the directory where TensorBoard logs will be stored
logdir = "runs/mobilenetv3-Monkey-Species"

# Initialize the TensorBoard SummaryWriter for logging training details
writer = SummaryWriter(logdir)

# Print a summary of the MobileNetV3 model, specifying input shape and display settings
summary(model, input_size=(1, 3, 224, 224), row_settings=["var_names"], device="cpu")

import torch.nn.functional as F  # Importing functional API for loss computation
from tqdm import tqdm  # Importing tqdm for progress bar visualization


def train(model, train_loader):
    """
    Trains the model for one epoch using the provided training data loader.

    Parameters:
    model (torch.nn.Module): The neural network model to be trained.
    train_loader (DataLoader): The DataLoader for training data.

    Returns:
    tuple: Average training loss and training accuracy.
    """
    model.train()  # Set the model to training mode
    model.to(DEVICE)  # Move the model to the appropriate device (GPU/CPU)

    running_loss = 0  # Variable to accumulate the total loss
    correct_predictions = 0  # Counter for correctly predicted samples
    total_train_samples = 0  # Counter for total training samples

    # Iterate through the training data
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)  # Move data to the selected device
        optimizer.zero_grad()  # Clear gradients from the previous step
        outputs = model(images)  # Forward pass to get predictions
        loss = F.cross_entropy(outputs, labels)  # Compute cross-entropy loss
        loss.backward()  # Backpropagation to compute gradients
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate loss
        _, predicted = torch.max(outputs.data, dim=1)  # Get predicted class labels
        total_train_samples += labels.shape[0]  # Update total number of samples
        correct_predictions += (predicted == labels).sum().item()  # Count correct predictions

    train_avg_loss = running_loss / len(train_loader)  # Compute average loss
    train_accuracy = 100 * correct_predictions / total_train_samples  # Compute accuracy percentage
    return train_avg_loss, train_accuracy  # Return loss and accuracy


import torch.nn.functional as F  # Importing functional API for loss computation
from tqdm import tqdm  # Importing tqdm for progress bar visualization


def validation(model, val_loader):
    """
    Evaluates the model performance using the provided validation data loader.

    Parameters:
    model (torch.nn.Module): The neural network model to be evaluated.
    val_loader (DataLoader): The DataLoader for validation data.

    Returns:
    tuple: Average validation loss and validation accuracy.
    """
    model.eval()  # Set the model to evaluation mode (disables dropout, batch norm updates)
    model.to(DEVICE)  # Move the model to the appropriate device (GPU/CPU)

    running_loss = 0  # Variable to accumulate the total loss
    correct_predictions = 0  # Counter for correctly predicted samples
    total_val_samples = 0  # Counter for total validation samples

    # Iterate through the validation data
    for images, labels in tqdm(val_loader, desc="Validation"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)  # Move data to the selected device

        with torch.no_grad():  # Disable gradient computation for efficiency
            outputs = model(images)  # Forward pass to get predictions

        loss = F.cross_entropy(outputs, labels)  # Compute cross-entropy loss
        running_loss += loss.item()  # Accumulate loss
        _, predicted = torch.max(outputs.data, dim=1)  # Get predicted class labels
        total_val_samples += labels.shape[0]  # Update total number of samples
        correct_predictions += (predicted == labels).sum().item()  # Count correct predictions

    val_avg_loss = running_loss / len(val_loader)  # Compute average loss
    val_accuracy = 100 * correct_predictions / total_val_samples  # Compute accuracy percentage
    return val_avg_loss, val_accuracy  # Return loss and accuracy

import torch  # Importing PyTorch for model saving

def main(model, train_loader, val_loader):
    """
    Trains and validates the model over multiple epochs.

    Parameters:
    model (torch.nn.Module): The neural network model to be trained and validated.
    train_loader (DataLoader): The DataLoader for training data.
    val_loader (DataLoader): The DataLoader for validation data.

    Returns:
    tuple: Lists of training and validation losses and accuracies.
    """
    train_losses, val_losses = [], []  # Lists to store loss values for training and validation
    train_accuracies, val_accuracies = [], []  # Lists to store accuracy values for training and validation

    best_val_acc = 0.0  # Initialize the best validation accuracy
    best_weights = None  # Store the best model weights

    # Training loop for the number of epochs specified in train_config
    for epoch in range(train_config.num_epochs):
        train_loss, train_accuracy = train(model, train_loader)  # Perform training for one epoch
        val_loss, val_accuracy = validation(model, val_loader)  # Perform validation for one epoch

        # Append losses and accuracies to the respective lists
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print training progress
        print(f"Epoch {epoch+1:0>2}/{train_config.num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Logging metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss)
        writer.add_scalar('Loss/val', val_loss)
        writer.add_scalar('Accuracy/train', train_accuracy)
        writer.add_scalar('Accuracy/val', val_accuracy)

        # Save the best model based on validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_weights = model.state_dict()
            print(f"Saving best model...💾")
            torch.save(best_weights, "best.pt")  # Save model weights to a file

    return train_losses, train_accuracies, val_losses, val_accuracies  # Return training and validation metrics

train_losses, train_accuracies, val_losses, val_accuracies = main(model, train_loader, val_loader)

import matplotlib.pyplot as plt  # Importing Matplotlib for visualization

# Create a new figure with specified size
plt.figure(figsize=(12, 5))

# Plot training and validation loss over epochs
plt.subplot(1, 2, 1)  # Create subplot (1 row, 2 columns, 1st plot)
plt.plot(range(1, train_config.num_epochs + 1), train_losses, label="Train Loss")  # Plot train loss
plt.plot(range(1, train_config.num_epochs + 1), val_losses, label="Validation Loss")  # Plot validation loss
plt.xlabel("Epochs")  # Label for x-axis
plt.ylabel("Loss")  # Label for y-axis
plt.legend()  # Show legend to differentiate lines

# Plot training and validation accuracy over epochs
plt.subplot(1, 2, 2)  # Create subplot (1 row, 2 columns, 2nd plot)
plt.plot(range(1, train_config.num_epochs + 1), train_accuracies, label="Train Accuracy")  # Plot train accuracy
plt.plot(range(1, train_config.num_epochs + 1), val_accuracies, label="Validation Accuracy")  # Plot validation accuracy
plt.xlabel("Epochs")  # Label for x-axis
plt.ylabel("Accuracy")  # Label for y-axis
plt.legend()  # Show legend to differentiate lines

plt.show()  # Display the plots

# Load the best model weights
model.load_state_dict(torch.load("best.pt"))
model.eval()


def prediction(model, val_loader):
    """
    Generates predictions using the trained model on validation data.

    Parameters:
    model (torch.nn.Module): The trained neural network model.
    val_loader (DataLoader): The DataLoader for validation data.

    Returns:
    tuple: Arrays containing images, true labels, predicted labels, and prediction probabilities.
    """
    model.eval()  # Set the model to evaluation mode
    model.to(DEVICE)  # Move the model to the appropriate device (GPU/CPU)

    all_images, all_labels = [], []  # Lists to store images and true labels
    all_pred_indices, all_pred_probs = [], []  # Lists to store predicted labels and probabilities

    # Iterate over the validation dataset
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)  # Move data to selected device

        with torch.inference_mode():  # Disable gradient calculations for efficiency
            outputs = model(images)  # Forward pass to get predictions

        prob = F.softmax(outputs, dim=1)  # Convert outputs to probabilities using softmax
        pred_indices = prob.data.max(dim=1)[1]  # Get predicted class index
        pred_probs = prob.data.max(dim=1)[0]  # Get corresponding prediction confidence

        all_images.append(images.cpu())  # Move images back to CPU and store
        all_labels.append(labels.cpu())  # Move labels back to CPU and store
        all_pred_indices.append(pred_indices.cpu())  # Store predicted indices
        all_pred_probs.append(pred_probs.cpu())  # Store predicted probabilities

    return (
        torch.cat(all_images).numpy(),  # Convert list of images to NumPy array
        torch.cat(all_labels).numpy(),  # Convert list of labels to NumPy array
        torch.cat(all_pred_indices).numpy(),  # Convert list of predicted labels to NumPy array
        torch.cat(all_pred_probs).numpy()  # Convert list of prediction probabilities to NumPy array
    )

def denormalize(image):
    """
    Reverses the normalization process applied during image preprocessing.

    Parameters:
    image (numpy.ndarray or torch.Tensor): The normalized image array.

    Returns:
    numpy.ndarray: The denormalized image array with values clipped between 0 and 1.
    """
    mean_ar = np.array(mean)  # Convert mean list to NumPy array
    std_ar = np.array(std)  # Convert standard deviation list to NumPy array

    image = image * std_ar + mean_ar  # Reverse normalization transformation
    return np.clip(image, 0, 1)  # Clip values to ensure they remain within the valid range [0,1]

def visualise_predictions(sample_images, sample_gt_labels, pred_indices, pred_probs, num_images=5):
    """
    Visualizes sample images along with their ground truth and predicted labels.

    Parameters:
    sample_images (numpy.ndarray): Array of sample images.
    sample_gt_labels (numpy.ndarray): Ground truth labels for the images.
    pred_indices (numpy.ndarray): Predicted class indices.
    pred_probs (numpy.ndarray): Prediction probabilities.
    num_images (int): Number of images to visualize.
    """
    fig = plt.figure(figsize=(20, 5))  # Create a figure with specified size

    for i in range(num_images):
        idx = random.randint(0, len(sample_images) - 1)  # Select a random index
        image = sample_images[idx].transpose(1, 2, 0)  # Convert (C,H,W) to (H,W,C) for display
        label = sample_gt_labels[idx]  # Get the ground truth label
        pred_idx = pred_indices[idx]  # Get the predicted class index
        pred_prob = pred_probs[idx]  # Get the corresponding prediction confidence

        image = denormalize(image)  # Denormalize the image for proper visualization

        ax = fig.add_subplot(1, num_images, i + 1)  # Create a subplot
        ax.imshow(image)  # Display the image
        ax.set_title(f"GT: {class_mapping[label]}\nPred: {class_mapping[pred_idx]} ({pred_prob:.2f})")  # Set title
        ax.axis('off')  # Remove axis labels for better display

    plt.show()  # Display the figure

# Generate predictions using the trained model on the validation dataset
val_images, val_gt_labels, pred_indices, pred_probs = prediction(model, val_loader)

# Visualize a subset of the predictions
visualise_predictions(val_images, val_gt_labels, pred_indices, pred_probs, num_images=5)

# Confucion matrix

# Compute the confusion matrix based on ground truth labels and model predictions
cm = confusion_matrix(y_true=val_gt_labels, y_pred=pred_indices)

# Create a figure for the confusion matrix
plt.figure(figsize=[10, 5])

# Plot the confusion matrix as a heatmap
sn.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 14})  # Display numerical values in the heatmap

# Label the axes
plt.xlabel("Predicted")  # X-axis represents predicted classes
plt.ylabel("Targets")  # Y-axis represents actual (ground truth) classes

# Set the title for the confusion matrix plot
plt.title(f"Confusion Matrix", color="gray")

# Display the plot
plt.show()
