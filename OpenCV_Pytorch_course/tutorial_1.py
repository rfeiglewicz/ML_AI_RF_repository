
#Download some images from MNIST dataset
# !wget -q "https://learnopencv.com/wp-content/uploads/2024/07/mnist_0.jpg" -O "mnist_0.jpg"
# !wget -q "https://learnopencv.com/wp-content/uploads/2024/07/mnist_1.jpg" -O "mnist_1.jpg"

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

# check Pytorch version
print("torch version: {}".format(torch.__version__))

#converting image to numpy array
digit_0_array_og = cv2.imread("mnist_0.jpg")
digit_1_array_og = cv2.imread("mnist_1.jpg")

digit_0_array_gray = cv2.imread("mnist_0.jpg", cv2.IMREAD_GRAYSCALE)
digit_1_array_gray = cv2.imread("mnist_1.jpg", cv2.IMREAD_GRAYSCALE)

# Visualize the image

fix,axs = plt.subplots(1,2, figsize=(10,5))

axs[0].imshow(digit_0_array_og, cmap='gray',interpolation='none')
axs[0].set_title("Digit 0 Image")
axs[0].axis('off')

axs[1].imshow(digit_1_array_og, cmap='gray',interpolation='none')
axs[1].set_title("Digit 1 Image")
axs[1].axis('off')

plt.show()

#Numpyt arrary with three channels
print("Image array shape: ", digit_0_array_og.shape)
print(f"Min pixel value: {np.min(digit_0_array_og)} ; Max pixel value : {np.max(digit_0_array_og)}")

# Pixel values of single channel image
print(digit_0_array_gray)

#---------------------------------------
# Converting numpy array to torch tensor

#Convert and normalize to (0,1) range
img_tensor_0 = torch.tensor(digit_0_array_og, dtype=torch.float32) / 255.0 
img_tensor_1 = torch.tensor(digit_1_array_og, dtype=torch.float32) / 255.0

print("Shape of Normalised Digit 0 Tensor: ", img_tensor_0.shape)
print(f"Normalised Min pixel value: {torch.min(img_tensor_0)} ; Normalised Max pixel value : { torch.max(img_tensor_0)}")

plt.imshow(img_tensor_0, cmap="gray")
plt.title("Normalised Digit 0 Image")
plt.axis('off')
plt.show()

# Creating Input Batch

batch_tensor = torch.stack([img_tensor_0, img_tensor_1])

# In Pytorch the forward pass of input images to the model is expected to have a batch_size > 1 
print("Batch Tensor Shape", batch_tensor.shape)

#Change the order of tensor dimmensions
batch_input = batch_tensor.permute(0,3,1,2)
print("Batch Tensor Shape", batch_input.shape)

#-------------------------------------------------------------

# Tensor with jst ones in a column
a = torch.ones(5)
# Print the tensor 
print(a)

# Create a Tensor with just zeros in a column
b = torch.zeros(5)
print(b)

# Tensor with custom values
c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(c)

d = torch.zeros(3,2)
print(d)

e = torch.ones(3,2)
print(e)

f = torch.tensor([[1.0,2.0],[3.0,4.0]])
print(f)

# 3D Tensor
g = torch.tensor([[[1.,2.], [3.,4.]], [[5.,6,], [7., 8.]]])
print(g)

# Check out the shape of created tensors using .shape
print(f.shape)
print(e.shape)
print(g.shape)

#--------------------------
# Access an element in Tensor

# Get element at index 2
print(c[2])

# All indices starting from 0

# row 1 column 0 
print(f[1,0])
# the same with different notation
print(f[1][0])

# Similarly for 3D tensor
print(g[1,0,0])
print(g[1][0][0])

#-------------

# all elements
print(f[:])

# All elements from index 1 to 3 (excluding element 3)
print(c[1:3])

# All elements till index 4 (exclusive)
print(c[:4])

# First row 
print(f[0, :])

# Second column
print(f[:,1])

#------------------------------------------------
# Specify data type of elements
# Tensor can only store one data type

int_tensor = torch.tensor([[1,2,3],[4,5,6]])
print(int_tensor.dtype)

# What if we changed any one element to floating point number ?
int_tensor = torch.tensor([[1,2,3],[4.,5,6]])
print(int_tensor.dtype)
print(int_tensor)

# This can be overriden as follows
float_tensor = torch.tensor([[1,2,3,], [4.,5,6]])
int_tensor = float_tensor.type(torch.int64) # explicitly change tensor to integer tensor
print(int_tensor.dtype)
print(int_tensor)

#-----------------------------------------
# Changing torch tensor to numpy array and vice versa

# Tensor to Array
f_numpy = f.numpy()
print(f_numpy)

# Array to Tensor
h = np.array([[8,7,6,5], [4,3,2,1]])
h_tensor = torch.from_numpy(h)
print(h_tensor)

#----------------------------------------------------
# Tensor arithmetic operations

# Create tensor 
tensor1 = torch.tensor([[1,2,3],[4,5,6]])
tensor2 = torch.tensor([[-1,2,-3],[4,-5,6]])

# Addition
print(tensor1 + tensor2)
# Alternatively 
print(torch.add(tensor1,tensor2))

# Subtraction
print(tensor1 - tensor2)
# Alternatively
print(torch.sub(tensor1,tensor2))

# Multiplication
# Tensor with scalar
print(tensor1 * 2)

# Tensor with another tensor
# Elementwise Multiplication
print(tensor1 * tensor2)

# Matrix multiplication
tensor3 = torch.tensor([[1,2],[3,4],[5,6]])
print(torch.mm(tensor1,tensor3))

# Division
# Tensor with scalar
print(tensor1 / 2)

#-------------------------------------------------
# Broadcasting
# When adding a and b , Pytorch broadcasts b to match the shape of a, resulting in ([1+4, 2+4, 3+4])

a = torch.tensor([1,2,3])
b = torch.tensor([4])

# adding a scalar to a vector
result = a + b

print("Result of Broadcasting:\n", result)

# Create two tensors with shapes (1,3) and (3,1)
a = torch.tensor([1,2,3])
b = torch.tensor([[4], [5], [6]])

# adding tensors of different shapes
result = a + b 
print("Shape: ", result.shape)
print("\n")
print("Result of Broadcasting:\n", result)

#--------------------------------------------------
# CPU vs GPU Tensor

# Create a tensor for CPU
# This will occupy CPU RAM
tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cpu')

# Create a tensor for GPU
# This will occupy GPU RAM
tensor_gpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
print(tensor_gpu.shape)
print(tensor_gpu.device)


# This uses CPU RAM
tensor_cpu = tensor_cpu * 5

#This uses GPU RAM
# Focus on GPU RAM Consumption
tensor_gpu = tensor_gpu * 5

# Move GPU tensor to CPU
tensor_gpu_cpu = tensor_gpu.to(device='cpu')

# Move CPU tensor to GPU
tensor_cpu_gpu = tensor_cpu.to(device='cuda')