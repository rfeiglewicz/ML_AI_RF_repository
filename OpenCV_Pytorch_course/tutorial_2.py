# Pytorch autograd tutorial

import torch
import matplotlib.pyplot as plt


# Create tensors with requires_grad=True
# saving gradient in memory, indicating that we want to compute gradient for these vectors
x = torch.tensor([2.0, 5.0], requires_grad=True) 
y = torch.tensor([3.0, 7.0], requires_grad=True)

# Perform some operations
z = x * y + y**2

z.retain_grad() # By default intermediate layer weight updation is not shown

# Compute the gradients
z_sum = z.sum().backward()

print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
print(f"Gradient of z: {z.grad}")
print(f"Result of the operation: z = {z.detach()}")

#-------------------------------------------------
# Vizualize the computation graph using torchviz

from torchviz import make_dot
# Vizualize the computation graph
dot = make_dot(z, params={"x": x, "y": y, "z" : z})
dot.render("grad_computation_graph", format="png")

img = plt.imread("grad_computation_graph.png")
plt.imshow(img)
plt.axis("off")
plt.show()

#----------------------------------------------------
# Detaching Tensor from coputation graph
# Let's detach z from the computation graph
print("Before detaching z from computation: ", z.requires_grad)
z_det = z.detach()
print("After detaching z from computation: ", z_det.requires_grad)

x = torch.tensor(2.0, requires_grad=False)
y = torch.tensor(3.0, requires_grad=False)

# Perform coputation
z = x * y + y**2

#Compute the gradients
try:
    z.backward()
except:
    print("Gradient computation is not available")

#-------------- Quiz
import torch
Y = torch.tensor([1.0,], requires_grad=True)
with torch.no_grad():
	new_tensor = Y*2
	print(new_tensor.requires_grad, Y.requires_grad)
