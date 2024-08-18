import torch
import tensorflow as tf


points = torch.tensor([1.0,4.0,2.0,1.0,3.0,5.0])
points_tf = tf.constant([1.0,4.0,2.0,1.0,3.0,5.0])

print(points[3])
print(points_tf[3])


print(points.shape)
print(points_tf.shape)

#one dimension flattened tensor
points = torch.tensor([[1.0,4.0],[2.0,1.0],[3.0,5.0]])
print(points.storage())

print(points.size())

points_tf = tf.constant([[1.0,4.0],[2.0,1.0],[3.0,5.0]])
print(points_tf.shape)

#offset of tensor - index of the first element of the tensor in the storage array
print(points.storage_offset())

print(points[1].storage_offset()) #it will be 2

#tensor stride - how many elements is needed to skip to get to the nex tensor
print(points.stride())

#specify datatype in pytorch
points = torch.tensor([[1.0,2.0],[3.0,4.0]],dtype=torch.float32)
points_tf = tf.constant([[1.0,2.0],[3.0,4.0]],dtype=tf.float32)

#store tensor elements to cpu
points = torch.tensor([[1.0,2.0],[3.0,4.0]],dtype=torch.float32,device='cpu')
print(points)
#copy points tensor to gpu
points_2 = points.to(device='cuda')
print(points_2)
