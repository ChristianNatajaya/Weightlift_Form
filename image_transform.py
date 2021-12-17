""" This module contains functions that help
transform PIL images to the required formats """

import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models
from skimage.morphology import skeletonize

class image_transform():
    def __init__(self, width, height):
        self.width = width
        self.height = height 
    
    def toTensor(self, pil_image):
        transform_function = T.ToTensor()
        return transform_function(pil_image)

    def toPIL(self, tensor_image):
        transform_function = T.ToPILImage()
        return transform_function(tensor_image)

    def transform(self, image):
        transform_function = T.Compose(
            [T.Resize((self.height, self.width)), 
            T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]
        )
        return transform_function(image)

def semantic_segmentation(model, image):
    """ Transform image pixels with dimensions (R,G,B) to Pytorch Tensor
    Pass input through pretrained FCN to get segmented image as output """
    transform_function = image_transform(224, 224)
    image = transform_function.toTensor(image)
    cnn_input = transform_function.transform(image)
    cnn_input = cnn_input.unsqueeze(0)
    cnn_output = model(cnn_input)['out']

    """ Each pixel (H*w) in output is classified as 1 out of 21 different colors (classes), so 21 channels 
    You want to remove all the colors that are not assigned to a pixel - find the class index """
    cnn_output = torch.argmax(cnn_output.squeeze(), dim=0).detach().cpu().numpy()
    return cnn_output

def binary_transformation(segmented_image):
    # Convert segmentage image to a binary image with values (0,1) then skeletonize
    binary_constant = np.max(segmented_image)
    binary_image = np.divide(segmented_image, binary_constant)
    skeleton_image = skeletonize(binary_image)

    # Need to convert binary image [[True,False]] back to RGB image [[[R,G,B]]]
    skeleton_array = np.zeros((np.shape(skeleton_image)[0], np.shape(skeleton_image)[1], 3))
    skeleton_array[skeleton_image]  = [0,0,0]
    skeleton_array[~skeleton_image] = [255,255,255]
    skeleton_array = np.swapaxes(skeleton_array,1,2)
    skeleton_array = np.swapaxes(skeleton_array,0,1)
    return skeleton_array

def resize_to_original(original_image, skeleton_array):
    """ First convert numpy array to PIL image and transform to original dimensions 
    Then save the PIL image inside a bytes-like object 
    Then base64 encode the bytes-like object into bytes representation of base64 string """
    original_width, original_height = original_image.size
    transform_function = image_transform(original_width, original_height)
    skeleton_tensor = transform_function.transform(torch.Tensor(skeleton_array))
    skeleton_pil = transform_function.toPIL(skeleton_tensor)

    # Convert all background by its NORMALIZED RGB value (G=235) to transparent value
    original_rgba = original_image.convert("RGBA")
    original_pixels = original_rgba.getdata()
    skeleton_pixels = skeleton_pil.getdata()
    overlay_image = []
    for i in range(len(original_pixels)):
        if skeleton_pixels[i][1] == 235:
            overlay_image.append(original_pixels[i])
        else:
            overlay_image.append((0, 0, 0, 255))  
    
    original_rgba.putdata(overlay_image)    
    overlay_file = BytesIO()
    original_rgba.save(overlay_file, format="PNG")
    overlay_base64 = base64.b64encode(overlay_file.getvalue())  

    return original_rgba, overlay_base64

def visualize(overlay_image):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    ax.imshow(overlay_image, cmap=plt.cm.gray)
    ax.axis('off')
    fig.tight_layout(); plt.show() 