import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy import ndimage
import imageio

def imread_gray(filename):
    """Read grayscale image."""
    image = imageio.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return image
    
def gauss(x, sigma):
    return 1.0 / np.sqrt(2.0 * np.pi) / sigma * np.exp(- x**2 / 2.0 / sigma**2)
def gaussdx(x, sigma):
    return - 1.0 / np.sqrt(2.0 * np.pi) / sigma**3 * x * np.exp(- x**2 / 2.0 / sigma**2)
    
def convolve_with_two(image, kernel1, kernel2):
    """Apply two filters, one after the other."""
    image = ndimage.convolve(image, kernel1)
    image = ndimage.convolve(image, kernel2)   
    return image
    
def gauss_derivs(image, sigma):
    kernel_radius = int(3.0 * sigma)
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]  
    G = gauss(x, sigma)
    D = gaussdx(x, sigma)
    image_dx = convolve_with_two(image, D, G.T)
    image_dy = convolve_with_two(image, G, D.T)
    return image_dx, image_dy