from PIL import Image
import numpy as np
from scipy.fft import fft2, ifft2

def image_to_numpy(image_path):
    
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    return np.array(image)

def numpy_to_image(np_array):
       
    image = Image.fromarray(np_array)
    image = image.convert('L')  # Ensure the image is in grayscale
    image.show()