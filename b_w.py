from PIL import Image
import numpy as np
from scipy.fft import fft2, ifft2

def image_to_numpy(image_path):
    """
    Convert an image to a numpy array.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    np.ndarray: The image as a numpy array.
    """
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    return np.array(image)