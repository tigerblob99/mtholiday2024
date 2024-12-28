import os
import numpy as np
from image_utils import image_to_numpy
from scipy.fft import fft2, ifft2


# Define the path to the images folder
images_folder = '/Users/lizhuoxuan/Documents/coding B1/b1-codingproject-mtholiday2024/images'

# Initialize a list to store the numpy arrays
numpy_arrays = []

# Iterate over each file in the images folder
for filename in os.listdir(images_folder):
    # Construct the full file path
    file_path = os.path.join(images_folder, filename)
            
    # Convert the image to a numpy array
    image_array = image_to_numpy(file_path)
        
    # Append the numpy array to the list
    numpy_arrays.append(image_array)


def apply_fft_convolution(image_array, kernel):
    
    # Pad the kernel to the size of the image
    pad_height = image_array.shape[0] - kernel.shape[0]
    pad_width = image_array.shape[1] - kernel.shape[1]
    kernel_padded = np.pad(kernel, ((0, pad_height), (0, pad_width)), mode='constant')

    # Apply FFT to both the image and the kernel
    image_fft = fft2(image_array)
    kernel_fft = fft2(kernel_padded)

    # Perform element-wise multiplication in the frequency domain
    convolved_fft = image_fft * kernel_fft

    # Apply the inverse FFT to get the convolved image
    convolved_image = ifft2(convolved_fft).real
    return convolved_image