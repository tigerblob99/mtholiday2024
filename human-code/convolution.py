import os
import numpy as np
from image_utils import image_to_numpy
from scipy.fft import fft2, ifft2
from image_utils import numpy_to_image


# Define the path to the images folder
images_folder = '/Users/lizhuoxuan/Documents/coding B1/b1-codingproject-mtholiday2024/images'

# Initialize a list to store the numpy arrays
numpy_arrays = []

# Iterate over each file in the images folder
for filename in os.listdir(images_folder):
    if filename.endswith(('.png', '.jpg', '.JPEG')):  # Check for image file extensions
        # Construct the full file path
        file_path = os.path.join(images_folder, filename)
        
        # Debug: Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        # Convert the image to a numpy array
        image_array = image_to_numpy(file_path)
        
        if image_array is not None:
            # Append the numpy array to the list
            numpy_arrays.append(image_array)
            print(f"Successfully processed: {filename}")

print(f"Found {len(numpy_arrays)} images to process")
def apply_fft_convolution(image_array, kernel):
    # Get the dimensions of the image and kernel
    rows, cols = image_array.shape
    krows, kcols = kernel.shape

    # Compute the size of the output so convolution is valid
    out_rows = rows + krows - 1
    out_cols = cols + kcols - 1

    # Pad both image and kernel to match output dimensions
    padded_image = np.pad(image_array, ((0, out_rows - rows), (0, out_cols - cols)), mode='constant')
    padded_kernel = np.pad(kernel, ((0, out_rows - krows), (0, out_cols - kcols)), mode='constant')

    # Perform FFT on the padded image and kernel
    fft_image = fft2(padded_image)
    fft_kernel = fft2(padded_kernel)
    # Multiply in the frequency domain and apply inverse FFT
    result_freq = fft_image * fft_kernel
    result_spatial = np.real(ifft2(result_freq))

    return result_spatial

vertical_edge_kernel = np.array([
    [-1,  0,  0,  0,  1],
    [-2,  0,  0,  0,  2],
    [-4,  0,  0,  0,  4],
    [-2,  0,  0,  0,  2],
    [-1,  0,  0,  0,  1]
])

# ...existing code...
convolved_results = []
# First process all images
for arr in numpy_arrays:
    if arr is not None:
        result = apply_fft_convolution(arr, vertical_edge_kernel)
        convolved_results.append(result)

# Then display all results
for i, result in enumerate(convolved_results):
    if result is not None:
        print(f"Displaying image {i+1}")
        result_image = numpy_to_image(result)
        if result_image:
            result_image.show()  # Remove title parameter as it's not supported

