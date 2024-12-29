import os
from PIL import Image
import numpy as np
from numpy.fft import fft2, ifft2

def process_images():
    images_folder = os.path.join(os.getcwd(), 'coding B1', 'b1-codingproject-mtholiday2024', 'images')
    image_list = []
    
    for file in os.listdir(images_folder):
        if file.endswith('.JPEG'):
            path = os.path.join(images_folder, file)
            with Image.open(path) as img:
                bw_img = img.convert('L')
                img_array = np.array(bw_img)
                image_list.append(img_array)
    
    return image_list

images = process_images()
if len(images) == 5:
    print("Successfully processed 5 images")
else:
    print(f"Expected 5 images, but found {len(images)}")

for i, img_array in enumerate(images):
    img = Image.fromarray(img_array.astype(np.uint8))
    img.show()
    
def fft_convolve(images, kernel):
    convolved_images = []
    for img in images:
        # Determine the size for padding to avoid circular convolution
        pad_height = img.shape[0] + kernel.shape[0] - 1
        pad_width = img.shape[1] + kernel.shape[1] - 1

        # Pad kernel
        kernel_padded = np.pad(kernel, 
                   ((0, pad_height - kernel.shape[0]),
                    (0, pad_width - kernel.shape[1])), 
                   'constant')
        kernel_fft = fft2(kernel_padded)

        # Pad image
        img_padded = np.pad(img,
                ((0, pad_height - img.shape[0]),
                 (0, pad_width - img.shape[1])),
                'constant')
        img_fft = fft2(img_padded)
        convolved_fft = img_fft * kernel_fft
        convolved = np.real(ifft2(convolved_fft))
        # Crop to original image size
        #convolved = convolved[:images[0].shape[0], :images[0].shape[1]]
        convolved_images.append(convolved)

    return convolved_images


edge_kernel = np.array([
    [-1, 0, 0, 0, 1],
    [-2, 0, 0, 0, 2],
    [-4, 0, 0, 0, 4],
    [-2, 0, 0, 0, 2],
    [-1, 0, 0, 0, 1]
])

convolved_images = fft_convolve(images, edge_kernel)
for i, convolved in enumerate(convolved_images):
    img = Image.fromarray(convolved.astype(np.uint8))
    img.show()