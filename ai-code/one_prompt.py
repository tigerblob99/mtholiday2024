import os
import numpy as np
import cv2

import matplotlib.pyplot as plt

def apply_fft_edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift) + 1)

    # Create a high-pass filter
    rows, cols = image.shape
    crow, ccol = rows//2, cols//2
    radius = 30
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)

    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize and convert to uint8
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = img_back.astype(np.uint8)

    edges = cv2.Canny(img_back, 100, 200)
    return edges

input_folder = 'images'
output_folder = 'edges'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        input_path = os.path.join(input_folder, filename)
        edges = apply_fft_edge_detection(input_path)
        output_path = os.path.join(output_folder, f'edges_{filename}')
        cv2.imwrite(output_path, edges)
        print(f'Processed {filename} and saved edges to {output_path}')

# Display one of the edge images
sample_edge = cv2.imread(os.path.join(output_folder, f'edges_{os.listdir(output_folder)[0]}'), cv2.IMREAD_GRAYSCALE)
plt.imshow(sample_edge, cmap='gray')
plt.title('Edge Detection Result')
plt.axis('off')
plt.show()