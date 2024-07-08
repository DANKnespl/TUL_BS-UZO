import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def segment_image(src, threshold, kernel_size):
    def close_holes(img, kernel_size):
        # Apply morphological operations for noise cancellation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1) 
        return img
    
    image = cv2.imread(src)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    segmented_image = np.where(gray > threshold, 0, 255).astype(np.uint8)
    return close_holes(segmented_image,kernel_size),image,gray

def create_centroids(areas, image, plus_size):
    global gl_markers_count
    global gl_image_with_points
    def get_area_sizes(areas):
        # Count occurrences of each area number
        counts = {}
        for sublist in areas:
            for num in sublist:
                counts[num] = counts.get(num, 0) + 1
        return counts

    def get_moments(image):
        height, width = image.shape[:2]
        areas = get_area_sizes(image)
        moments = {}
        for key, val in areas.items():
            moments[key] = [val, 0, 0]
        moments.pop(0)  # Remove background
        for y in range(height):
            for x in range(width):
                if image[y, x] != 0:
                    moments[image[y, x]][1] += y
                    moments[image[y, x]][2] += x
        return moments

    def add_plus(image, y_coord, x_coord, plus_size):
        # Add plus sign around centroids
        points = [[0, 0]]
        height, width = image.shape[:2]
        for i in range(1, plus_size + 1):
            points.extend([[-i, 0], [i, 0], [0, -i], [0, i]])

        for point in points:
            if 0<= y_coord + point[0] < height and 0<= x_coord + point[1] < width:
                image[y_coord + point[0], x_coord + point[1]] = [0.0, 0.0, 255.0]
        return image

    moment_data = get_moments(areas)
    for _, val in moment_data.items():
        x_coord = int(val[2] / val[0])  # Calculate centroid x-coordinate
        y_coord = int(val[1] / val[0])  # Calculate centroid y-coordinate
        image = add_plus(image, y_coord, x_coord, plus_size)
    
    gl_markers_count = len(moment_data)
    gl_image_with_points = image

def draw_images():
    plt.figure(figsize=(10, 8))
    plt.subplot(2,3,1)
    plt.imshow(gl_segmented)
    plt.title("sure background")
    plt.axis("off")
    plt.subplot(2,3,2)
    plt.imshow(gl_dist)
    plt.title("distance map")
    plt.axis("off")
    plt.subplot(2,3,3)
    plt.imshow(gl_centers)
    plt.title("sure foreground")
    plt.axis("off")
    
    
    plt.subplot(2,2,3)
    plt.imshow(gl_markers)
    plt.colorbar()
    plt.title("areas")
    plt.axis("off")
    
    plt.subplot(2,2,4)    
    plt.imshow(cv2.cvtColor(gl_image_with_points,cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("result")
    plt.show()

def watershedAndColoring(img_data):
    global gl_segmented
    global gl_dist
    global gl_markers
    global gl_centers
    """
    https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
    """
    kernel = np.ones((3,3),np.uint8)
    # sure background area
    
    sure_bg = cv2.dilate(img_data[0],kernel,iterations=0)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(img_data[0],cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img_data[1],markers)
    markers = np.where(markers<=1, 0, markers).astype(np.uint8)
    
    
    gl_dist = dist_transform
    gl_segmented = sure_bg
    gl_centers = sure_fg
    gl_markers = markers

    return markers


if __name__=="__main__":
    images = ["./data/cv09_rice.bmp","./data/cv10_mince.jpg"]
    plusSize = 15
    im1 = segment_image(images[1],135,6)
    create_centroids(watershedAndColoring(im1),im1[1].copy(),plusSize)
    draw_images()
