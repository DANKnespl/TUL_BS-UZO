import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def find_areas(image):
    def merge_arrays(array_of_arrays):
        edges = defaultdict(set)
    
        # Constructing edges between overlapping arrays
        for i in range(len(array_of_arrays)):
            for j in range(i + 1, len(array_of_arrays)):
                if set(array_of_arrays[i]) & set(array_of_arrays[j]):
                    edges[i].add(j)
                    edges[j].add(i)
        
        # Merging connected components based on edges
        merged_arrays = []
        visited = set()
        for i in range(len(array_of_arrays)):
            if i not in visited:
                component = set()
                stack = [i]
                while stack:
                    node = stack.pop()
                    visited.add(node)
                    component.add(node)
                    stack.extend(edges[node] - visited)
                merged_arrays.append(list(component))

        # Merging the actual arrays
        final_merged_arrays = []
        for component in merged_arrays:
            merged_array = np.concatenate([array_of_arrays[i] for i in component])
            final_merged_arrays.append(np.unique(merged_array))

        return final_merged_arrays

    height, width = image.shape[:2]
    areas = np.zeros_like(image)
    area_types = np.uint8(1)
    conflicts = []

    for y in range(height):
        for x in range(width):
            neighbours = []
            if image[y, x] == 255:
                if y - 1 >= 0 and x - 1 >= 0 and areas[y - 1, x - 1] != 0:
                    neighbours.append(areas[y - 1, x - 1])
                if y - 1 >= 0 and areas[y - 1, x] != 0:
                    neighbours.append(areas[y - 1, x])
                if y - 1 >= 0 and x + 1 < width and areas[y - 1, x + 1] != 0:
                    neighbours.append(areas[y - 1, x + 1])
                if x - 1 >= 0 and areas[y, x - 1] != 0:
                    neighbours.append(areas[y, x - 1])
                neighbours = list(set(neighbours))

                if len(neighbours) > 1:
                    areas[y, x] = neighbours[0]
                    conflicts.append(neighbours)
                elif len(neighbours) == 1:
                    areas[y, x] = neighbours[0]
                else:
                    areas[y, x] = area_types
                    area_types += 1

    # Merge conflicting areas
    conflicts = merge_arrays(conflicts)
    for y in range(height):
        for x in range(width):
            if areas[y, x] != 0:
                for _, arr in enumerate(conflicts):
                    if areas[y, x] in arr:
                        areas[y, x] = np.min(arr)

    return areas

def segment_image(src, threshold,threshold_top, kernel_size):
    def cancel_noise(img, kernel_size):
        # Apply morphological operations for noise cancellation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = cv2.erode(img, kernel, iterations=1) 
        img = cv2.dilate(img, kernel, iterations=1)
        return img

    image = cv2.imread(src)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tophat_image = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,np.ones((13,13),np.uint8))
    # Perform thresholding to segment the image
    segmented_image = np.where(gray > threshold, 255, 0).astype(np.uint8)
    segmented_tophat = np.where(tophat_image > threshold_top, 255, 0).astype(np.uint8)
    return cancel_noise(segmented_tophat,3), image, segmented_image, gray, tophat_image

def create_centroids(areas, image, plus_size, rice_size):
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
    rice_count = 0
    for _, val in moment_data.items():
        x_coord = int(val[2] / val[0])  # Calculate centroid x-coordinate
        y_coord = int(val[1] / val[0])  # Calculate centroid y-coordinate
        if val[0]>rice_size:
            image = add_plus(image, y_coord, x_coord, plus_size)
            rice_count+=1
    return image, rice_count

def draw_images(centroid_data, image_data_array):
    original_with_points = centroid_data[0]
    original = image_data_array[3]
    top_hat = image_data_array[4]
    segmented_original = image_data_array[2]
    segmented_top_hat  = image_data_array[0]
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(original.ravel(), bins=256, range=[0,256])
    plt.title('Original histogram')
    
    plt.subplot(2, 2, 2)
    plt.hist(top_hat.ravel(), bins=256, range=[0,256])
    plt.title('Top-hat histogram')
    plt.subplot(2, 3, 4)
    plt.imshow(segmented_original, cmap="gray")
    plt.title("Original segmented")
    plt.axis("off")
    
    plt.subplot(2, 3, 5)
    plt.imshow(segmented_top_hat,cmap="gray")
    plt.title("Top-hat segmented")
    plt.axis("off")

    plt.subplot(2,3,6)
    plt.imshow(cv2.cvtColor(original_with_points,cv2.COLOR_BGR2RGB))
    plt.title(str(centroid_data[1])+" grains of rice detected")
    plt.axis("off")
    plt.show()

if __name__=="__main__":
    images = ["./data/cv09_rice.bmp","./data/cv10_mince.jpg"]
    #+-135 original
    #+-50 tophat
    plusSize = 5
    rice_size = 0
    im1 = segment_image(images[0],135,50,5)
    draw_images(create_centroids(find_areas(im1[0]),im1[1].copy(),plusSize,rice_size),im1)
