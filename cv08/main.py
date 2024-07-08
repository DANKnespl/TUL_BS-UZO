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

def segment_image(src, threshold, kernel_size, seg_type):
    def cancel_noise(img, kernel_size):
        # Apply morphological operations for noise cancellation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = cv2.erode(img, kernel, iterations=1) 
        img = cv2.dilate(img, kernel, iterations=1)
        return img

    # Read the image and convert it to grayscale
    image = cv2.imread(src)
    image = image.astype(np.float32)
    ycrcb  = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    gray = None
    segmented_image_noise = None
    if seg_type==1:
        gray = ycrcb[:,:,1]
        segmented_image_noise = np.where(gray > threshold, 255, 0).astype(np.uint8)
    
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        segmented_image_noise = np.where(gray < threshold, 255, 0).astype(np.uint8)
    
    # Apply noise cancellation
    segmented_image = cancel_noise(segmented_image_noise, kernel_size)
    return segmented_image, image, segmented_image_noise

def create_centroids(areas, image, plus_size):
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
        for i in range(1, plus_size + 1):
            points.extend([[-i, 0], [i, 0], [0, -i], [0, i]])

        for point in points:
            image[y_coord + point[0], x_coord + point[1]] = [0.0, 255.0, 0.0]
        return image

    moment_data = get_moments(areas)
    for _, val in moment_data.items():
        x_coord = int(val[2] / val[0])  # Calculate centroid x-coordinate
        y_coord = int(val[1] / val[0])  # Calculate centroid y-coordinate
        image = add_plus(image, y_coord, x_coord, plus_size)
    return image

def draw_images(with_points, segmented, with_noise, original):
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(np.uint8))
    #plt.title('Original Image')
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    plt.imshow(with_noise, cmap="gray")
    #plt.title('Segmented Image')
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(segmented, cmap="gray")
    #plt.title('Morphed Image')
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(with_points, cv2.COLOR_BGR2RGB).astype(np.uint8))
    #plt.title('Image with centroids')
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    images = ["./data/cv08_im1.bmp","./data/cv08_im2.bmp"]
    plusSize = 5
    im1 = segment_image(images[0],95,5, 0)
    im2 = segment_image(images[1],80,5, 0)
    im3 = segment_image(images[1],5,1,  1)
    draw_images(create_centroids(find_areas(im1[0]),im1[1].copy(),plusSize),im1[0],im1[2],im1[1])
    draw_images(create_centroids(find_areas(im2[0]),im2[1].copy(),plusSize),im2[0],im2[2],im2[1])
    draw_images(create_centroids(find_areas(im3[0]),im3[1].copy(),plusSize),im3[0],im3[2],im3[1])
    
