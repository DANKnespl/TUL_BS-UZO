import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def find_areas(image):
    def merge_arrays(array_of_arrays):
        edges = defaultdict(set)
        for i in range(len(array_of_arrays)):
            for j in range(i + 1, len(array_of_arrays)):
                if set(array_of_arrays[i]) & set(array_of_arrays[j]):
                    edges[i].add(j)
                    edges[j].add(i)
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
            neighbors = []
            if image[y, x] == 255:
                if y - 1 >= 0 and x - 1 >= 0 and areas[y - 1, x - 1] != 0:
                    neighbors.append(areas[y - 1, x - 1])
                if y - 1 >= 0 and areas[y - 1, x] != 0:
                    neighbors.append(areas[y - 1, x])
                if y - 1 >= 0 and x + 1 < width and areas[y - 1, x + 1] != 0:
                    neighbors.append(areas[y - 1, x + 1])
                if x - 1 >= 0 and areas[y, x - 1] != 0:
                    neighbors.append(areas[y, x - 1])
                neighbors = list(set(neighbors))

                if len(neighbors) > 1:
                    areas[y, x] = neighbors[0]
                    conflicts.append(neighbors)
                elif len(neighbors) == 1:
                    areas[y, x] = neighbors[0]
                else:
                    areas[y, x] = area_types
                    area_types += 1

    conflicts = merge_arrays(conflicts)

    for y in range(height):
        for x in range(width):
            if areas[y, x] != 0:
                for _, arr in enumerate(conflicts):
                    if areas[y, x] in arr:
                        areas[y, x] = np.min(arr)

    return areas

def segment_image(src):
    def get_threshold(array):
        # Calculate threshold for segmentation
        threshold = np.where((array[1:-1] < array[0:-2]) * (array[1:-1] < array[2:]))[0]
        threshold = threshold[0]  # Extracting the threshold value
        return threshold
    
    def morph(img):
        # Apply morphological operations for noise reduction and hole filling
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)  # Fill holes
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)  # Reduce noise
        img = cv2.dilate(img, kernel, iterations=1)
        return img
    
    image = cv2.imread(src)
    image = image.astype(np.float32)
    blue, green, red = cv2.split(image)
    denominator = red + green + blue
    epsilon = 1e-5
    values = np.floor(np.clip((green * 255) / (denominator + epsilon), 0, 255))  # Calculate values
    threshold = get_threshold(cv2.calcHist([values], [0], None, [256], [0, 256]))  # Calculate threshold
    g_fun = np.where(values < threshold, 255, 0).astype(np.uint8)  # Thresholding
    return morph(g_fun), image  # Return segmented image and original image

def decode_money(areas, image, plus_size):
    def get_area_sizes(areas):
        # Flatten the array of arrays into a single list
        flat_list = [item for sublist in areas for item in sublist]
        counts = {}
        for num in flat_list:
            if num in counts:
                counts[num] += 1
            else:
                counts[num] = 1
        return counts
    
    def get_moments(image):
        height, width = image.shape[:2]
        areas = get_area_sizes(image)
        moments = {}
        for key, val in areas.items():
            moments[key] = [val, 0, 0]
        moments.pop(0)
        for y in range(height):
            for x in range(width):
                if image[y, x] != 0:
                    moments[image[y, x]][1] += y
                    moments[image[y, x]][2] += x
        return moments
    
    def add_plus(image, y_coord, x_coord, type, plus_size):
        points = [[0, 0]]
        color = []
        if type == "5Kč":
            color = [0.0, 255.0, 255.0]
        else:
            color = [255.0, 255.0, 0.0]
        for i in range(1, plus_size + 1):
            points.extend([[-i, 0], [i, 0], [0, -i], [0, i]])
        for point in points:
            image[y_coord + point[0], x_coord + point[1]] = color
        return image

    moment_data = get_moments(areas)
    print("Nalezeno: ")
    total_value = 0
    for _, val in moment_data.items():
        x_coord = int(val[2] / val[0])
        y_coord = int(val[1] / val[0])
        money_type = ""
        if val[0] >= 4000:
            money_type = "5Kč"
            total_value += 5
        else:
            money_type = "1Kč"
            total_value += 1
        print("   " + money_type + " s těžištěm v: x=" + str(x_coord) + ", y=" + str(y_coord))
        image = add_plus(image, y_coord, x_coord, money_type, plus_size)
    print("Jsem bohatý, mám " + str(total_value) + "Kč!")
    return image

def draw_images(image_original, image_segmented):
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB).astype(np.uint8))
    plt.title('Original Image')
    
    plt.subplot(2, 1, 2)
    plt.imshow(image_segmented, cmap="gray")
    plt.title('Segmented Image')
    
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    images = ["./data/cv07_segmentace.bmp","./data/cv07_barveni.bmp"]
    #threshold = 97
    im = segment_image(images[0])
    img=decode_money(find_areas(im[0]),im[1],10)
    draw_images(img,im[0])


