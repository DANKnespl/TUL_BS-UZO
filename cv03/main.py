"""
Script for image rotation and spinning
"""
import cv2
import numpy as np

def display_images(image_names, img_data):
    """
    draw image windows
    """
    for i, name in enumerate(image_names):
        cv2.imshow(name, img_data[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def batch_rotate(source, rotations):
    """
    rotate single image by different degrees
    """
    data = []
    names = []
    for _, r in enumerate(rotations):
        data.append(rotate_img(source, r))
        names.append("Rotace o " + str(r % 360) + " stupnu")
    display_images(names, data)


def thats_a_good_trick(source, start_angle, step):
    """
    Function to generate 360 rotated states of image
    and draw them iteratively in a window
    source = path to image to be spun
    start_angle = first rotation frame
    step = step of rotation
    """
    data = []
    #generate rotation frames
    for r in range(0, 360):
        data.append(rotate_img(source, r))
    i = start_angle
    #draw rotation frames infintely
    while True:
        cv2.imshow("rotator", data[i % 360])
        i += step
        key = cv2.waitKey(10*step) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

def get_rotation_matrix(height_diff, width_diff, angle_rad, center):
    """
    generates 2x3 transformation matrix
    1. rotation around center (x,y) by angle_rad in radians counterclockwise
    2. resizing and moving by height_diff and width_diff
    """
    #matrix template without values
    rotation_matrix = np.array([
        [np.cos(-angle_rad),
         -np.sin(-angle_rad), 
         center[0] *(1 - np.cos(-angle_rad)) + center[1] * np.sin(-angle_rad) + width_diff // 2],
        [np.sin(-angle_rad),
          np.cos(-angle_rad), 
          center[1] * (1 - np.cos(-angle_rad)) - center[0] * np.sin(-angle_rad) + height_diff//2]])

    return rotation_matrix

def rotate_img(image_path, angle):
    """
    function to quickly rotate image by angle in degrees counterclockwise
    """
    #setting basic variables
    original_image = cv2.imread(image_path)
    height, width = original_image.shape[:2]
    angle_rad = np.radians(angle)

    #rotation matrix setup + creation
    rotated_width = int(np.ceil(width * abs(np.cos(angle_rad)) + height * abs(np.sin(angle_rad))))
    rotated_height = int(np.ceil(width * abs(np.sin(angle_rad)) + height * abs(np.cos(angle_rad))))    
    rotation_matrix = get_rotation_matrix(rotated_height - height, rotated_width - width, angle_rad, [width // 2, height // 2])

    #rotating weeee
    rotated_image = cv2.warpAffine(original_image, rotation_matrix, (rotated_width, rotated_height))

    return rotated_image

def meh_rotate(src, angle, background_color=(0, 0, 0)):
    """
    function to slowly rotate image by angle in degrees counterclockwise
    """
    angle_rad = np.radians(angle)
    original_image = cv2.imread(src) 
    height, width = original_image.shape[:2]
    
    rotated_width = int(np.ceil(width * np.abs(np.cos(angle_rad)) + height * np.abs(np.sin(angle_rad))))
    rotated_height = int(np.ceil(width * np.abs(np.sin(angle_rad)) + height * np.abs(np.cos(angle_rad))))

    #2x3 -> 3x3 rotation matrix
    rotation_matrix = get_rotation_matrix(rotated_height - height, rotated_width - width, angle_rad, [width // 2, height // 2])
    rotation_matrix = np.concatenate((rotation_matrix, np.array([[0, 0, 1]])), axis=0)
    
    rotated_image = np.full((rotated_height, rotated_width, 3), background_color, dtype=np.uint8)

    for y in range(rotated_height):
        for x in range(rotated_width):
            original_pixel = np.dot(np.linalg.inv(rotation_matrix), [x, y, 1])

            original_x = original_pixel[0]
            original_y = original_pixel[1]


            #pixely na které existuje zpětné zobrazení
            if 0 <= original_x < width - 1 and 0 <= original_y < height - 1:
                #určení hodnoty zobrazovaného pixelu (bilinearní transformace)
                x1, y1 = int(np.floor(original_x)), int(np.floor(original_y))
                x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)

                weight_x2 = original_x - x1
                weight_x1 = 1 - weight_x2
                weight_y2 = original_y - y1
                weight_y1 = 1 - weight_y2

                pixel_value = (
                    weight_x1 * weight_y1 * original_image[y1, x1] +
                    weight_x2 * weight_y1 * original_image[y1, x2] +
                    weight_x1 * weight_y2 * original_image[y2, x1] +
                    weight_x2 * weight_y2 * original_image[y2, x2]
                )

                rotated_image[y, x] = pixel_value.astype(np.uint8)
    cv2.imshow("Rotated Image", rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    images = ["./data/cv03_robot.bmp", "./data/cv02_vzor_hrnecek.bmp","./data/uzo_cv02_im04.jpg", "./data/photo.jpg", "./data/WeirdChamp.png"]
    meh_rotate(images[0],273)
    meh_rotate(images[0],124)
    
    batch_rotate(images[2], [45,-75,13,0,500]) 
    thats_a_good_trick(images[3], 30, 1)
