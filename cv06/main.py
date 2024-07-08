import matplotlib.pyplot as plt
import numpy as np
import cv2


def getDFT(src):
    """
    get spectrum from DFT of image
    1. element = low frequencies in corners
    2. element = low frequencies in the middle
    """
    gray = cv2.imread(src,cv2.IMREAD_GRAYSCALE)
    fft2 = np.fft.fft2(gray)
    fft2_shifted = np.fft.fftshift(fft2)
    return fft2,fft2_shifted

def convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    kernel_flipped = np.flipud(np.fliplr(kernel))
    padded_image = np.pad(image, ((kernel_height//2, kernel_height//2), 
                                  (kernel_width//2, kernel_width//2)), mode='constant')
    result = np.zeros_like(image)
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            result[i, j] = np.sum(region * kernel_flipped)
    
    return result



def shiftArrs(arr_in):
    arr = []
    for array in arr_in:
        fft2 = np.fft.fft2(array)  # Compute FFT for each array
        fft2_shifted = np.fft.fftshift(fft2)
        arr.append(fft2_shifted)
    return arr

def getConvolutionKernel(detection_type):
    match detection_type:
        case "Laplace":
            return [np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]])]
        case "Sobel":
            return [
            np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]),
            np.array([[0, 1, 2],[-1, 0, 1],[-2, -1, 0]]),
            np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]),
            np.array([[-2, -1, 0],[-1, 0, 1],[0, 1, 2]]),
            np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]),
            np.array([[0, -1, -2],[1, 0, -1],[2, 1, 0]]),
            np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]),
            np.array([[2, 1, 0],[1, 0, -1],[0, -1, -2]])
        ]
        case "Kirsche":
            return [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]) 
        ]
        case _:
            return [np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])]
    
def edgeDetection(src, detection_type):
    image = cv2.imread(src,cv2.IMREAD_GRAYSCALE)
    kernel = getConvolutionKernel(detection_type)
    image_float = image.astype(np.float32)
    edges = np.max([convolution(image_float, k) for k in kernel],axis = 0)
    #edges = np.clip(edges, 0, 255)
    #edges_uint8 = edges.astype(np.uint8)
    return [image,edges]
    


def drawFilter(image_arr,draw_cmap, label_3):
    """
    draw filtered spectrums and filtered image
    coloring based on chosen cmap
    """
    shifted_arr = shiftArrs(image_arr)
    columns = 2
    rows = 2
    counter = 1
    for i,image in enumerate(image_arr):
        if counter > 1:
            plt.subplot(rows,columns,counter), 
            plt.imshow(image,cmap=draw_cmap)
            if counter==3:
                plt.title(label_3)
            else:
                plt.title('Image'),
            plt.xticks([]), plt.yticks([])
            plt.colorbar()
            
        else:
            plt.subplot(rows,columns,counter), 
            plt.imshow(image,cmap="gray")
            plt.title('Image'),
            plt.xticks([]), plt.yticks([])
        counter+=1
        plt.subplot(rows,columns,counter)
        plt.imshow(np.log(1+np.abs(shifted_arr[i])),cmap=draw_cmap)
        plt.title('Spectrum'),
        plt.xticks([]), plt.yticks([])
        if counter > 1:
            plt.colorbar()
        counter+=1
    plt.show()

def batch(image,detectors,draw_cmap):
    for detector in detectors:
        drawFilter(edgeDetection(image,detector),draw_cmap,detector)


if __name__=="__main__":
    images = ["./data/cv05_PSS.bmp","./data/cv04c_robotC.bmp"]
    batch(images[1],["Laplace","Sobel","Kirsche","Knespl"],"jet")
    