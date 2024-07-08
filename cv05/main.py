import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def averaging(src,kernel_size):
    image = cv2.imread(src,cv2.IMREAD_GRAYSCALE)
    result = np.zeros_like(image, dtype=np.float32)
    padding = kernel_size // 2
    
    for i in range(padding, image.shape[0] - padding):
        for j in range(padding, image.shape[1] - padding):
            neighbors = getNeigbors(image,padding,i,j,0)
            result[i, j] = np.mean(neighbors)/255
    return [image, result]

def getNeigbors(image,padding, i, j, type):
    xoff = 0
    yoff = 0
    match type:
        case 0:
            xoff=0
            yoff=0    
        case 1:
            xoff=0
            yoff=1
        case 2:
            xoff=1
            yoff=1
        case 3:
            xoff=1
            yoff=0
        case 4:
            xoff=1
            yoff=-1
        case 5:
            xoff=0
            yoff=-1
        case 6:
            xoff=-1
            yoff=-1
        case 7:
            xoff=-1
            yoff=0
        case 8:
            xoff=-1
            yoff=-1
    try:
        return image[i - padding + xoff:i + padding + 1 + xoff, j - padding + yoff:j + padding + 1 + yoff]
    except:
        return None

def rotating_mask_filter(src, kernel_size):
    image = cv2.imread(src,cv2.IMREAD_GRAYSCALE)
    result = np.zeros_like(image)
    padding = kernel_size // 2
    

    for i in range(padding, image.shape[0] - padding):
        for j in range(padding, image.shape[1] - padding):
            min_variance = float('inf')
            for rotation in range(9):
                neigbours = getNeigbors(image,padding,i,j, rotation)
                if neigbours is not None and neigbours.size > 0:
                    variance = np.var(neigbours)      
                    if variance < min_variance:
                        min_variance = variance
                        result[i, j] = np.mean(neigbours)
    return [image, result]

def median_filter(src, kernel_size):
    image = cv2.imread(src,cv2.IMREAD_GRAYSCALE)
    result = np.zeros_like(image)
    padding = kernel_size // 2
    
    for i in range(padding, image.shape[0] - padding):
        for j in range(padding, image.shape[1] - padding):
            neighbors = image[i - padding:i + padding + 1, j - padding:j + padding + 1]
            result[i, j] = np.median(neighbors)
    
    return [image,result]

def shiftArrs(arr_in):
    arr = []
    for array in arr_in:
        fft2 = np.fft.fft2(array)  # Compute FFT for each array
        fft2_shifted = np.fft.fftshift(fft2)
        arr.append(fft2_shifted)
    return arr


def drawFilter(image_arr,draw_cmap):
    """
    draw filtered spectrums and filtered image
    coloring based on chosen cmap
    """
    shifted_arr = shiftArrs(image_arr)
    columns = 2
    rows = 2
    counter = 1
    for i,image in enumerate(image_arr):
        plt.subplot(rows,columns,counter)
        plt.imshow(image,cmap="gray")
        plt.title('Image'),
        plt.xticks([]), plt.yticks([])
        counter+=1

        plt.subplot(rows,columns,counter)
        plt.imshow(np.log(1+np.abs(shifted_arr[i])),cmap=draw_cmap)
        plt.title('Spectrum'),
        plt.xticks([]), plt.yticks([])
        plt.colorbar()
        counter+=1
    plt.show()


if __name__ == "__main__":
    images = ["./data/cv05_PSS.bmp","./data/cv05_robotS.bmp"]
    drawFilter(averaging(images[1],3),"jet")
    drawFilter(rotating_mask_filter(images[1],3), "jet")
    drawFilter(median_filter(images[1],3),"jet")

    



