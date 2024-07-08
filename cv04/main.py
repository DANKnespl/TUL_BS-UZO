"""
popis
"""
import MTFunctions as mt
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

def drawSpectum(spectrum,shifted,draw_cmap):
    """
    draw standard and shifted spectrums
    coloring based on chosen cmap
    """

    plt.subplot(121)
    plt.imshow(np.log(1 + np.abs(spectrum)),cmap=draw_cmap)
    plt.title("Spectrum")
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(np.log(1 + np.abs(shifted)),cmap=draw_cmap)
    plt.title("Shifted Spectrum")
    plt.colorbar()
    plt.show()

def drawFilter(spectrum_arr,shifted_arr,image_arr,draw_cmap, state):
    """
    draw filtered spectrums and filtered image
    coloring based on chosen cmap
    """
    columns = 0
    if state == 0:
        columns = 3
    if state == 1:
        #shifted
        columns = 2
    if state == 2:
        #inverted
        columns = 2
    rows = 2
    counter = 1
    window = 1
    for i,image in enumerate(image_arr):
        if state == 0 or state == 2:
            plt.subplot(rows,columns,counter)
            plt.imshow(np.log(1 + np.abs(spectrum_arr[i])), cmap=draw_cmap)
            if counter <= columns:
                plt.title('Filtered Spectrum'),
            plt.xticks([]), plt.yticks([])
            counter+=1

        if state == 0 or state == 1:
            plt.subplot(rows,columns,counter)
            plt.imshow(np.log(1+np.abs(shifted_arr[i])),cmap=draw_cmap)
            if counter <= columns:
                plt.title('Filtered Spectrum'),
            plt.xticks([]), plt.yticks([])
            counter+=1

        plt.subplot(rows,columns,counter), plt.imshow(image,cmap="gray")
        if counter <= columns:
            plt.title('Filtered Image'),
        plt.xticks([]), plt.yticks([])
        
        
        if i==len(image_arr)-1:
            break
        if counter == columns*2:
            plt.show()
            counter=0
            window += 1
        counter+=1
        
    plt.show()

def filter(spectrum, mask_src):
    """
    creates filtered spectrums and image
    from spectrum and img source of mask
    """
    
    mask = cv2.imread(mask_src,cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask != 0, 1, mask)
    shifted_filtered_spectrum = spectrum * mask
    filtered_spectrum = np.fft.ifftshift(shifted_filtered_spectrum)
    filtered_image = np.abs(np.fft.ifft2(filtered_spectrum))

    return filtered_spectrum,shifted_filtered_spectrum,filtered_image

def multiFilter(shifted,masks,draw_cmap,columns_to_draw):
    """
    batch filtering single shifted spectrum using different masks
    """
    f_list = []
    sf_list = []
    im_list = []
    for mask in masks:
        data = filter(shifted,mask)
        f_list.append(data[0])
        sf_list.append(data[1])
        im_list.append(data[2])
    drawFilter(f_list,sf_list,im_list,draw_cmap,columns_to_draw)


if __name__ == '__main__':
    #used_cmap = "hot"
    draw_mask = 1
    used_cmap = "jet"
    
    #MT4
    gamma1=mt.gammaCorection("./data/cv04_f01.bmp","./data/cv04_e01.bmp")
    gamma2=mt.gammaCorection("./data/cv04_f02.bmp","./data/cv04_e02.bmp")
    eq=mt.histogramEqualisation("./data/cv04_rentgen.bmp",0,255)
    mt.output(gamma1,gamma2,eq)
    
    #UZO4
    original, shifted = getDFT("./data/cv04c_robotC.bmp")
    drawSpectum(original,shifted,used_cmap)
    multiFilter(shifted,["./data/cv04c_filtDP.bmp","./data/cv04c_filtDP1.bmp","./data/cv04c_filtHP.bmp","./data/cv04c_filtHP1.bmp"],used_cmap,draw_mask)
    