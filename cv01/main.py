import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def drawCv2(matrix, images):
    canvas_size = 900  # Adjust the size as needed
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    rows = len(matrix)
    image_size = canvas_size // rows  # Calculate the size of each grid cell
    matcounter = 0
    for mat in enumerate(matrix):
        indcounter = 0
        for ind in enumerate(mat[1]):
            image = cv2.imread(images[ind[1]])
            image = cv2.resize(image, (image_size, image_size))
            x_start, y_start = indcounter * image_size, matcounter * image_size
            x_end, y_end = x_start + image_size, y_start + image_size
            canvas[y_start:y_end, x_start:x_end, :] = image
            indcounter += 1
        matcounter += 1
    cv2.imshow('Grid of Images', canvas)


def drawPlt(matrix, images):
    plt.figure()
    rows = len(images)
    matcounter = 0
    for mat in enumerate(matrix):
        indcounter = 0
        for ind in enumerate(mat[1]):
            image = cv2.imread(images[ind[1]])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.subplot(rows, rows, matcounter*rows+indcounter+1)
            plt.imshow(image)
            plt.axis("off")
            indcounter += 1
        matcounter += 1
    plt.show()


def getHistogram(imgPath):
    image_bgr = cv2.imread(imgPath)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist


def getMatrix(images):
    indexMatrix = []
    for _, im1 in enumerate(images):
        histG = getHistogram(im1)
        lineGray = []
        for _, im2 in enumerate(images):
            histG2 = getHistogram(im2)
            lineGray.append(cv2.compareHist(
                histG, histG2, cv2.HISTCMP_INTERSECT))
        lineGrayIndex = np.array(
            sorted(range(len(lineGray)), key=lambda k: lineGray[k]))
        print(lineGrayIndex[::-1]+1)
        indexMatrix.append(lineGrayIndex[::-1])
    return indexMatrix


if __name__ == '__main__':
    #images = ["data/uzo_cv02_im01.jpg", "data/uzo_cv02_im02.jpg", "data/uzo_cv02_im03.jpg", "data/uzo_cv02_im04.jpg",
    #          "data/uzo_cv02_im05.jpg", "data/uzo_cv02_im06.jpg", "data/uzo_cv02_im07.jpg", "data/uzo_cv02_im08.jpg", "data/uzo_cv02_im09.jpg"]
    images = ["data/uzo_cv02_im01.jpg", "data/uzo_cv02_im02.jpg",
              "data/uzo_cv02_im05.jpg", "data/uzo_cv02_im04.jpg"]
    matrixGray = getMatrix(images)
    drawCv2(matrixGray, images)
    drawPlt(matrixGray, images)
