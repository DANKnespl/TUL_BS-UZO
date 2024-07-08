"""
popis
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogramEqualisation(img,q0,qk):
    ImgDataBGR=cv2.imread(img)
    ImgDataYCrCb=cv2.cvtColor(ImgDataBGR,cv2.COLOR_BGR2YCR_CB)
    ImgDataRGB=cv2.cvtColor(ImgDataBGR,cv2.COLOR_BGR2RGB)
    width=ImgDataBGR.shape[1]
    height=ImgDataBGR.shape[0]
    hist = cv2.calcHist([ImgDataYCrCb],[0],None,[256],[0,256])
    data=[[0 for i in range(width)] for j in range(height)]
    FH=hist.cumsum()
    IMGCONST=(qk-q0)/(width*height)
    for j in range(height):
        for i in range(width):
            p=ImgDataYCrCb[j][i][0]
            data[j][i]=[IMGCONST*FH[p]+q0,ImgDataYCrCb[j][i][1],ImgDataYCrCb[j][i][2]]
    data=np.uint8(data)
    dataHist=cv2.calcHist([data],[0],None,[256],[0,256])
    return ImgDataRGB,hist,cv2.cvtColor(data,cv2.COLOR_YCR_CB2RGB),dataHist

def gammaCorection(withEr,Er):
    withErDataBGR = cv2.imread(withEr)
    ErDataBGR = cv2.imread(Er)
    mask = (ErDataBGR != 0) & (~np.isnan(ErDataBGR))
    data = np.zeros_like(withErDataBGR)
    data[mask] = np.divide(withErDataBGR[mask].astype(float), ErDataBGR[mask].astype(float)) * 255
    return (cv2.cvtColor(withErDataBGR,cv2.COLOR_BGR2RGB),cv2.cvtColor(ErDataBGR,cv2.COLOR_BGR2RGB),cv2.cvtColor(np.uint8(data),cv2.COLOR_BGR2RGB))
    #width=withErDataBGR.shape[1]
    #height=withErDataBGR.shape[0]
    #withErDataYCrCb=cv2.cvtColor(withErDataBGR,cv2.COLOR_BGR2YCR_CB)
    #ErDataYCrCb=cv2.cvtColor(ErDataBGR,cv2.COLOR_BGR2YCR_CB)
    #ErDataGray=cv2.cvtColor(ErDataBGR,cv2.COLOR_BGR2GRAY)
    #data=[[0 for i in range(width)] for j in range(height)]
    #data2=[[0 for i in range(width)] for j in range(height)]
    #for j in range(height):
    #    for i in range(width):
    #        tmp=withErDataYCrCb[j][i]
    #        tmp2=ErDataYCrCb[j][i]
    #        tmp4=ErDataGray[j][i]
    #        tmp3=withErDataBGR[j][i]
    #        if tmp2[0]==0:
    #            y=0
    #        else:
    #            y=float(tmp[0])/float(tmp2[0])*255
    #        data[j][i]=[y,tmp[1],tmp[2]]
    #        data2[j][i]=[tmp3[2]/tmp4*255,tmp3[1]/tmp4*255,tmp3[0]/tmp4*255]
    #return (cv2.cvtColor(withErDataBGR,cv2.COLOR_BGR2RGB),cv2.cvtColor(np.uint8(data2),cv2.COLOR_BGR2RGB))
    #'''

def output(g1,g2,equal):
    plt.subplot(2,3,1)
    plt.imshow(g1[0])
    plt.title("S chybou")
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(2,3,2)
    plt.imshow(g1[1])
    plt.title("Chyba")
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(2,3,3)
    plt.imshow(g1[2])
    plt.title("Bez chyby")
    plt.xticks([]), plt.yticks([])
        
    plt.subplot(2,3,4)
    plt.imshow(g2[0])
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(2,3,5)
    plt.imshow(g2[1])
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(2,3,6)
    plt.imshow(g2[2])
    plt.xticks([]), plt.yticks([])


    plt.show()


    plt.subplot(2,2,1)
    plt.imshow(equal[0])
    plt.title("Nízký kontrast")
    plt.xticks([]), plt.yticks([])

    plt.subplot(2,2,2)
    plt.imshow(equal[2])
    plt.title("Vysoký kontrast")
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(2,2,3)
    plt.plot(equal[1])
    plt.title("LC-Histogram")
    
    plt.subplot(2,2,4)
    plt.title("HC-Histogram")
    plt.plot(equal[3])


    plt.show()

if __name__ == '__main__':
    print("Bráško, takhle asi ne")
