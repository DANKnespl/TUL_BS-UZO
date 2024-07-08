"""
Somewhat working implementation of CumShift
"""
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_points(x1, y1, x2, y2):
    """Generates square points from their coordinates"""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def get_coordinates(xt, yt, full_sum, tol):
    w = 0.7 * math.sqrt(full_sum/tol)
    h = 0.9 * math.sqrt(full_sum/tol)
    x_1 = int(xt - w/2)
    x_2 = int(xt + w/2)
    y_1 = int(yt - h/2)
    y_2 = int(yt + h/2)
    return x_1, x_2, y_1, y_2


def get_trailing_window_coordinates(x1, x2, y1, y2, hist, frame, tol):
    """Generates centroid of coordinates of trailing window"""
    centroid_x, centroid_y, full_sum = 0, 0, 0
    for y in range(y1, y2-1):
        for x in range(x1, x2-1):
            val = hist[frame[y][x][0]][0]
            centroid_x += x*val
            centroid_y += y*val
            full_sum += val
    if full_sum != 0:
        full_sum = full_sum
        centroid_x = centroid_x/full_sum
        centroid_y = centroid_y/full_sum
    else:
        centroid_x, centroid_y = 0, 0
    return get_coordinates(centroid_x, centroid_y, full_sum, tol)


def get_trailing_window_coordinates_fast(x1, x2, y1, y2, hist, frame, tol):
    """Generates coordinates of trailing window efficiently"""
    hist_vals = hist[frame[y1:y2-1, x1:x2-1, 0]]
    full_sum = np.sum(hist_vals)
    if full_sum != 0:
        centroid_x = np.sum(np.indices(hist_vals.shape)[
                            1] * hist_vals) / full_sum + x1
        centroid_y = np.sum(np.indices(hist_vals.shape)[
                            0] * hist_vals) / full_sum + y1
    else:
        centroid_x, centroid_y = 0, 0
    return get_coordinates(centroid_x, centroid_y, full_sum, tol)


def do_cumshift(img, vid, tolerance):
    """Function to Shift cum"""
    im = cv2.imread(img)
    x1, x2, y1, y2 = 0, 0, 0, 0
    stream = cv2.VideoCapture(vid)
    _, frame = stream.read()
    hsv_roi = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                       np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    while 1:
        _, frame = stream.read()
        if frame is None:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if x2 == 0:
            x2 = len(frame[0])
            y2 = len(frame)
        x1, x2, y1, y2 = get_trailing_window_coordinates_fast(
            x1, x2, y1, y2, roi_hist, hsv, tolerance)
        pts = get_points(x1, y1, x2, y2)
        pts = np.intp(pts)
        result = cv2.polylines(frame, [pts], True, [0, 255, 255], 2)
        cv2.imshow("tw2", result)
        cv2.waitKey(30)
    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    do_cumshift("./data/cv02_vzor_hrnecek.bmp", "./data/cv02_hrnecek.mp4", 50)
