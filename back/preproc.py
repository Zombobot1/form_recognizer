from PIL import Image
import cv2
from math import atan2, cos, sin, sqrt, pi
import numpy as np


def remove_rotation2(img):
    img = crop_n_pad_image(img, pad_x=100, pad_y=100)
    inverse = cv2.bitwise_not(img)
    thresh = cv2.threshold(
        inverse, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((2, 2), np.uint8)
    shifted = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, 255*np.ones((1, 100), dtype=np.uint8))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    detected_line = cv2.morphologyEx(
        shifted, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    # cv2_imshow(shifted)
    contours, hierarchy = cv2.findContours(
        detected_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    angle = 0
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        height, width = img.shape
        new_img = np.full((height, width), fill_value=0, dtype=np.uint8)
        new_img[y:y+h, x:x+w] = detected_line[y:y+h, x:x+w]
        angle = get_orientation(c, new_img)*180/pi
        rotated = rotate_img(img, angle, max_angle=10)
    else:
        rotated = img
    # print(angle)
    rotated = crop_n_pad_image(rotated, pad_x=10, pad_y=10)
    return rotated


def get_orientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
          cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
          cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    # orientation in radians
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    return angle


def rotate_img(img, angle, max_angle=10):
    if abs(angle) <= 10:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated = img
    return rotated


def crop_n_pad_image(img, pad_x=0, pad_y=0):
    inverse = cv2.bitwise_not(img)
    thresh = cv2.threshold(
        inverse, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thresh)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)
    rect = np.full((h+2*pad_y, w+2*pad_x), fill_value=255, dtype=np.uint8)
    rect[pad_y:pad_y+h, pad_x:pad_x+w] = img[y:y+h, x:x+w]
    return rect


def scale_img(img, px):
    inverse = cv2.bitwise_not(img)
    thresh = cv2.threshold(
        inverse, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((2, 2), np.uint8)
    shifted = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, 255*np.ones((1, 100), dtype=np.uint8))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    detected_line = cv2.morphologyEx(
        shifted, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(
        detected_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    angle = 0
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(inverse,(x,y),(x+w,y+h),(255,255,255),2)
        scaling = px/h
        # print(scaling)
        if scaling >= 1:
            img = cv2.resize(img, None, fx=scaling, fy=scaling,
                             interpolation=cv2.INTER_CUBIC)
        if scaling < 1:
            img = cv2.resize(img, None, fx=scaling, fy=scaling,
                             interpolation=cv2.INTER_LINEAR)
    return img


def scale_and_remove_noise(img, opt_height=100):
    h, w = img.shape
    if h < opt_height:
        scaling = opt_height/h
        img = cv2.resize(img, None, fx=scaling, fy=scaling,
                         interpolation=cv2.INTER_CUBIC)

    kernel = np.ones((2, 2), np.uint8)
    inv = cv2.bitwise_not(img)
    opening = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    img = cv2.adaptiveThreshold(cv2.bitwise_not(
        opening), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)

    img = cv2.threshold(cv2.GaussianBlur(img, (3, 3), 0),
                        70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img


def remove_horizontal_line(img):
    thresh = cv2.threshold(
        img, 130, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    shifted = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, 255*np.ones((1, 10), dtype=np.uint8))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 1))
    detected_lines = cv2.morphologyEx(
        shifted, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(
        detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    height, width = img.shape
    filter_img = np.full((height, width), fill_value=0, dtype=np.uint8)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        pad = 6
        filter_img[max(y-pad, 0):min(y+h+pad, height), max(0, x-pad):min(width-1, x+w+pad)
                   ] = img[max(y-pad, 0):min(y+h+pad, height), max(0, x-pad):min(width-1, x+w+pad)]
        cv2.drawContours(filter_img, [c], -1, (255, 255, 255), 2)
        # Repair image
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        filter_img = 255 - \
            cv2.morphologyEx(255 - filter_img, cv2.MORPH_CLOSE,
                             repair_kernel, iterations=1)
        pad = 1
        img[max(y-pad, 0):min(y+h+pad, height-1), max(0, x-pad):min(width-1, x+w+pad)
            ] = filter_img[max(y-pad, 0):min(y+h+pad, height-1), max(0, x-pad):min(width-1, x+w+pad)]

    result = img
    return result


def remove_border_snippets(img):
    _, thresh = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY_INV)  # Change

    # Perform morphological closing
    out = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 255 *
                           np.ones((3, 3), dtype=np.uint8))
    # Perform dilation to expand the borders of the text to be sure
    out = cv2.dilate(thresh, 255*np.ones((10, 100), dtype=np.uint8))
    # For OpenCV 3.0
    # _,contours,hierarchy = cv2.findContours(out,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # Change
    # For OpenCV 2.4.x
    contours, hierarchy = cv2.findContours(
        out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the area made by each contour
    areas = [cv2.contourArea(c) for c in contours]

    # Figure out which contour has the largest area
    idx = np.argmax(areas)

    # Choose that contour, then get the bounding rectangle for this contour
    cnt = contours[idx]
    x, y, w, h = cv2.boundingRect(cnt)
    height, width = img.shape
    new_img = np.full((height, width), fill_value=255, dtype=np.uint8)
    new_img[y:y+h, x:x+w] = img[y:y+h, x:x+w]

    # kernel_dil = np.ones((3,3), np.uint8)

    # img_dilation = cv2.dilate(new_img, kernel_dil, iterations=1)
    # cv2_imshow(img_dilation)
    kernel_er = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(new_img, kernel_er, iterations=3)
    # cv2_imshow(img_erosion)
    _, mask_img = cv2.threshold(
        img_erosion, 220, 255, cv2.THRESH_BINARY_INV)  # Change
    # cv2_imshow(mask_img)
    kernel = np.ones((2, 2), np.uint8)
    shifted = cv2.morphologyEx(
        mask_img, cv2.MORPH_CLOSE, 255*np.ones((1, 100), dtype=np.uint8))
    shifted = cv2.dilate(shifted, kernel, iterations=1)

    kernel_er_shif = np.ones((5, 1), np.uint8)
    shifted_eroded = cv2.erode(shifted, kernel_er_shif, iterations=3)
    # cv2_imshow(shifted)
    # cv2_imshow(shifted_eroded)

    contours, hierarchy = cv2.findContours(
        shifted_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    height, width = img.shape
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        mask_img[0:, 0:x] = 0
        mask_img[0:y, 0:] = 0
        mask_img[0:, min(x+w, width-1)] = 0
        mask_img[min(y+h, height-1):, 0:] = 0
        another_mask = np.full(
            (height, width), fill_value=0, dtype=np.uint8)
        another_mask = cv2.fillPoly(another_mask, pts=[c], color=(255))
        # cv2_imshow(another_mask)

        # cv2_imshow(mask_img)
    new_img = cv2.bitwise_and(new_img, mask_img)
    new_img[mask_img == 0] = 255  # Optional
    new_img = cv2.bitwise_and(new_img, another_mask)
    new_img[another_mask == 0] = 255  # Optional

    # Crop
    new_img[0:3, 0:] = 255
    new_img[height-3:height, 0:] = 255

    new_img[0:, 0:3] = 255
    new_img[0:, width-3:width] = 255
    # cv2_imshow(thresh)
    # cv2_imshow(out)
    # cv2_imshow(new_img)
    return new_img


def preprocess(img):

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = scale_and_remove_noise(img)
    img = remove_horizontal_line(img)
    img = remove_border_snippets(img)
    #img = remove_rotation2(img)
    img = crop_n_pad_image(img, pad_x=25, pad_y=30)
    img = scale_img(img, 25)
    return img
