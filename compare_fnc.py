import cv2
import numpy as np
from pythonRLSA import rlsa


def equalize_shape(imgA, imgB):
    '''make two image have same shape by adding padding'''
    height = imgA.shape[0] - imgB.shape[0]
    width = imgA.shape[1] - imgB.shape[1]
    if height > 0:
        imgB = cv2.copyMakeBorder(
                                                imgB, 0, height, 0, 0,
                                                borderType=cv2.BORDER_CONSTANT,
                                                value=(255, 255, 255))
    else:
        imgA = cv2.copyMakeBorder(
                                                imgA, 0, np.abs(height), 0, 0,
                                                borderType=cv2.BORDER_CONSTANT,
                                                value=(255, 255, 255))
    if width > 0:
        imgB = cv2.copyMakeBorder(
                                                imgB, 0, 0, 0, width,
                                                borderType=cv2.BORDER_CONSTANT,
                                                value=(255, 255, 255))
    else:
        imgA = cv2.copyMakeBorder(
                                                imgA, 0, 0, 0, np.abs(width),
                                                borderType=cv2.BORDER_CONSTANT,
                                                value=(255, 255, 255))
    return imgA, imgB


def get_crop(img):
    ''' crop text area in image'''
    a_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a_size = cv2.resize(a_gray, (1000, 1400), interpolation=cv2.INTER_AREA)
    A = cv2.threshold(a_size, 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # use run length smooth algorithm to detect text region
    rlsa_A = rlsa.rlsa(A, True, False, 10)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    dilate = cv2.erode(rlsa_A, kernel, iterations=2)
    cnts = cv2.findContours(
                                            dilate.copy(),
                                            cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)[0]

    sorted_cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
    try:
        x, y, w, h = cv2.boundingRect(sorted_cnts[-2])
    except IndexError:
        x, y, w, h = cv2.boundingRect(sorted_cnts[0])
    crop = A[y:y+h, x:x+w]
    return crop


def compare(imgA, imgB):
    cropA = get_crop(imgA)
    cropB = get_crop(imgB)
    cropA, cropB = equalize_shape(cropA, cropB)
    sub = (cropA.astype('int32')/255 - cropB.astype('int32')/255)
    result = np.abs(np.sum(sub))
    if result < 27000:
        return True
    else:
        return False
