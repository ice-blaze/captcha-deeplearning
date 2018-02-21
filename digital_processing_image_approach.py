from PIL import Image
from PIL import ImageEnhance
import PIL.ImageOps
import pytesseract
import argparse
import cv2
import os
import numpy
import random
from generate_captchas import CHAR_POSSIBILITIES

import cv2
import numpy as np
from skimage.morphology import opening

CAPTCHAS_PATH = "./real-captchas/"
# CAPTCHAS_PATH = "./generate-captchas/generated/"
CONFIG = "-c tessedit_char_whitelist=" + CHAR_POSSIBILITIES + " -psm 8"

def clean_image_kernel4(image, start_letters=16, end_letters=135):
    image = cv2.bitwise_not(image)
    #image = cv2.imread('easy.png', 0)
    kernel = np.ones((4,4), np.uint8)

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # cv2.imshow(path + 'closing', image)

    #opening_skimage = opening(image, kernel)

    #cv2.imshow('opening', opening)
    ret,image = cv2.threshold(image,207,255,cv2.THRESH_BINARY)
    image = cv2.erode(image,kernel,iterations = 1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow(path + 'erosion1', image)

    kernel = np.array([[0, 0, 1],
                       [1, 1, 0],
                       [0, 1, 0]])
    image = cv2.filter2D(image,-1,kernel)
    # image[:, end_letters:] = 0
    # image[:, :start_letters] = 0
    image = image[:, start_letters:end_letters]
    image = image[:, :, :1]  # keep only one channel
    # cv2.imshow(path+'closing->thresh', image)
    # cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

    # cv2.imshow(path + 'erosion1', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

def clean_image_kernel3(image, start_letters=16, end_letters=135):
    image = cv2.bitwise_not(image)
    kernel = np.ones((3,3), np.uint8)

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    ret,image = cv2.threshold(image,207,255,cv2.THRESH_BINARY)

    image = cv2.erode(image,kernel,iterations = 1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    kernel = np.array([[0, 0, 1],
                       [1, 1, 0],
                       [0, 1, 0]])
    image = cv2.filter2D(image,-1,kernel)
    image[:, end_letters:] = 0
    image[:, :start_letters] = 0
    # cv2.imshow(path+'closing->thresh', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image


def resolve_captcha(path):
    image = cv2.imread(path, 0)

    image = clean_image_kernel4(image)

    text = pytesseract.image_to_string(image, config=CONFIG)
    return text[:5]


def resolve_all():
    captchas = os.listdir(CAPTCHAS_PATH)[:40]
    true_counter = 0
    total_counter = 0
    for captcha in captchas:
        total_counter += 1
        if total_counter%10 == 0:
            print("10 done")
        solution = captcha.split("-")[0]
        result = resolve_captcha(os.path.join(CAPTCHAS_PATH, captcha))
        if solution == result:
            true_counter += 1
        print(str(solution == result ) + " = " + result + " " + solution)

    print(true_counter / len(captchas))

    # cv2.waitKey(0)

if __name__ == "__main__":
    # captchas = os.listdir(CAPTCHAS_PATH)
    # random.shuffle(captchas)
    # for captcha in captchas[:5]:
    #     resolve_captcha(CAPTCHAS_PATH + captcha)
    #     resolve_captcha_kernel3(CAPTCHAS_PATH + captcha)
    resolve_all()
