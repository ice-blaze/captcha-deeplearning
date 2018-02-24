import pytesseract
import cv2
import os
import numpy as np

from generate_captchas import CHAR_POSSIBILITIES

CAPTCHAS_PATH = "./real-captchas/"
CONFIG = "-c tessedit_char_whitelist=" + CHAR_POSSIBILITIES + " -psm 8"


def clean_image_kernel4(image, start_letters=16, end_letters=135):
    """
    :param image numpy array2d: image to clean
    :param start_letters int: column index where the captcha start in the image
    :param end_letters int: column index where the captcha end in the image
    :return image numpy array2d: image cleaned
    """
    inverted_color_image = cv2.bitwise_not(image)
    kernel_44 = np.ones((4, 4), np.uint8)

    image_morph_open = cv2.morphologyEx(
        inverted_color_image, cv2.MORPH_OPEN, kernel_44
    )

    ret, image_threshold = cv2.threshold(
        image_morph_open, 207, 255, cv2.THRESH_BINARY
    )
    image_eroded = cv2.erode(image_threshold, kernel_44, iterations=1)
    image_morph_close = cv2.morphologyEx(
        image_eroded, cv2.MORPH_CLOSE, kernel_44
    )

    kernel_33 = np.array([
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
    ])
    image_filtered_kernel_33 = cv2.filter2D(image_morph_close, -1, kernel_33)

    image_crop_outside_captcha = image_filtered_kernel_33[
        :, start_letters:end_letters]
    image_keep_one_channel = image_crop_outside_captcha[:, :, :1]

    return image_keep_one_channel


def resolve_captcha(captcha_path, length_of_captcha=5):
    """
    :param captcha_path string: path of the captcha
    :param length_of_captcha int: number of char in the captcha
    :return captcha_code string: captcha code
    """
    image = cv2.imread(captcha_path, 0)

    image_cleaned = clean_image_kernel4(image)

    captcha_code = pytesseract.image_to_string(image_cleaned, config=CONFIG)
    return captcha_code[:length_of_captcha]


def resolve_all(max_captcha=10):
    """
    :param max_captcha int: maximum captcha to be resolved
    """
    captchas = os.listdir(CAPTCHAS_PATH)[:max_captcha]
    true_counter = 0
    current_counter = 0
    for captcha in captchas:
        current_counter += 1
        if current_counter % 10 == 0:
            print("10 done")
        solution = captcha.split("-")[0]
        result = resolve_captcha(os.path.join(CAPTCHAS_PATH, captcha))
        if solution == result:
            true_counter += 1
        print(str(solution == result) + " = " + result + " " + solution)

    print(true_counter / len(captchas))


if __name__ == "__main__":
    resolve_all()
