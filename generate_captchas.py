from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2
import os
import random
import numpy as np

# CHAR_POSSIBILITIES = "0123456789abcdefghijklmnopqrstuvwxyz"
CHAR_POSSIBILITIES = "2345678abcdefgmnpwxy"
CHAR_COUNT = 5
ALL_LINES = [
    [(30, 48), (34, 25), (76, 15), (259, 23)],
    [(30, 1), (32, 9), (40, 15), (55, 23), (147, 35), (271, 33)],
    [(39, 36), (42, 17), (95, 22), (271, 31)],
    [(30, 16), (36, 26), (65, 25), (143, 16), (205, 15), (271, 18)],
    [(29, 30), (37, 21), (51, 19), (87, 22), (142, 29), (271, 45)],
    [(29, 7), (32, 30), (34, 27), (56, 33), (77, 35), (148, 29), (271, 9)],
    [(30, 37), (36, 24), (80, 18), (271, 9)],
    [(30, 15), (41, 20), (108, 30), (149, 33), (261, 37)],
    [(30, 35), (37, 30), (63, 26), (87, 25), (152, 27), (202, 31), (271, 43)],
    [(36, 35), (43, 19), (74, 20), (271, 39)],
]


def get_random_captcha_name(
        length,
        possibilities,
):
    """
    :param length int: number of character in the captcha
    :param possibilities string: all character possible in the captcha
    :return string: random captcha name
    """
    random_name = ''.join(
        random.SystemRandom().choice(possibilities) for _ in range(length)
    )
    return random_name


def get_random_captcha_names(
        how_many=1,
        length=1,
        possibilities="ab",
):
    """
    :param how_many int: number of name+line generated
    :param length int: number of character in the captcha
    :param possibilities string: all character possible in the captcha
    :return list string: list containing random captcha name
    """
    for number in range(0, how_many):
        yield get_random_captcha_name(length, possibilities)


def get_random_captcha_names_and_lines(
        how_many=1,
        length=CHAR_COUNT,
        possibilities=CHAR_POSSIBILITIES,
):
    """
    :param how_many int: number of name+line generated
    :param length int: number of character in the captcha
    :param possibilities string: all character possible in the captcha
    :return string: string containing random_name-line_idx
    """
    how_many_codes = int(how_many / len(ALL_LINES))
    how_many_images = how_many_codes * len(ALL_LINES)
    print("New number of codes: " + str(how_many_codes))
    print("New number of images: " + str(how_many_images))
    random_names = get_random_captcha_names(
        how_many_codes, length, possibilities
    )
    for random_name in random_names:
        for line_idx in range(len(ALL_LINES)):
            yield random_name + "-" + str(line_idx)


def generate_captcha(
        captcha_code,
        line_idx,
        base_image_path="./generate-captchas/base.jpg",
):
    """
    :param name string: captcha code that will be generated
    :param line_idx int: index of the line to draw on the captcha
    :param base_image_path string: path of the background image for captcha
    :return base numpy array2d: captcha image generated
    """
    CHAR_WIDTH_DELTA = 3
    START_Y = -3
    font = ImageFont.truetype("./fonts/FrutigerBQ-Bold.otf", 42)

    back_ground = Image.open(base_image_path)
    color = (0, 0, 0)

    start_x = 17
    base = back_ground.copy()
    draw = ImageDraw.Draw(base)
    for letter in captcha_code:
        if letter in ["n"]:
            start_x -= 2
        draw.text((start_x, START_Y), letter, color, font=font)
        if letter in ["n", "m"]:
            start_x -= 2
        size = font.getsize(letter)[0]
        start_x += size - CHAR_WIDTH_DELTA

    # draw line for the code
    draw.line(ALL_LINES[int(line_idx)], fill=color, width=4)
    return np.array(base)


def compare_generate_with_real_captcha(real_captcha_path):
    """
    Compare the images of a real captcha and a generated captcha
    :param real_captcha_path string: folder containing real captcha
    """
    real_captcha_files = os.listdir(real_captcha_path)
    random.shuffle(real_captcha_files)
    for real_captcha_file in real_captcha_files[:1]:
        code = real_captcha_file.split("-")[0]
        line_idx = 0
        generated_image = generate_captcha(code, line_idx)
        real_image = cv2.imread(real_captcha_path + real_captcha_file, 0)

        cv2.imshow('generated', generated_image)
        cv2.imshow('real', real_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    compare_generate_with_real_captcha("./real-captchas/")
    # print(generate_captcha())
