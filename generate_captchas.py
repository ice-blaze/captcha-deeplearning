from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2
import os, errno
import random
import itertools
import uuid
import random
import numpy as np

# CHAR_POSSIBILITIES = "0123456789abcdefghijklmnopqrstuvwxyz"
CHAR_POSSIBILITIES = "2345678abcdefgmnpwxy"
CHAR_COUNT = 5
ALL_LINES = [
    [(30,48), (34,25), (76,15), (259,23)],
    [(30,1), (32,9), (40,15), (55,23), (147,35), (271,33)],
    [(39,36), (42,17), (95,22), (271,31)],
    [(30,16), (36,26), (65,25), (143,16), (205,15), (271,18)],
    [(29, 30), (37, 21), (51, 19), (87, 22), (142, 29), (271, 45)],
    [(29, 7), (32, 30), (34, 27), (56, 33), (77, 35), (148, 29), (271, 9)],
    [(30, 37), (36, 24), (80, 18), (271, 9)],
    [(30, 15), (41, 20), (108, 30), (149, 33), (261, 37),],
    [(30, 35), (37,30), (63, 26), (87, 25), (152, 27), (202, 31), (271, 43)],
    [(36, 35), (43, 19), (74, 20), (271, 39)],
    # [(,), (,), (,), (,), (,), (,), (,)],
]

def try_create_folder(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_random_name(length, possibilities):
    random_name = ''.join(random.SystemRandom().choice(possibilities) for _ in range(length))
    return random_name


def get_random_names(how_many=1, length=1, possibilities="ab"):
    for number in range(0, how_many):
        yield get_random_name(length, possibilities)


def get_random_names_and_lines(how_many=1, length=CHAR_COUNT, possibilities=CHAR_POSSIBILITIES):
    how_many_codes = int(how_many / len(ALL_LINES))
    how_many_images = how_many_codes * len(ALL_LINES)
    print("New number of codes: " + str(how_many_codes))
    print("New number of images: " + str(how_many_images))
    random_names = get_random_names(how_many_codes, length, possibilities)
    for random_name in random_names:
        for line_idx in range(len(ALL_LINES)):
            yield random_name + "-" + str(line_idx)


def generate_captcha(
        name,
        line_idx,
        base_image="./generate-captchas/base.jpg",
):
    CHAR_WIDTH_DELTA = 3
    START_Y = -3
    font = ImageFont.truetype("./fonts/FrutigerBQ-Bold.otf", 42) # not bad

    base_image = Image.open(base_image)
    color = (0, 0, 0)

    start_x = 17
    base = base_image.copy()
    draw = ImageDraw.Draw(base)
    for letter in name:
        if letter in ["n"]:
            start_x -= 2
        draw.text((start_x, START_Y ), letter, color, font=font)
        if letter in ["n", "m"]:
            start_x -= 2
        size = font.getsize(letter)[0]
        start_x += size - CHAR_WIDTH_DELTA

    # draw line for the code
    draw.line(ALL_LINES[int(line_idx)], fill=color, width=4)
    return np.array(base)

    # id = str(uuid.uuid4()).replace("-", "")
    # line_base.save(output_path + random_name + '-' + id + '.png')


def generate_save_captchas():
    # TODO convert generate_catpchas to handle generate_captcha
    pass


def generate_captchas(
        how_many=10,
        char_count=CHAR_COUNT,
        char_possibilities=CHAR_POSSIBILITIES,
        base_image="./generate-captchas/base.png",
        output_path="./generate-captchas/generated/",
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #DEBUG
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    try_create_folder(output_path)

    how_many_codes = int(how_many / len(ALL_LINES))
    how_many_images = how_many_codes * len(ALL_LINES)
    print("New number of codes: " + str(how_many_codes))
    print("New number of images: " + str(how_many_images))
    random_names = get_random_names(how_many_codes, char_count, char_possibilities)
    CHAR_WIDTH_DELTA = 4
    START_Y = -4
    font = ImageFont.truetype("./fonts/Frutiger-Black.otf", 42)

    base_image = Image.open(base_image)
    color = (0, 0, 0)

    for name in random_names:
        start_x = 16
        base = base_image.copy()
        draw = ImageDraw.Draw(base)
        for letter in name:
            draw.text((start_x, START_Y ), letter, color, font=font)
            size = font.getsize(letter)[0]
            start_x += size - CHAR_WIDTH_DELTA
            if letter in ["n", "m", "d"]:
                start_x -= 2

        # draw all possible lines for the code
        for line in ALL_LINES:
            line_base = base.copy()
            draw = ImageDraw.Draw(line_base)
            draw.line(line, fill=color, width=3)

            id = str(uuid.uuid4()).replace("-", "")
            line_base.save(output_path + name + '-' + id + '.png')


def generate_all_possibilites(
        char_count=CHAR_COUNT,
        char_possibilities=CHAR_COUNT,
):
    all_possibilites = []
    counter = 0
    for combination in itertools.product(char_possibilities, repeat=char_count):
        all_possibilites.append(''.join(combination))
        counter += 1
        if counter%100000 == 0:
            print(combination)

    print(len(all_possibilites))


def check_generate_with_real_captcha(
        real_captcha_path
):
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
    # generate_captchas(300000)
    check_generate_with_real_captcha("./real-captchas/")
    # print(generate_captcha())
