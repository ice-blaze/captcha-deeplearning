from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os, errno
import random
import itertools
import uuid

CHAR_POSSIBILITIES = "0123456789abcdefghijklmnopqrstuvwxyz"
CHAR_COUNT = 5

def try_create_folder(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_random_names(how_many=1, length=1, possibilities="ab"):
    for number in range(0, how_many):
        random_name = ''.join(random.SystemRandom().choice(possibilities) for _ in range(length))
        yield random_name


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

    random_names = get_random_names(how_many, char_count, char_possibilities)
    CHAR_WIDTH = 18
    START_X = 15
    START_Y = -4
    font = ImageFont.truetype("./fonts/frutiger-bold.ttf", 39)

    base_image = Image.open(base_image)

    for name in random_names:
        base = base_image.copy()
        draw = ImageDraw.Draw(base)
        for idx, letter in enumerate(name):
            draw.text((START_X + (idx * CHAR_WIDTH), START_Y ), letter, (0, 0, 0), font=font)

        # TODO draw a line

        id = str(uuid.uuid4()).replace("-", "")
        base.save(output_path + name + '-' + id + '.png')


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


if __name__ == "__main__":
    generate_captchas(800000)
