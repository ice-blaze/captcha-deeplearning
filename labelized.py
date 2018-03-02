import requests
import uuid
import random
from time import sleep
import os
from PIL import Image

from scrap import extract_captcha


def save_nolabeled_real_captcha(output_path):
    """
    Load and save one real captcha

    :param output_path string: folder where you save the loaded captcha
    :return None:
    """
    session_requests = requests.session()
    captcha_url = "--humhum--"
    result = session_requests.get(captcha_url)

    image_base_64 = extract_captcha(result)

    random_name = str(uuid.uuid4()).replace("-", "")
    filename = output_path + "needlabel-" + random_name + ".jpeg"
    with open(filename, "wb") as fh:
        fh.write(image_base_64)
        fh.close()


def save_nolabeled_real_captchas(
        output_path, count=10, min_sec=1.0, max_sec=5.0
):
    """
    Load and save real captchas

    :param output_path string: path where to save captcha
    :param count int: number of captcha saved
    :param min_sec float: minimum waiting time before getting next catpcha
    :param max_sec float: maximum waiting time before getting next catpcha
    :return None:
    """
    for n in range(count):
        save_nolabeled_real_captcha(OUTPUT_PATH)
        print(
            "image processed: " + str(n + 1) + "/" + str(count) +
            " " + str(n / count * 100) + "%"
        )
        sleep(random.uniform(min_sec, max_sec))


def labelize_real_captchas(unlabelized_captcha_path, labelized_captcha_path):
    """
    Display a captcha and ask the label to the user.
    Move this image from input_path to output_path folder.

    :param unlabelized_captcha_path string: folder containing unlabelized captchas
    :param labelized_captcha_path string: folder containing labelized captchas
    :return None:
    """
    files = os.listdir(unlabelized_captcha_path)
    for file in files:
        img = Image.open(unlabelized_captcha_path + file)
        img.show()
        captcha_code = input("Captcha ?")
        print(captcha_code)

        # move the file
        new_filename = labelized_captcha_path + captcha_code + "-" + file.split("-")[1]
        os.rename(unlabelized_captcha_path + file,  new_filename)


if __name__ == "__main__":
    OUTPUT_PATH = "./need-label-real-captchas/"
    LABELED_REAL_CAPTCHAS_PATH = "./real-captchas/"

    # save_nolabeled_real_captchas(OUTPUT_PATH, 100)
    labelize_real_captchas(OUTPUT_PATH, LABELED_REAL_CAPTCHAS_PATH)
