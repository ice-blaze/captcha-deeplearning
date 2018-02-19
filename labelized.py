import requests
from lxml import html
import uuid
import random
from time import sleep
import os
from PIL import Image

from scrap import extract_captcha


def save_nolabeled_real_captcha(output_path):
    session_requests = requests.session()
    login_url = "--humhum--"
    result = session_requests.get(login_url)

    image_base_64 = extract_captcha(result)

    random_name = str(uuid.uuid4()).replace("-", "")
    filename = output_path + "needlabel-" + random_name + ".jpeg"
    with open(filename, "wb") as fh:
        fh.write(image_base_64)
        fh.close()


def save_nolabeled_real_captchas(output_path, count=10, min_sec=1, max_sec=5):
    for n in range(count):
        save_nolabeled_real_captcha(OUTPUT_PATH)
        print("image processed: " + str(n + 1) + "/" + str(count) + " " + str(n/count * 100) + "%")
        sleep(random.uniform(min_sec, max_sec))

def labelize_real_captchas(input_path, output_path):
    files = os.listdir(input_path)
    for file in files:
        img = Image.open(input_path + file)
        img.show()
        captcha_code = input("Captcha ?")
        print(captcha_code)
        new_filename = output_path + captcha_code + "-" + file.split("-")[1]
        os.rename(input_path + file,  new_filename)

        # move file


if __name__ == "__main__":
    OUTPUT_PATH = "./need-label-real-captchas/"
    LABELED_REAL_CAPTCHAS_PATH = "./real-captchas/"

    # save_nolabeled_real_captchas(OUTPUT_PATH, 100)
    labelize_real_captchas(OUTPUT_PATH, LABELED_REAL_CAPTCHAS_PATH)
