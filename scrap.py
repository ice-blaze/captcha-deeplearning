import requests
from lxml import html
import base64
import uuid


def extract_captcha(result):
    tree = html.fromstring(result.text)
    raw_captcha_balisa = tree.xpath("//captcha/div/@style")

    if len(raw_captcha_balisa) != 1:
        raise "No or too much captcha found"
    else:
        raw_captcha_balisa = raw_captcha_balisa[0]

    first_coma_index = raw_captcha_balisa.index(",")
    first_closing_parenthesis_index = raw_captcha_balisa.index(")")
    image_base64_string = raw_captcha_balisa[first_coma_index + 1:first_closing_parenthesis_index - 1].encode()

    return base64.b64decode(image_base64_string)
