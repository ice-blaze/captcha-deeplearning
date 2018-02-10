FROM debian:stretch-slim

RUN apt-get update && apt-get install -y \
	build-essential \
	python3 \
	python3-dev \
	libgtk2.0-dev \
	tesseract-ocr \
	python3-pip

WORKDIR /code

ADD . /code

RUN pip3 install -r requirements.txt
