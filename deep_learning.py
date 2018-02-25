from generate_captchas import CHAR_POSSIBILITIES
from generate_captchas import generate_captcha
from generate_captchas import get_random_captcha_names_and_lines
from digital_processing_image_approach import clean_image_kernel4
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import os
import imageio
import random
import numpy as np
np.random.seed(123)  # for reproducibility


def add_dict(a, b):
    """
    :param a dict: Dictionary we will merge with b
    :param b dict: Dictionary that will be merged into a
    :return a dict: Merged dictionary of a and b
    """
    for key in b:
        a[key] = a.get(key, 0) + b[key]

    return a


def similar(real, predicted):
    """
    Compare if the captcha code predicted is close to the real one
    :param real string: Real captcha string
    :param predicted string: Predicted captcha string
    :return
      wrong_letter_count float: Percentage of wrong letter
      wrong_letter_dict dict: Dict of all wrong letters as key and a counter
        of failed as value
    """
    wrong_letter_count = 0

    wrong_letter_dict = {}
    for real_letter, preddicted_letter in zip(real, predicted):
        if real_letter != preddicted_letter:
            wrong_letter_dict[real_letter] = \
                wrong_letter_dict.get(real_letter, 0) + 1
            wrong_letter_count += 1

    wrong_letter_count /= len(real)
    wrong_letter_count = 1.0 - wrong_letter_count

    return wrong_letter_count, wrong_letter_dict


def create_model(input_shape, number_of_classes):
    """
    :param input_shape numpy1d: Shape of the image
    :param number_of_classes int: Class number the model should handle
    :return model Model: Keras model
    """
    model = Sequential()
    model.add(Conv2D(
        20,
        kernel_size=(5, 5),
        padding="same",
        strides=(1, 1),
        activation='relu',
        input_shape=(input_shape)
    ))

    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64*8*8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="Adamax",
        metrics=['accuracy']
    )

    return model


def chunks(array, chunk_size):
    """
    Convert a 1D list into a 2D list with length of the array of array equal
      to chunk_size
    :param array list: list of object
    :param chunk_size int: length of the chunks
    :return 2d list:
    """
    for i in range(0, len(array), chunk_size):
        yield array[i:i + chunk_size]


def one_label(char):
    """
    Convert one char into a binarized label
    :param char string: one character
    :return zeros list int: binarized label
    """
    zeros = [0.0] * len(CHAR_POSSIBILITIES)
    char_index = CHAR_POSSIBILITIES.index(char)
    zeros[char_index] = 1.0
    return zeros


def char_to_num(captcha_name):
    """
    Convert catpcha character to binarized labels
    :param captcha_name string: code of the captcha
    :return all_labels list int: name transform into binarized labels
    """
    all_labels = []
    for char in captcha_name:
        all_labels += one_label(char)
    return all_labels


def num_to_char(captcha_binarized_label, char_count):
    """
    Convert catpcha binarized labels to char
    :param captcha_binarized_label list int: captcha binarized
    :param char_count int: length of the original captcha name
    :return captcha_name string: captcha code
    """
    captcha_name = ""

    for x in range(char_count):
        length = len(CHAR_POSSIBILITIES)
        char_range = captcha_binarized_label[x * length:(x + 1) * length]
        char_index = np.argmax(char_range)
        captcha_name += CHAR_POSSIBILITIES[char_index]

    return captcha_name


def load_data_no_generator(generated_captcha_path, captchas, char_count):
    """
    :param generated_captcha_path strig: folder containing captchas
    :param catpchas list string: All captcha names
    :param char_count int: Length of the catpcha name
    """
    x = np.array([
        clean_image_kernel4(imageio.imread(generated_captcha_path + captcha))
        for captcha in captchas
    ])

    # Binarizide the labels (multi class)
    label_in_list = [
        list(captcha[:char_count])
        for captcha in captchas
    ]
    label_in_numlist = [
        char_to_num(label)
        for label in label_in_list
    ]
    # label need to be list [0,1,0,0,1,...]
    y = np.array(label_in_numlist)

    # 5. Preprocess input data
    x = x.astype(float)
    x /= np.max(x)  # normalize

    return x, y


def load_data(captchas):
    """
    :param captchas list string: Captcha names
    :return list tuple numpy2d,labels: Tuple of image and labels binarized
    """
    while True:
        for captcha_chunk in captchas:
            x = np.array([
                # TODO opti possible
                clean_image_kernel4(generate_captcha(
                    captcha.split("-")[0], captcha.split("-")[1])
                )
                for captcha in captcha_chunk
            ])

            # Binarizide the labels (multi class)
            label_in_list = [
                list(captcha.split("-")[0])
                for captcha in captcha_chunk
            ]
            label_in_numlist = [
                char_to_num(label)
                for label in label_in_list
            ]
            # label need to be list [0,1,0,0,1,...]
            y = np.array(label_in_numlist)

            # 5. Preprocess input data
            x = x.astype(float)
            x /= np.max(x)  # normalize

            yield x, y


def train_and_test_model(number_of_captchas=10, model_path=None):
    """
    :param number_of_captchas int: Number of captcha we want to for the train
    :param model_path string: Path of the model if it exist
    :return None: Print test result
    """
    number_of_classes = len(CHAR_POSSIBILITIES)
    captchas = list(get_random_captcha_names_and_lines(number_of_captchas))
    random.shuffle(captchas)
    char_count = len(captchas[0].split("-")[0])
    batch_size = 250

    pivot = int(len(captchas) / 10)
    x_five, y_five = next(load_data([captchas[:1]]))

    captchas_train = list(chunks(captchas[pivot:], batch_size))
    captchas_test = list(chunks(captchas[:pivot], batch_size))

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = create_model(x_five[0].shape, number_of_classes * char_count)

        epochs = 1
        model.fit_generator(
            load_data(captchas_train),
            steps_per_epoch=len(captchas_train),
            epochs=epochs,
            verbose=1,
        )

        # Save model
        model.save(model_path)

    score = model.evaluate_generator(
        load_data(captchas_test),
        steps=batch_size,
    )

    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Test with real captchas
    path = "./real-captchas/"
    real_captchas = os.listdir(path)
    print_test(model, path, real_captchas, char_count, 100)


def print_test(model, path, captchas, char_count, max_size=100):
    """
    :param model Model: Keras model to read captchas
    :param path string: Path where are stored real captchas
    :param catpchas list string: All captcha names
    :param char_count int: Length of the catpcha name
    :param max_size int: Number of captcha we want to test
    :return None: Print captcha test results
    """
    print("Real captcha test")
    data = load_data_no_generator(path, captchas, char_count)
    x = data[0]
    y = data[1]
    allx = model.predict(x)

    predicted = [
        num_to_char(predict, char_count) for predict in allx[:max_size]
    ]
    real = [num_to_char(real_label, char_count) for real_label in y[:max_size]]
    ziper = zip(real, predicted)
    correct = 0
    mean_similar = 0
    error_dict = {}
    for z in ziper:
        sim, sim_dict = similar(z[0], z[1])
        mean_similar += sim
        error_dict = add_dict(error_dict, sim_dict)
        if z[0] == z[1]:
            correct += 1
        print(str(z[0] == z[1]) + " " + str(z) + " simili: " + str(sim))
    print("overall: " + str(correct/len(predicted)))
    print("overall similarity: " + str(mean_similar / len(predicted)))
    print(error_dict)
    print(sorted(error_dict.keys()))


if __name__ == "__main__":
    model_path = "model.h5"
    # train_and_test_model(1600000, model_path)
    train_and_test_model(800000, model_path)
