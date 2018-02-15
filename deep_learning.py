from generate_captchas import CHAR_POSSIBILITIES
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Convolution2D
from keras.utils import np_utils
from keras.optimizers import SGD
import os
import imageio
import numpy as np
np.random.seed(123)  # for reproducibility


def one_label(char):
    zeros = [0.0] * len(CHAR_POSSIBILITIES)
    char_index = CHAR_POSSIBILITIES.index(char)
    zeros[char_index] = 1.0
    return zeros


def char_to_num(captcha_name):
    number_of_captcha = len(captcha_name)
    all_labels = []
    for char in captcha_name:
        all_labels += one_label(char)
    return all_labels
    # return [
    #     1.0 if char in captcha_name else 0.0
    #     for char in CHAR_POSSIBILITIES
    # ]


def num_to_char(captcha_num, char_count):
    captcha_name = ""

    for x in range(char_count):
        length = len(CHAR_POSSIBILITIES)
        char_range = captcha_num[x * length:(x + 1) * length]
        char_index = np.argmax(char_range)
        captcha_name += CHAR_POSSIBILITIES[char_index]

    return captcha_name



def load_data(path, number_of_captchas=None):
    # load data
    GENERATED_CAPTCHA_PATH = path
    CAPTCHAS = os.listdir(GENERATED_CAPTCHA_PATH)[:number_of_captchas]
    NUM_CAPTCHA = len(CAPTCHAS)
    x = np.array([
        imageio.imread(GENERATED_CAPTCHA_PATH + captcha)
        for captcha in CAPTCHAS
    ])

    # Binarizide the labels (multi class)
    CHAR_COUNT = len(CAPTCHAS[0].split("-")[0])
    label_in_list = [list(captcha_filename[:CHAR_COUNT]) for captcha_filename in CAPTCHAS]
    label_in_numlist = [
        char_to_num(captcha)
        for captcha in label_in_list
    ]
    # label need to be list [0,1,0,0,1,...]
    y = np.array(label_in_numlist)

    # 5. Preprocess input data
    print(x)
    x = x.astype(float)
    x /= np.max(x)  # normalize
    print(label_in_list)



    return x, y, CHAR_COUNT


def create_model(input_shape, number_of_classes):
    model = Sequential()
    model.add(Conv2D(
        32,
        kernel_size=(9, 9),
        strides=(1, 1),
        activation='relu',
        input_shape=(input_shape)
    ))

    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64*8*8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        # loss=keras.losses.binary_crossentropy,
        optimizer="Adamax",
        metrics=['accuracy']
    )

    return model


def main(number_of_captchas=None, model_path=None):
    number_of_classes = len(CHAR_POSSIBILITIES)

    x_one, y_one, one_char_count = load_data(
        "./generate-captchas/one/",
        number_of_captchas,
    )

    x_five, y_five, char_count = load_data(
        "./generate-captchas/generated/",
        number_of_captchas,
    )

    # create training and test dataset
    pivot = int(len(x_five) / 10)
    x_test = x_five[:pivot]
    y_test = y_five[:pivot]
    x_train = x_five[pivot:]
    y_train = y_five[pivot:]

    # 6. Preprocess class labels
    # y_train = np_utils.to_categorical(y_train, number_of_classes)
    # y_test = np_utils.to_categorical(y_test, number_of_classes)

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = create_model(x_train[0].shape, number_of_classes * char_count)

        batch_size = 300  # TODO what's a batch size ?
        epochs = 3
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # validation_data=(x_test, y_test),
        )

        # Save model
        model.save(model_path)

    score = model.evaluate(x_test, y_test, verbose=1)  # Evaluate the trained model on the test set!

    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print_test(model, x_test, y_test, char_count, 100)


def print_test(model, x, y, char_count, max_size=100):
    allx = model.predict(x)
    predicted = [num_to_char(predict, char_count) for predict in allx[:max_size]]
    real = [num_to_char(real_label, char_count) for real_label in y[:max_size]]
    ziper = zip(predicted, real)
    for z in ziper:
        print(str(z[0]==z[1]) + " " + str(z))


def test_real(model_path):
    number_of_classes = len(CHAR_POSSIBILITIES)

    x_real, y_real, char_count = load_data(
        "./real-captchas/",
    )

    model = load_model(model_path)

    print_test(model, x_real, y_real, char_count, None)



if __name__ == "__main__":
    model_path = "model.h5"
    # main(50000, model_path)
    test_real(model_path)

    # returns a compiled model
    # identical to the previous one
