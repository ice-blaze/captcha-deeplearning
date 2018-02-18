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


def create_model(input_shape, number_of_classes):
    model = Sequential()
    model.add(Conv2D(
        64,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation='relu',
        input_shape=(input_shape)
    ))

    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
    model.add(Dropout(0.5))

    # model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    # model.add(Dense(64*8*8, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        # loss=keras.losses.binary_crossentropy,
        optimizer="Adamax",
        metrics=['accuracy']
    )

    return model


def chunks(array, chunk_size):
    for i in range(0, len(array), chunk_size):
        yield array[i:i + chunk_size]


def one_label(char):
    zeros = [0.0] * len(CHAR_POSSIBILITIES)
    char_index = CHAR_POSSIBILITIES.index(char)
    zeros[char_index] = 1.0
    return zeros


def char_to_num(captcha_name):
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

def load_data_no_generator(GENERATED_CAPTCHA_PATH, CAPTCHAS, CHAR_COUNT):
    # load data
    NUM_CAPTCHA = len(CAPTCHAS)
    x = np.array([
        imageio.imread(GENERATED_CAPTCHA_PATH + captcha)
        for captcha in CAPTCHAS
    ])

    # Binarizide the labels (multi class)
    label_in_list = [
        list(captcha[:CHAR_COUNT])
        for captcha in CAPTCHAS
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


def load_data(GENERATED_CAPTCHA_PATH, CAPTCHAS, CHAR_COUNT, batch_size=100):
    # load data
    NUM_CAPTCHA = len(CAPTCHAS)
    while True:
        for captcha_chunk in CAPTCHAS:
            x = np.array([
                imageio.imread(GENERATED_CAPTCHA_PATH + captcha)
                for captcha in captcha_chunk
            ])

            # Binarizide the labels (multi class)
            label_in_list = [
                list(captcha[:CHAR_COUNT])
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


def main(number_of_captchas=None, model_path=None):
    number_of_classes = len(CHAR_POSSIBILITIES)
    PATH = "./generate-captchas/generated/"
    CAPTCHAS = os.listdir(PATH)[:number_of_captchas]
    CHAR_COUNT = len(CAPTCHAS[0].split("-")[0])
    batch_size = 300

    pivot = int(len(CAPTCHAS) / 10)
    x_five, y_five = next(load_data(
        PATH,
        [CAPTCHAS[:1]],
        CHAR_COUNT,
    ))

    captchas_train = list(chunks(CAPTCHAS[pivot:], batch_size))
    captchas_test = list(chunks(CAPTCHAS[:pivot], batch_size))

    # 6. Preprocess class labels
    # y_train = np_utils.to_categorical(y_train, number_of_classes)
    # y_test = np_utils.to_categorical(y_test, number_of_classes)

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = create_model(x_five[0].shape, number_of_classes * CHAR_COUNT)

        epochs = 1
        model.fit_generator(
            load_data(PATH, captchas_train, CHAR_COUNT, batch_size),
            steps_per_epoch=len(captchas_train),
            epochs=epochs,
            verbose=1,
            # workers=2,
            # use_multiprocessing=True,
            # max_queue_size=100,
            # max_queue_size=len(captchas_train),
        )

        # Save model
        model.save(model_path)

    # score = model.evaluate(x_test, y_test, verbose=1)  # Evaluate the trained model on the test set!
    score = model.evaluate_generator(
        load_data(PATH, captchas_test, CHAR_COUNT),
        steps=batch_size,
    )


    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Test with real captchas
    PATH = "./real-captchas/"
    real_captchas = os.listdir(PATH)
    print_test(model, PATH, real_captchas, CHAR_COUNT, 100)


def print_test(model, path, captchas, char_count, max_size=100):
    print("Real captcha test")
    data = load_data_no_generator(path, captchas, char_count)
    x = data[0]
    y = data[1]
    allx = model.predict(x)

    predicted = [num_to_char(predict, char_count) for predict in allx[:max_size]]
    real = [num_to_char(real_label, char_count) for real_label in y[:max_size]]
    ziper = zip(predicted, real)
    for z in ziper:
        print(str(z[0]==z[1]) + " " + str(z))


def test_real(model_path):
    number_of_classes = len(CHAR_POSSIBILITIES)

    # x_real, y_real, char_count = load_data(
    #     "./real-captchas/",
    # )

    model = load_model(model_path)

    print_test(model, x_real, y_real, char_count, None)



if __name__ == "__main__":
    model_path = "model.h5"
    main(800000, model_path)
    # test_real(model_path)

    # returns a compiled model
    # identical to the previous one
