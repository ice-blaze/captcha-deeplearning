from generate_captchas import CHAR_POSSIBILITIES
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
import os
import imageio
import keras
import numpy as np
np.random.seed(123)  # for reproducibility

def main(number_of_captchas=None):
    # load data
    GENERATED_CAPTCHA_PATH = "./generate-captchas/generated/"
    CAPTCHAS = os.listdir(GENERATED_CAPTCHA_PATH)[:number_of_captchas]
    NUM_CAPTCHA = len(CAPTCHAS)
    X = np.array([
        imageio.imread(GENERATED_CAPTCHA_PATH + captcha)
        for captcha in CAPTCHAS
    ])

    # Binarizide the labels (multi class)
    label_in_list = [list(x[:-4]) for x in CAPTCHAS]
    label_in_numlist = [
        [
            1.0 if char in captcha else 0.0
            for char in CHAR_POSSIBILITIES
        ]
        for captcha in label_in_list
    ]
    # label need to be list [0,1,0,0,1,...]
    y = np.array(label_in_numlist)


    # 5. Preprocess input data
    X = X.astype('float32')
    X /= np.max(X)  # normalize
    number_of_class = len(CHAR_POSSIBILITIES)


    # create training and test dataset
    pivot = int(NUM_CAPTCHA / 4)
    X_test = X[:pivot]
    y_test = y[:pivot]
    X_train = X[pivot:]
    y_train = y[pivot:]

    # 6. Preprocess class labels
    Y_train = np_utils.to_categorical(y_train, number_of_class)
    Y_test = np_utils.to_categorical(y_test, number_of_class)

    model = Sequential()
    model.add(Conv2D(
        32,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation='relu',
        input_shape=(X[0].shape)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(number_of_class, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(lr=0.01),
        metrics=['accuracy']
    )

    batch_size = 100  # TODO what's a batch size ?
    epochs = 5
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test),
    )

    score = model.evaluate(X_test, y_test, verbose=1)  # Evaluate the trained model on the test set!

    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    main(100)
