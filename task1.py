import wandb
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from wandb.keras import WandbCallback

wandb.init(project="wieik-si-lab-cnn")

epochs = 42
batch_size = 25
validation_steps = 10000

(x_train, y_train), (x_test, y_test) = mnist.load_data()

width = x_train.shape[1]
height = x_train.shape[2]
channels = 1  # greyscale

x_train = x_train.reshape(x_train.shape[0], width, height, channels)
x_test = x_test.reshape(x_test.shape[0], width, height, channels)

# create model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="Valid", activation="relu", input_shape=(width, height, channels),))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="Valid", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(5, 5), padding="Valid", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="Valid", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dense(519, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"],
)

annealer = ReduceLROnPlateau(monitor="val_acc", patience=1, verbose=2, factor=0.5, min_lr=0.0000001)

# create augmentator
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
)

train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
test_generator = datagen.flow(x_test, y_test, batch_size=batch_size)

print(train_generator)

# train model
model.fit_generator(
    train_generator,
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=validation_steps // batch_size,
    callbacks=[WandbCallback(), annealer],
)

score = model.evaluate(x_test, y_test)
print("Test accuracy: ", score[1])
