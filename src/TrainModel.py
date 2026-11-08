# import required packages
import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.optimizers.schedules import InverseTimeDecay

train_data_dir = "data/train"
test_data_dir = "data/test"

# variables needed for parameters
picture_pixel_size_x = 48  # in pixels
picture_pixel_size_y = 48  # in pixels
picture_mode = 1  # 1 for grayscale 3 for RGB
batch_size = 64


def load_images(folder):
    # one sub-folder per class; pixels rescaled from 0-255 to 0-1
    images = keras.utils.image_dataset_from_directory(
        folder,
        image_size=(picture_pixel_size_x, picture_pixel_size_y),
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="categorical",
    )
    return images.map(lambda x, y: (x / 255.0, y))


# Preprocess all train images
train_generator = load_images(train_data_dir)

# Preprocess all test images
validation_generator = load_images(test_data_dir)

# create 3 layer neural network for the model
emotion_model = Sequential()

emotion_model.add(keras.Input(shape=(picture_pixel_size_x, picture_pixel_size_y, picture_mode)))
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))

# RELU was used to help the model learn complex and non-linear representations of the input data. and converge the model faster.
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())

emotion_model.add(Dense(1024, activation="relu"))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(4, activation="softmax"))


cv2.ocl.setUseOpenCL(False)

emotion_model.compile(
    loss="categorical_crossentropy",
    # learning rate 1e-5, decayed as 1e-5 / (1 + 1e-6 * step), as the old decay=1e-6 did
    optimizer=Adam(learning_rate=InverseTimeDecay(0.00001, decay_steps=1, decay_rate=1e-6)),
    metrics=["accuracy"],
)

# Train the neural network/model

emotion_model_info = emotion_model.fit(
    train_generator,
    epochs=1000,
    validation_data=validation_generator,
)

# save model structure in jason file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weights (Keras 3 needs the .weights.h5 ending)
emotion_model.save_weights("emotion_model.weights.h5")
