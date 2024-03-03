# import required packages
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1.0 / 255)
validation_data_gen = ImageDataGenerator(rescale=1.0 / 255)

# variables needed for parameters
train_data_qtt = 1628  ########################################CHANGE
test_data_qtt = 405  ########################################CHANGE
picture_pixel_size_x = 48  # in pixels
picture_pixel_size_y = 48  # in pixels
picture_mode = 1  # 1 for grayscale 3 for RGB

# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
    "data/train",  ########################################CHANGE
    target_size=(picture_pixel_size_x, picture_pixel_size_y),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
)

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
    "data/test",  ########################################CHANGE
    target_size=(picture_pixel_size_x, picture_pixel_size_y),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
)

# create 3 layer neural network for the model
emotion_model = Sequential()

emotion_model.add(
    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(picture_pixel_size_x, picture_pixel_size_x, picture_mode),
    )
)

emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
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
    optimizer=Adam(lr=0.00001, decay=1e-6),
    metrics=["accuracy"],
)

"""
learning_rate = 0.001
momentum = 0.9
nesterov = True

optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
)

emotion_model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)
"""
# Train the neural network/model

emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch=train_data_qtt // 128,
    epochs=5000,
    validation_data=validation_generator,
    validation_steps=test_data_qtt // 128,
)

# save model structure in jason file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights("emotion_model.h5")
