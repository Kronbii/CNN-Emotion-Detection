import numpy as np
import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


emotion_dict = {
    0: "Red",
    1: "Yellow",
    2: "Red",
    3: "Green",
    4: "Blue",
    5: "Blue",
    6: "Yellow",
}


# load json and create model
json_file = open("model/emotion_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Preprocess all test images: one sub-folder per class, pixels rescaled to 0-1,
# kept in file order so the predictions line up with the labels
test_data = keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    label_mode="int",
    shuffle=False,
).map(lambda x, y: (x / 255.0, y))
true_classes = np.concatenate([np.asarray(y) for _, y in test_data])

# do prediction on test data
predictions = emotion_model.predict(test_data)

# see predictions
# for result in predictions:
#     max_index = int(np.argmax(result))
#     print(emotion_dict[max_index])

print("-----------------------------------------------------------------")
# confusion matrix
c_matrix = confusion_matrix(true_classes, predictions.argmax(axis=1))
print(c_matrix)
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=c_matrix, display_labels=emotion_dict
)
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

# Classification report
print("-----------------------------------------------------------------")
print(classification_report(true_classes, predictions.argmax(axis=1)))
