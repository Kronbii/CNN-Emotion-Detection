# Emotion_detection_with_CNN

### Packages need to be installed
- All packages required are found in the requirements text file.
- In your conda environment or file directory, use:
  - pip install -r "path"
  - replace path by the path to the requirements text file.
  - make sure to add .txt to the end of the path

### Dataset Used
- The used dataset is confidential and can not be shared.
- You can add your own dataset to the data file by dividing it into the
  test and train folders.
- The model applies minor preprocessing operations to the dataset before processing it

### Train Emotion detector
- with all face expression images in the Dataset
- command --> python TrainModel.py

It will take several hours depends on your processor. (On AMD processor ryzen 7500H with 16 GB RAM it took me around 4 hours)
after Training , you will find the trained model structure and weights are stored in your project directory.
emotion_model.json
emotion_model.h5

copy these two files create model folder in your project directory and paste it.

### run your emotion detection test file
python TestModel.py

### Evaluate your model by getting its accuracy and confusion matrix
python EvaluateModel.py
