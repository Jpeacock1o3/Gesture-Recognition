# Gesture-Recognition
This is a starter computer vision project that allows for you to train an AI to learn different gestures and detect them in real-time.
There are three files, the names are pretty self explanatory but:
## requirements
I recommend to install the virtual environment because it makes things easier in my opinion. To install it, run "python3 -m venv .venv" to create the venv, then according to your os:
  - Windows (powershell): ".\.venv\Scripts\Activate.ps1"
  - Windows (command prompt): ".\.venv\Scripts\activate.bat"
  - Macos/linux (bash/Zsh): "source .venv/bin/activate"    
Requirement.txt is the file for all the dependencies and libraries for the project. Run "pip install -r requirements.txt" in your command line to download all libraries in the txt file to your .venv
## collect_gesture_data
collect_gesture_data.py is the first program you should run. It will allow you to train the AI using your own gestures, right now it takes 200 frames for each gesture, this can be changed to take more or less. Only run this file when you want to make a new dataset with the same gestures or more gestures
## Train_custom_model.
train_custom_model uses the data gathered from the training set and uses it to train the model. Only run this model when you have new data you want the model to be trained with
## Gesture.py
This is the final file you should run when everything is done, this will open your camera and identify the action being done by your hand and will show its confidence that it is right. you can run this as many times as you want without needing to run the rest
## Libraries used
  - os: This was used for ths functionality that allows for certain gestures to open destinations, handles file paths, creates directories
  - json: reading and writing gesture data in json format
  - time: timestamps data files, cooldown periods between action triggers, and adding delays in data collection
  - numpy: handling numerical data like hand landmarks and handles math for data like preprocessing and model training
  - opencv-python (cv2): captures video from webcam, process and display video frames, and draws landmarks and annotations on frames for visualization
  - mediapipe: real-time hand landmark detection, precise hand cords from frames for recognition
  - scikit-learn (sklearn): transforms labels/gestures into numbers using LabelEncoder, standardized features using StandardScaler to improve performance, divides data into training, validation, and testing using train_test_split, asses model performance with classification_report and confusion_matrix
  - joblib: saves and preprocesses objects like StandardScalar and LabelEncoder
  - tensorflow: builds and trains neural network models using tensorflow.keras, handles deep learning task
  - matplotlib: plots training metrics for accuracy and loss
  - subprocess: executes system commands
  - platform: detects operaring system, it reads the users os, so that no change is needed other than the user changing what gestures do what

## Extras
  - the commands for the gestures are sensitive, i'm working on making them less sensitive, but they activate very easily sometimes too easliy, so you may want to comment out lines 198 to 210, and unindent line 211 by 1 in gesture.py to stop the commands from activating if you just want to test out the recognition
  - I am also working on making to allow the use of a already created data set like the Hand Gesture Recognition Database on kaggle
  - to add more gestures, just add your wanted gestures the way the rest are added in line 11 of collect_gesture_data.py
  - to add more commands, add them in your respective os same way the rest are added
