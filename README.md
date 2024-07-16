# Description
EmoBrand is a web tool that aims to help companies discover how their clients feel about them by segmenting social media posts into the basic 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise.

https://github.com/user-attachments/assets/55289475-cdd5-4c35-ac56-dca357c6431a

## Usage:
**1. Install Requirements**
- Download and install the latest version from python from [here](https://www.python.org/downloads/) and make sure to select the option "Add to path" while installing.
- Install the rest of the requirements using the following commands.

  ```
  pip install -r requirements.txt
  ```
  
**2. Run model code**
- To be able to run the application without any issues, you need to run the notebook `roberta_emotion_classification_model.ipynb`
- This notebook will train the model and save it into a new folder called `model`. This folder will be used while running the app.

**3. Run application**
Congrats! Now the app is ready to run!! 
- Go to `index` folder and double click on `app.py`.
- Open up your browser and type `http://127.0.0.1:5000/`
