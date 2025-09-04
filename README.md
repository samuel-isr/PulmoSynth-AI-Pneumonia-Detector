
# **PulmoSynth AI:**

An Advanced Pneumonia Detector

This is a model that uses an ensemble of two deep learning models to detect pneumonia in chest X-ray images with **92.63% accuracy**.

Welcome to PulmoSynth AI. This project was a deep dive into a real world challenge could we build a truly reliable tool to detect pneumonia from X-rays? The journey had its fair share of twists and turns, from hitting performance walls to completely rethinking the approach. What came out of it was something remarkable, a powerful ensemble model that hits 92.63% accuracy on data it has never seen before, all wrapped up in a simple, stable desktop app.

This repository tells the story of that process and gives you everything you need to run it yourself.

Project Author :
This project was conceptualized, coded, and trained by I Samuel.

What Makes This Project Special:

It's More Than Just One AI Instead of relying on a single model, PulmoSynth uses two of the best, EfficientNetB0 and DenseNet121. By having them "vote" on the final diagnosis, we get a result that is more accurate and reliable than what either could achieve on its own.

No Servers, No Fuss:

The final application is a clean desktop GUI built with Tkinter. It runs entirely on your machine, which means your data stays private and you do not have to worry about web servers or internet connections.

Built on a Strong Foundation:

The models came pre trained on the massive ImageNet dataset, giving them a head start in understanding shapes, textures, and patterns. I then carefully fine tuned that knowledge to make them experts at one specific task, spotting pneumonia.

Technical Stack:
Modeling & Application Python

Deep Learning Framework:
TensorFlow / Keras

Core Libraries:
Scikit learn, Pillow (PIL), Matplotlib

Desktop GUI Tkinter:
(Python's native GUI library)

Project Structure:


Here’s a look at the main files and what they do:
<img width="857" height="521" alt="image" src="https://github.com/user-attachments/assets/7780cd27-b12c-473c-9b7d-0bee1bac5484" />



How to Run This Project
1. Set Up Your Environment
First, you will want to get a local copy of this project. Then, from the project folder, the best way to keep things clean is to create a Python virtual environment.

# Navigate to the project folder
cd path/to/pneumonia_ai

# Create a new virtual environment
python -m venv venv

# Activate the environment
# On Windows (PowerShell):
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate


2. Install the Dependencies
I have included a requirements.txt file with the exact library versions that are known to work together.

# Make sure your venv is active
pip install -r requirements.txt


3. Train the AI Models (The Important Part)
This is where the magic happens. You will need to run the training script to create the two .h5 model files that the app uses. This is a pretty intensive process and will take a while.
python train.py


When it is done, you will have two new files in your folder, pneumonia_model_EfficientNetB0.h5 and pneumonia_model_DenseNet121.h5.

4. Launch the Application
With the models trained, you are ready to go. Just run the app script.

python pneumonia_app.py


A window should pop up. It will take a moment to load the AI models, and the status bar at the bottom will let you know when it is ready.

Project Author :
This project was conceptualized, coded, and trained by I Samuel.

License:
_**This project is licensed under the MIT License. See the LICENSE file for more details.**_

Acknowledgements:
I could not have made this project without the "Chest X-Ray Images (Pneumonia)" dataset, which is publicly available on Kaggle. A big thank you to the creators and curators of this dataset for making their work accessible.
