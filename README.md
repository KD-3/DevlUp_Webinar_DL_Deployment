# DevlUp Webinar
## Deploying Deep Learning Applications
For any deep learning model in general:
- Train the model
- Save the model architecture and weights(json and h5)
- Load up the model and weights and serve them at the needed endpoint

For the MNIST Handwritten App
- Train the model using the given Colab notebook
- Save the json and h5 files
- Load your model through the load.py
- Preprocess your data to store the base64 data into image and further resize and reshape as per the model requirements.
- Send your image as an input to the model
- Store the prediction and make a POST request to the Web App

Steps for running inference app:
- python3 app.py