from fastapi import FastAPI, UploadFile, File
import uvicorn
import sys
import keras
from tensorflow.keras.models import Sequential
import numpy as np
from PIL import Image
import io

# Retrieves the model path from the command line argument
MODEL_PATH = sys.argv[1]

# Function to load a pre-trained TensorFlow model from a specified file path
def load_model(model_path: str) -> Sequential:
    # Uses Keras' load_model function to load and return the model from the given path
    return keras.models.load_model(model_path)

# Function to predict the class of the digit using a pre-trained model and an input data point
def predict_digit(model: Sequential, data_point: np.ndarray) -> str:
    # The model makes a prediction on the data_point, which is a reshaped image array
    # np.argmax is used to find the index of the maximum value in the prediction output array,
    # which corresponds to the predicted class. The result is converted to a string before returning.
    return str(np.argmax(model.predict(data_point, verbose=True)))

# Load the pre-trained model from the given path when the script is run
model = load_model(MODEL_PATH)

# Initialize the FastAPI app
app = FastAPI()

@app.get("/")
# Define a root endpoint that provides information on how to use the API
def read_root():
    # Returns a message directing users to the documentation page
    return {"Message": "Please visit the /docs page to use the prediction API."}

@app.post("/predict/")
# Define an endpoint for predicting digits from uploaded images
async def predict_digit_from_image(file: UploadFile = File(...)):
    # Reads the image file uploaded by the user
    contents = await file.read()
    # Opens the image and converts it to grayscale
    image = Image.open(io.BytesIO(contents)).resize((28, 28)).convert("L")
    # Converts the processed image to a numpy array and reshapes it to match the model's input shape
    formatted_image = np.array(image).reshape(1, 784)
    # Predicts the digit using the pre-loaded model and the processed image data
    digit = predict_digit(model, formatted_image)
    # Returns the predicted digit as follows
    return {"digit": digit}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.4", port=1000)
