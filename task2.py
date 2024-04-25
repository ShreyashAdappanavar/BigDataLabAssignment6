from fastapi import FastAPI, UploadFile, File
import uvicorn
import sys
import keras
from tensorflow.keras.models import Sequential
import numpy as np
from PIL import Image
import io

# Retrieve the path to the model file from the command line arguments
MODEL_PATH = sys.argv[1]

# Define a function to load a pre-trained TensorFlow model from the specified file path
def load_model(model_path: str) -> Sequential:
    # Load and return the Keras model stored at the provided path
    return keras.models.load_model(model_path)

# Define a function to predict the class of a digit using the loaded model and a prepared image data point
def predict_digit(model: Sequential, data_point: np.ndarray) -> str:
    # Model makes a prediction on the formatted image data, then np.argmax finds the index of the maximum
    # probability which corresponds to the predicted digit. This index is returned as a string.
    return str(np.argmax(model.predict(data_point, verbose=True)))

# Define a function to process the image file uploaded by the user
def format_image(contents):
    # Load the image from bytes, resize it to 28x28 pixels, convert it to grayscale,
    # and reshape it into the required input format for the model (1x784 array).
    image = Image.open(io.BytesIO(contents))
    image = image.resize((28, 28)).convert("L")
    return np.array(image).reshape(1, 784)

# Load the model from the specified path when the script is initialized
model = load_model(MODEL_PATH)

# Create a new FastAPI application instance
app = FastAPI()

@app.get("/")
# Define a root endpoint that provides information on how to use the API
def read_root():
    # Returns a simple JSON message directing users to the documentation page
    return {"Message": "Please visit the /docs page to use the prediction API."}

@app.post("/predict/")
# Define a POST endpoint for predicting the digit from an uploaded image
async def predict_digit_from_image(file: UploadFile = File(...)):
    # Asynchronously read the contents of the uploaded file
    contents = await file.read()
    # Process the image to get it in the required format for prediction
    formatted_image = format_image(contents)
    # Predict the digit using the processed image and return the result
    digit = predict_digit(model, formatted_image)
    return {"digit": digit}

# Specify the entry point for running the FastAPI application using uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.3", port=8000)  # Run the server on the specified host and port
