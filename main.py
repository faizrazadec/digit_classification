import io
import pickle

import numpy as np
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert('L')
    pil_image = PIL.ImageOps.invert(pil_image)
    pil_image = pil_image.resize((28, 28), PIL.Image.Resampling.LANCZOS)
    img_array = np.array(pil_image).reshape(1, -1)
    prediction = model.predict(img_array)
    return {'prediction': int(prediction[0])}