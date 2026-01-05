import uvicorn
import numpy as np
import onnxruntime as ort

from keras_image_helper import create_preprocessor
from pydantic import BaseModel, HttpUrl
from PIL import Image
from fastapi import FastAPI

app = FastAPI(title="animals-classifier")

def preprocess_pytorch_style(X):
    X = X / 255.0
    
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    
    X = X.transpose(0, 3, 1, 2)
    X = (X - mean) / std
    
    return X.astype(np.float32)

preprocessor = create_preprocessor(
    preprocess_pytorch_style,
    target_size=(224, 224)
)

session = ort.InferenceSession(
    "animals_classification_latest.onnx", providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = [
    "antelope",
    "badger",
    "bat",
    "bear",
    "bee",
    "beetle",
    "bison",
    "boar",
    "butterfly",
    "cat",
    "caterpillar",
    "chimpanzee",
    "cockroach",
    "cow",
    "coyote",
    "crab",
    "crow",
    "deer",
    "dog",
    "dolphin",
    "donkey",
    "dragonfly",
    "duck",
    "eagle",
    "elephant",
    "flamingo",
    "fly",
    "fox",
    "goat",
    "goldfish",
    "goose",
    "gorilla",
    "grasshopper",
    "hamster",
    "hare",
    "hedgehog",
    "hippopotamus",
    "hornbill",
    "horse",
    "hummingbird",
    "hyena",
    "jellyfish",
    "kangaroo",
    "koala",
    "ladybugs",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "mosquito",
    "moth",
    "mouse",
    "octopus",
    "okapi",
    "orangutan",
    "otter",
    "owl",
    "ox",
    "oyster",
    "panda",
    "parrot",
    "pelecaniformes",
    "penguin",
    "pig",
    "pigeon",
    "porcupine",
    "possum",
    "raccoon",
    "rat",
    "reindeer",
    "rhinoceros",
    "sandpiper",
    "seahorse",
    "seal",
    "shark",
    "sheep",
    "snake",
    "sparrow",
    "squid",
    "squirrel",
    "starfish",
    "swan",
    "tiger",
    "turkey",
    "turtle",
    "whale",
    "wolf",
    "wombat",
    "woodpecker",
    "zebra",
]

class PredictRequest(BaseModel):
    url: HttpUrl

class PredictResponse(BaseModel):
    predictions: dict[str, float]
    top_class: str
    top_probability: float

def predict(url: str):
    X = preprocessor.from_url(url)
    result = session.run([output_name], {input_name: X})
    float_predictions = result[0][0].tolist()
    predictions_dict = dict(zip(classes, float_predictions))
    
    top_class = max(predictions_dict, key=predictions_dict.get)
    top_probability = predictions_dict[top_class]
    
    return predictions_dict, top_class, top_probability

@app.get("/")
def root():
    return {"message": "Animals Classification Service"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    predictions, top_class, top_prob = predict(str(request.url))
    
    return PredictResponse(
        predictions=predictions,
        top_class=top_class,
        top_probability=top_prob
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
