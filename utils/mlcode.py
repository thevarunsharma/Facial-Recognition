import numpy as np
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
from pickle import load

with CustomObjectScope({'tf':tf}):
    model = load_model("./model/facerec.h5")

face_embs = np.load("./faces/face_embs.npy")
with open("./faces/nameIds.pickle", "r") as fh:
    names = load(fh)

def get_embeddings(img):
    emb = model.predict(img.reshape(1, *img.shape))
    return emb

def cosine_distance(org, cod):
    return cod.dot(org)/(np.linalg.norm(org)*np.linalg.norm(cod, axis=-1))

def recognize_face(face):
    emb = model.predict(face.reshape(1, *face.shape))[0]
    dists = cosine_distance(emb, face_embs)
    best_dist = dists.argmax()
    return names[best_dist] if dists[best_dist]>0.7 else None
