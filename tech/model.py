import os, uuid, math

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tech.helper import *

model = load_model("./data/model1_fold_batch32_epoch20_lr0.0001.h5", compile=False)
model_dua = load_model("./data/model2_fold_batch32_epoch20_lr0.0001.h5", compile=False)
model_tiga = load_model("./data/model3_fold_batch64_epoch20_lr0.001.h5",  compile=False)
model_empat = load_model("./data/model4_fold_batch32_epoch10_lr0.001.h5",  compile=False)

tokener = token("./data/tokenizer_1.pickle")

pada = ["Perorangan", "Kelompok"]

tentang = ["Religi", "Ras", "Fisik", "Gender", "Lainnya"]

tingkat = ["Lemah", "Sedang", "Kuat"]

def presentences(st):
    text = [helper(st)]
    seq = tokener.texts_to_sequences(text)
    seq = pad_sequences(seq)
    return seq

def pada_model(st):
    seq = presentences(st)
    hasil = model_dua.predict(seq, verbose=0)[0]
    i = tf.math.argmax(hasil).numpy()
    return [pada[i], hasil[i]]

def tentang_model(st):
    seq = presentences(st)
    hasil = model_tiga.predict(seq, verbose=0)[0]
    i = tf.math.argmax(hasil).numpy()
    return [tentang[i], hasil[i]]

def tingkat_model(st):
    seq = presentences(st)
    hasil = model_empat.predict(seq, verbose=0)[0]
    i = tf.math.argmax(hasil).numpy()
    return [tingkat[i], hasil[i]]

def full_predict(st):
    hasil_kepada = pada_model(st)
    kepada = {"Kepada":hasil_kepada[0], "Nilai":str(hasil_kepada[1])}
    hasil_tentang = tentang_model(st)
    tentang = {"Tentang":hasil_tentang[0], "Nilai":str(hasil_tentang[1])}
    hasil_tingkat = tingkat_model(st)
    tingkat = {"Tingkat":hasil_tingkat[0], "Nilai":str(hasil_tingkat[1])}
    return {
        "Hasil":[kepada, tentang, tingkat]
    }

def model_predict(st : str):
    text = [helper(st)]
    seq = tokener.texts_to_sequences(text)
    seq = pad_sequences(seq)
    
    logits = model.predict(seq, verbose=0)
    
    predictor = round(logits[0][0])

    if predictor == 0:
        return {"hasil":predictor}
    else:
        hasil = full_predict(st)
    return hasil
