# chatbot/model.py
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "unsupervised_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "unsupervised_vectorizer.pkl")


def build_and_train_model(train_texts, n_clusters=6):
    """
    Entrena un modelo no supervisado (KMeans) para agrupar frases similares.
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_texts)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Modelo no supervisado entrenado y guardado correctamente.")
    return model, vectorizer


def load_model():
    """
    Carga el modelo y vectorizador si existen, o None si no.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        print("📂 Modelo cargado desde disco.")
        return model, vectorizer
    else:
        print("⚠️ No hay modelo guardado. Será necesario entrenarlo.")
        return None, None


def predict_cluster(model, vectorizer, user_text):
    """
    Predice el grupo (cluster) al que pertenece una nueva frase.
    """
    X = vectorizer.transform([user_text])
    cluster = model.predict(X)[0]
    return cluster
