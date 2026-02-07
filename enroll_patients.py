import os
from pathlib import Path
from deepface import DeepFace
from pymongo import MongoClient

def load_env(path: str = ".env") -> None:
    file_path = Path(__file__).resolve().parent / path
    if not file_path.exists():
        return
    for line in file_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# Setup MongoDB Connection
load_env()
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client['Users']  # Database name
patients_col = db['Galit_Tauber'] # Collection name

def enroll_patients_to_db(db_path):
    """Scans the face_db folder and saves embeddings to Mongo."""
    for person_name in os.listdir(db_path):
        person_dir = os.path.join(db_path, person_name)
        if os.path.isdir(person_dir):
            img_path = os.path.join(person_dir, os.listdir(person_dir)[0])
            
            # Generate the 512-number vector
            # Facenet512 is the 2026 standard for high-accuracy medical use
            embedding_objs = DeepFace.represent(img_path=img_path, model_name="Facenet512")
            embedding = embedding_objs[0]["embedding"]
            
            # Save to MongoDB
            patients_col.update_one(
                {"name": person_name},
                {"$set": {"face_vector": embedding, "last_updated": "2026-02-07"}},
                upsert=True
            )
            print(f"✅ Enrolled {person_name} into MongoDB")

# Run it
enroll_patients_to_db("./face_db")