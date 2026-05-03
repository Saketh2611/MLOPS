# upload_model.py
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def upload_model(pkl_path: str, filename: str):
    with open(pkl_path, "rb") as f:
        data = f.read()

    response = client.storage.from_("models").upload(
        path=filename,
        file=data,
        file_options={"content-type": "application/octet-stream", "upsert": "true"}
    )
    print(f"Uploaded → {filename}")
    return response

# Upload all 3 models
upload_model("models/best_model.pkl", "best_model.pkl")