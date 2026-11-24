from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from PIL import Image
import io
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- YAPAY ZEKA MODELLERİNİ YÜKLEME ---
# 1. Resim Analizi İçin (Hata vermeyen, garantili yöntem)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
image_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2. Metin Sınıflandırma İçin (İçeriği kategorize eder)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

UPLOAD_DIR = "uploads"
DB_FILE = "database.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)
if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

def read_db():
    with open(DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def write_db(data):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(contents)

    ai_description = ""
    ai_category = "Genel"

    # --- RESİM ANALİZİ ---
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Resmi aç ve yapay zekanın anlayacağı RGB formatına çevir
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Yapay zeka resme bakıyor...
        inputs = processor(image, return_tensors="pt")
        out = image_model.generate(**inputs)
        ai_description = processor.decode(out[0], skip_special_tokens=True)
        
        ai_category = "Görsel"

    # --- METİN/İÇERİK ANALİZİ ---
    elif file.filename.lower().endswith(('.txt')):
        text_content = contents.decode("utf-8")
        # Olası kategorilerimiz
        candidate_labels = ["finans", "eğitim", "sağlık", "teknoloji"]
        # Yapay zeka metni okuyor...
        result = classifier(text_content, candidate_labels)
        ai_category = result['labels'][0] # En yüksek olasılıklı kategori
        ai_description = text_content[:50] + "..." # İçerikten kısa bir özet

    # Veritabanına kaydet
    db = read_db()
    db.append({
        "filename": file.filename, 
        "category": ai_category, 
        "description": ai_description
    })
    write_db(db)
    
    return {"status": "success", "category": ai_category, "description": ai_description}

@app.get("/files")
def list_files():
    return read_db()

@app.get("/search")
def search_files(q: str = ""):
    db = read_db()
    # Hem dosya adında hem de AI'nın oluşturduğu açıklamada arama yapar
    return [f for f in db if q.lower() in f['filename'].lower() or q.lower() in f['description'].lower()]