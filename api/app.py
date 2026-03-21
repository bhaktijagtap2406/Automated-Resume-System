
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, re, os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app  = Flask(__name__,
             static_folder="../frontend",
             template_folder="../frontend")
CORS(app)

# Load model
MODEL   = joblib.load("models/resume_classifier.pkl")
ENCODER = joblib.load("models/label_encoder.pkl")

# Load data
DATA = pd.read_csv(
    "data/processed/resumes_nlp.csv")
DATA["NLP_Resume"] = DATA["NLP_Resume"].fillna("")

def clean(text):
    text = str(text)
    text = re.sub(r"[^a-zA-Z\s]"," ",text)
    return text.lower().strip()

# ── 1. Health check ───────────────────────────
@app.route("/api/health")
def health():
    return jsonify({
        "status":     "ok",
        "candidates": len(DATA),
        "categories": DATA["Category"].nunique()
    })

# ── 2. Predict category ───────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("resume_text","")
    if not text:
        return jsonify(
            {"error": "No text provided"}), 400
    cleaned  = clean(text)
    pred     = MODEL.predict([cleaned])[0]
    proba    = MODEL.predict_proba([cleaned])[0]
    category = ENCODER.inverse_transform(
                   [pred])[0]
    confidence = round(
        float(proba.max())*100, 1)
    top3_idx = proba.argsort()[::-1][:3]
    top3 = [
        {"category": ENCODER.inverse_transform(
                         [i])[0],
         "confidence": round(
             float(proba[i])*100,1)}
        for i in top3_idx]
    return jsonify({
        "category":   category,
        "confidence": confidence,
        "top3":       top3})

# ── 3. Screen candidates ──────────────────────
@app.route("/api/screen", methods=["POST"])
def screen():
    data     = request.get_json()
    jd       = data.get("job_description","")
    top_n    = int(data.get("top_n", 10))
    category = data.get("category_filter")
    if not jd:
        return jsonify(
            {"error": "No JD provided"}), 400
    df = DATA.copy()
    if category:
        df = df[df["Category"]==category]
    clean_jd  = clean(jd)
    all_texts = ([clean_jd] +
                 df["NLP_Resume"].tolist())
    vec = TfidfVectorizer(
        sublinear_tf=True, min_df=1)
    mat = vec.fit_transform(all_texts)
    scores = cosine_similarity(
        mat[0:1], mat[1:]).flatten()
    df        = df.copy()
    df["score"] = scores
    top       = df.nlargest(top_n,"score")
    results   = []
    for rank, (_, row) in enumerate(
            top.iterrows(), 1):
        results.append({
            "rank":       rank,
            "category":   row["Category"],
            "score":      round(
                float(row["score"])*100,2),
            "years_exp":  int(
                row.get("Years_Exp",0)),
            "email":      str(
                row.get("Email","")),
            "preview":    str(
                row.get("Clean_Resume",
                        ""))[:200]
        })
    return jsonify({
        "total_screened": len(df),
        "results":        results})

# ── 4. Upload resume ──────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify(
            {"error": "No file"}), 400
    f    = request.files["file"]
    ext  = f.filename.rsplit(".",1)[-1].lower()
    text = ""
    if ext == "pdf":
        import fitz
        doc  = fitz.open(
            stream=f.read(), filetype="pdf")
        text = " ".join([
            p.get_text() for p in doc])
    elif ext == "docx":
        from docx import Document
        import io
        doc  = Document(io.BytesIO(f.read()))
        text = " ".join([
            p.text for p in doc.paragraphs])
    elif ext == "txt":
        text = f.read().decode("utf-8")
    else:
        return jsonify(
            {"error": "Use PDF/DOCX/TXT"}), 400
    cleaned  = clean(text)
    pred     = MODEL.predict([cleaned])[0]
    proba    = MODEL.predict_proba([cleaned])[0]
    category = ENCODER.inverse_transform(
                   [pred])[0]
    return jsonify({
        "filename":   f.filename,
        "category":   category,
        "confidence": round(
            float(proba.max())*100,1),
        "preview":    text[:300]})

# ── 5. Categories ─────────────────────────────
@app.route("/api/categories")
def categories():
    counts = DATA["Category"].value_counts()
    return jsonify([
        {"category": k, "count": int(v)}
        for k,v in counts.items()])

# ── Serve frontend ────────────────────────────
from flask import send_from_directory
@app.route("/")
def index():
    return send_from_directory(
        "../frontend","index.html")

if __name__ == "__main__":
    print("\n API running at "
          "http://127.0.0.1:5000\n")
    app.run(debug=True,
            host="0.0.0.0",
            port=5000)
