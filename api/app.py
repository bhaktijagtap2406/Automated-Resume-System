from flask import Flask, request, jsonify
from flask import send_from_directory
from flask_cors import CORS
import joblib
import re
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ── App setup ─────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder   = "../frontend",
    template_folder = "../frontend"
)
CORS(app)

# ── Load model once at startup ────────────────────────────
BASE = os.path.dirname(
       os.path.dirname(
       os.path.abspath(__file__)))

MODEL   = joblib.load(
    os.path.join(BASE,
    "models/resume_classifier.pkl"))

ENCODER = joblib.load(
    os.path.join(BASE,
    "models/label_encoder.pkl"))

# ── Load resume data ──────────────────────────────────────
DATA = pd.read_csv(
    os.path.join(BASE,
    "data/processed/resumes_nlp.csv"),
    encoding = "utf-8")
DATA["NLP_Resume"]   = DATA["NLP_Resume"].fillna("")
DATA["Clean_Resume"] = DATA["Clean_Resume"].fillna("")
DATA["Category"]     = DATA["Category"].fillna("")
DATA["Years_Exp"]    = DATA["Years_Exp"].fillna(0)
DATA["Email"]        = DATA["Email"].fillna("")

print(f"Model loaded!")
print(f"Data loaded: {len(DATA)} resumes")
print(f"Categories : {DATA['Category'].nunique()}")

# ── Helper: clean text ────────────────────────────────────
def clean(text):
    text = str(text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().strip()


# ════════════════════════════════════════════════════════
#  ENDPOINT 1 — Health Check
#  GET /api/health
# ════════════════════════════════════════════════════════
@app.route("/api/health")
def health():
    return jsonify({
        "status":     "ok",
        "candidates": len(DATA),
        "categories": int(
            DATA["Category"].nunique()),
        "model":      "resume_classifier.pkl"
    })


# ════════════════════════════════════════════════════════
#  ENDPOINT 2 — Predict Category
#  POST /api/predict
#  Body: { "resume_text": "..." }
# ════════════════════════════════════════════════════════
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("resume_text", "").strip()

    if not text:
        return jsonify(
            {"error": "resume_text is required"}
        ), 400

    try:
        cleaned  = clean(text)
        pred     = MODEL.predict([cleaned])[0]
        proba    = MODEL.predict_proba([cleaned])[0]
        category = ENCODER.inverse_transform(
                       [pred])[0]
        confidence = round(
            float(proba.max()) * 100, 1)

        # Top 3 predictions
        top3_idx = proba.argsort()[::-1][:3]
        top3 = [
            {
                "category": ENCODER.inverse_transform(
                                [i])[0],
                "confidence": round(
                    float(proba[i]) * 100, 1)
            }
            for i in top3_idx
        ]

        return jsonify({
            "category":   category,
            "confidence": confidence,
            "top3":       top3
        })

    except Exception as e:
        return jsonify(
            {"error": str(e)}), 500


# ════════════════════════════════════════════════════════
#  ENDPOINT 3 — Screen Candidates
#  POST /api/screen
#  Body: { "job_description": "...",
#           "top_n": 10,
#           "category_filter": "HR" }
# ════════════════════════════════════════════════════════
@app.route("/api/screen", methods=["POST"])
def screen():
    data     = request.get_json(silent=True) or {}
    jd       = data.get(
        "job_description", "").strip()
    top_n    = int(data.get("top_n", 10))
    category = data.get("category_filter")

    if not jd:
        return jsonify(
            {"error": "job_description is required"}
        ), 400

    try:
        df = DATA.copy()

        # Filter by category if given
        if category:
            df = df[df["Category"] == category]

        if len(df) == 0:
            return jsonify(
                {"error": "No candidates found"}
            ), 404

        # Cosine similarity ranking
        clean_jd  = clean(jd)
        all_texts = ([clean_jd] +
                     df["NLP_Resume"].tolist())

        vec = TfidfVectorizer(
            sublinear_tf = True,
            min_df       = 1)
        mat = vec.fit_transform(all_texts)

        scores     = cosine_similarity(
            mat[0:1], mat[1:]).flatten()
        df         = df.copy()
        df["score"] = scores

        top     = df.nlargest(top_n, "score")
        results = []

        for rank, (_, row) in enumerate(
                top.iterrows(), 1):
            results.append({
                "rank":      rank,
                "category":  row["Category"],
                "score":     round(
                    float(row["score"])*100, 2),
                "years_exp": int(
                    row.get("Years_Exp", 0)),
                "email":     str(
                    row.get("Email", "")),
                "preview":   str(
                    row.get("Clean_Resume","")
                )[:250]
            })

        return jsonify({
            "job_description": jd[:100],
            "total_screened":  len(df),
            "top_n":           top_n,
            "results":         results
        })

    except Exception as e:
        return jsonify(
            {"error": str(e)}), 500


# ════════════════════════════════════════════════════════
#  ENDPOINT 4 — Upload Resume File
#  POST /api/upload
#  Form: file = resume.pdf / .docx / .txt
# ════════════════════════════════════════════════════════
@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify(
            {"error": "No file uploaded"}), 400

    f   = request.files["file"]
    ext = f.filename.rsplit(".", 1)[-1].lower()

    try:
        # Extract text based on file type
        if ext == "pdf":
            import fitz
            doc  = fitz.open(
                stream   = f.read(),
                filetype = "pdf")
            text = " ".join([
                page.get_text()
                for page in doc])

        elif ext == "docx":
            from docx import Document
            import io
            doc  = Document(io.BytesIO(f.read()))
            text = " ".join([
                para.text
                for para in doc.paragraphs])

        elif ext == "txt":
            text = f.read().decode("utf-8")

        else:
            return jsonify({
                "error": "Use PDF, DOCX or TXT"
            }), 400

        if len(text.strip()) < 30:
            return jsonify({
                "error": "File is empty or unreadable"
            }), 400

        # Predict
        cleaned    = clean(text)
        pred       = MODEL.predict([cleaned])[0]
        proba      = MODEL.predict_proba(
                         [cleaned])[0]
        category   = ENCODER.inverse_transform(
                         [pred])[0]
        confidence = round(
            float(proba.max()) * 100, 1)

        # Top 3
        top3_idx = proba.argsort()[::-1][:3]
        top3 = [
            {
                "category": ENCODER.inverse_transform(
                                [i])[0],
                "confidence": round(
                    float(proba[i])*100, 1)
            }
            for i in top3_idx
        ]

        return jsonify({
            "filename":   f.filename,
            "category":   category,
            "confidence": confidence,
            "top3":       top3,
            "preview":    text[:400],
            "word_count": len(text.split())
        })

    except Exception as e:
        return jsonify(
            {"error": str(e)}), 500


# ════════════════════════════════════════════════════════
#  ENDPOINT 5 — Get All Categories
#  GET /api/categories
# ════════════════════════════════════════════════════════
@app.route("/api/categories")
def categories():
    counts = DATA["Category"].value_counts()
    return jsonify([
        {
            "category": str(k),
            "count":    int(v)
        }
        for k, v in counts.items()
    ])


# ════════════════════════════════════════════════════════
#  ENDPOINT 6 — Dashboard Stats
#  GET /api/stats
# ════════════════════════════════════════════════════════
@app.route("/api/stats")
def stats():
    return jsonify({
        "total_resumes":    len(DATA),
        "total_categories": int(
            DATA["Category"].nunique()),
        "avg_word_count":   round(
            float(DATA["Word_Count"].mean()), 1),
        "avg_experience":   round(
            float(DATA["Years_Exp"].mean()), 1),
        "top_category":     str(
            DATA["Category"].value_counts(
            ).index[0])
    })


# ════════════════════════════════════════════════════════
#  Serve Frontend
#  GET /
# ════════════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory(
        "../frontend", "index.html")


# ════════════════════════════════════════════════════════
#  Error Handlers
# ════════════════════════════════════════════════════════
@app.errorhandler(404)
def not_found(_):
    return jsonify(
        {"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify(
        {"error": str(e)}), 500

@app.errorhandler(413)
def too_large(_):
    return jsonify(
        {"error": "File too large (max 5MB)"}
    ), 413


# ════════════════════════════════════════════════════════
#  Run App
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*45)
    print("   Resume Screening API")
    print("="*45)
    print("  URL : http://127.0.0.1:5000")
    print("  UI  : http://127.0.0.1:5000")
    print("="*45 + "\n")
    app.run(
        debug = True,
        host  = "0.0.0.0",
        port  = 5000)