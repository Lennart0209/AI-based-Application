import os
import tempfile
import uuid
import json
import numpy as np
from flask import (
    Flask, request, jsonify, session,
    send_from_directory, Response
)
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF, um PDFs zu lesen
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import openai
from jsonschema import validate, ValidationError
import re

# API-Key aus Umgebungsvariable holen (für OpenAI)
openai.api_key = os.getenv("OPENAI_API_KEY")
# OpenAI Client mit speziellem Basis-URL (Academic Cloud)
client = openai.OpenAI(
    api_key=openai.api_key,
    base_url="https://chat-ai.academiccloud.de/v1"
)

# Flask App starten, statische Dateien direkt aus dem aktuellen Ordner servieren
app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = os.urandom(24)  # Zufälliger Schlüssel für Session-Cookies

# Temporärer Ordner zum Speichern der PDFs
app.config["UPLOAD_FOLDER"] = tempfile.gettempdir()

# Erlaubte Dateiendung (nur PDFs)
ALLOWED_EXTENSIONS = {"pdf"}

# Schema zur Validierung der Anfrage an /api/ask
SCHEMA_ASK = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "query":  {"type": "string"},              # Frage vom Nutzer
        "source": {"type": "string", "enum": ["1", "2", "both"]}  # Quelle: PDF 1, 2 oder beide
    },
    "required": ["query", "source"]
}

# Hilfsfunktion: Prüfen ob Datei erlaubt ist
def allowed_file(filename):
    return filename.lower().endswith(".pdf")

# PDF Text extrahieren (alle Seiten)
def extract_text(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

# Text in kleine Abschnitte splitten (max 500 Zeichen)
def chunk_text(text, max_len=500):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) < max_len:
            buf += " " + p
        else:
            if buf:
                chunks.append(buf.strip())
            buf = p
    if buf:
        chunks.append(buf.strip())
    return chunks

# Klasse zum Erzeugen von Text-Embeddings
class Embedder:
    def __init__(self, docs):
        self.docs = docs
        self.vec = CountVectorizer().fit(docs)          # Vektorisierer für TF-IDF
        self.tfidf_matrix = self.vec.transform(docs).toarray()
        sentences = [d.split() for d in docs]           # Worte splitten
        self.w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=2)  # Word2Vec Modell trainieren

        # Word2Vec Embeddings für jeden Abschnitt berechnen
        w2v_dim = self.w2v.vector_size
        w2v_matrix = np.zeros((len(docs), w2v_dim))
        for i, chunk in enumerate(docs):
            words = [w for w in chunk.split() if w in self.w2v.wv]
            if words:
                w2v_matrix[i] = np.mean([self.w2v.wv[w] for w in words], axis=0)
        self.w2v_matrix = w2v_matrix

    # Embeddings für die Nutzereingabe erzeugen (TF-IDF + Word2Vec)
    def embed(self, query):
        q_tfidf = self.vec.transform([query]).toarray()[0]
        words = [w for w in query.split() if w in self.w2v.wv]
        q_w2v = np.mean([self.w2v.wv[w] for w in words], axis=0) if words else np.zeros(self.w2v.vector_size)
        return q_tfidf, q_w2v

# Hier speichern wir für jede Session den Embedder (für die PDFs)
embedders = {}

@app.route("/")
def index():
    # Startseite (HTML)
    return app.send_static_file("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    # Upload erste PDF
    file = request.files.get("file")
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Ungültige PDF."}), 400
    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # Text aus PDF extrahieren und in Chunks splitten
    chunks1 = chunk_text(extract_text(path))
    session["len1"] = len(chunks1)       # Anzahl der Chunks der ersten PDF merken
    session["filename"] = filename
    session.pop("filename2", None)        # Falls zweite PDF geladen, löschen wir die
    session["uid"] = str(uuid.uuid4())   # Neue Session-ID (für den Embedder)
    session["histories"] = {filename: []} # Chat-Historie für erste PDF starten
    session["current_pdf"] = filename
    embedders[session["uid"]] = Embedder(chunks1)  # Embedder erzeugen

    return jsonify({"filename": filename})

@app.route("/upload2", methods=["POST"])
def upload2():
    # Upload zweite PDF (ähnlich wie erste)
    file = request.files.get("file")
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Ungültige PDF."}), 400
    filename2 = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename2)
    file.save(path)

    # Text aus zweiter PDF
    chunks2 = chunk_text(extract_text(path))
    session["filename2"] = filename2
    session["histories"][filename2] = []

    uid = session["uid"]
    first_chunks = embedders[uid].docs   # Erste Chunks holen
    embedders[uid] = Embedder(first_chunks + chunks2)  # Embedder mit beiden PDFs neu bauen
    return jsonify({"filename": filename2})

@app.route("/pdf/<filename>")
def pdf_view(filename):
    # Liefert die PDF Datei zum Browser (für Vorschau)
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/api/ask", methods=["POST"])
def api_ask():
    # API Endpoint für Fragen an die PDFs
    data = request.get_json(force=True)
    try:
        validate(instance=data, schema=SCHEMA_ASK)  # Anfrage validieren
    except ValidationError as e:
        return jsonify(error=str(e)), 400

    query, source = data["query"], data["source"]
    uid = session["uid"]
    emb = embedders[uid]
    all_chunks = emb.docs
    len1 = session["len1"]

    # Embeddings für die Frage erzeugen
    qc, qw = emb.embed(query)

    # Funktion für die Relevanzbewertung der Chunks
    def batch_scores(qt, qw_, tf, w2): return tf.dot(qt) + w2.dot(qw_)

    # Je nach Quelle relevante Chunks bestimmen
    if source == "1":
        tfm, w2m = emb.tfidf_matrix[:len1], emb.w2v_matrix[:len1]
        idxs = np.argsort(-batch_scores(qc, qw, tfm, w2m))[:3]
        ctx_chunks = [all_chunks[i] for i in idxs]
    elif source == "2":
        tfm, w2m = emb.tfidf_matrix[len1:], emb.w2v_matrix[len1:]
        idxs = np.argsort(-batch_scores(qc, qw, tfm, w2m))[:3]
        ctx_chunks = [all_chunks[len1 + i] for i in idxs]
    else:  # beide PDFs
        mat1_tf, mat1_w2 = emb.tfidf_matrix[:len1], emb.w2v_matrix[:len1]
        mat2_tf, mat2_w2 = emb.tfidf_matrix[len1:], emb.w2v_matrix[len1:]
        idx1 = np.argsort(-batch_scores(qc, qw, mat1_tf, mat1_w2))[:3]
        idx2 = np.argsort(-batch_scores(qc, qw, mat2_tf, mat2_w2))[:3]
        top1 = [all_chunks[i] for i in idx1]
        top2 = [all_chunks[len1 + i] for i in idx2]
        ctx_chunks = ["=== PDF1 ===", *top1, "", "=== PDF2 ===", *top2]

    # Kontext für Prompt zusammensetzen
    context = "\n---\n".join(ctx_chunks)
    prompt = f"{context}\n\nFrage: {query}"

    # Systemnachricht an AI
    system_msg = {
        "role": "system",
        "content": "Du beantwortest präzise anhand des Kontextes. Bei \"both\" vergleiche PDF1 und PDF2."
    }
    user_msg = {"role": "user", "content": prompt}

    # Anfrage an OpenAI Modell
    resp = client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=[system_msg, user_msg],
        max_tokens=500,
        temperature=0.2
    )
    answer = resp.choices[0].message.content.strip()

    # Chat Verlauf speichern
    current = session["current_pdf"]
    hist = session["histories"].setdefault(current, [])
    hist += [{"role": "User", "text": query}, {"role": "Assistant", "text": answer}]
    session.modified = True

    return jsonify(chat=hist)

@app.route("/download")
def download_chat():
    # Chat Verlauf als JSON zum Download anbieten
    hist = session["histories"].get(session["current_pdf"], [])
    data = json.dumps(hist, ensure_ascii=False, indent=2)
    return Response(data, mimetype="application/json",
                    headers={"Content-Disposition": "attachment;filename=chat_log.json"})

@app.route("/download_keypoints")
def download_keypoints():
    # Keypoints aus PDF als JSON herunterladen
    uid = session["uid"]
    emb = embedders[uid]
    chunks = emb.docs
    len1 = session["len1"]
    fn = session["current_pdf"]

    # Je nachdem ob erste oder zweite PDF, relevante Chunks nehmen
    relevant = chunks[len1:] if fn == session.get("filename2") else chunks[:len1]
    context = "\n\n".join(relevant[:5])  # Nur ersten 5 Abschnitte nehmen

    # System Nachricht: AI soll reines JSON mit bestimmten Feldern zurückgeben
    system_msg = {
        "role": "system",
        "content": (
            "Du bist ein präziser JSON-Generator. "
            "Extrahiere ausschließlich die folgenden Felder aus dem gegebenen Text und gib _nur_ das reine JSON zurück, "
            "ohne jeglichen zusätzlichen Text, Erklärungen oder Markdown-Formatierung:\n"
            "- name\n"
            "- CO2\n"
            "- NOX\n"
            "- Number_of_Electric_Vehicles\n"
            "- Impact\n"
            "- Risks\n"
            "- Opportunities\n"
            "- Strategy\n"
            "- Actions\n"
            "- Targets\n"
        )
    }
    user_msg = {
        "role": "user",
        "content": f"{context}\n\nGib ausschließlich das JSON zurück, ohne Text oder Formatierungen."
    }

    resp = client.chat.completions.create(
        model="llama-3.3-70b-instruct",
        messages=[system_msg, user_msg],
        max_tokens=400,
        temperature=0
    )
    text = resp.choices[0].message.content.strip()

    # JSON aus der Antwort extrahieren (Markdown-Codeblock entfernen falls da)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if match:
        json_text = match.group(1)
    else:
        json_text = text

    # Prüfen ob gültiges JSON, sonst rohen Text zurückgeben
    try:
        obj = json.loads(json_text)
        data = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        data = json.dumps({"raw": text}, ensure_ascii=False, indent=2)

    return Response(data, mimetype="application/json",
                    headers={"Content-Disposition": f"attachment;filename={fn}_keypoints.json"})

if __name__ == "__main__":
    # Server starten, Port 5000, Debug-Modus an
    app.run(debug=True, port=5000)
