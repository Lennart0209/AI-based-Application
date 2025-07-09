import os
import tempfile
import uuid
import json
import numpy as np
from flask import (
    Flask, request, render_template, redirect, url_for,
    jsonify, session, send_from_directory, Response
)
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import openai
from jsonschema import validate, ValidationError

# ─── Konfiguration ─────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(
    api_key=openai.api_key,
    base_url="https://chat-ai.academiccloud.de/v1"
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {"pdf"}

SCHEMA_ASK = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "query":  {"type": "string"},
        "source": {"type": "string", "enum": ["1", "2", "both"]}
    },
    "required": ["query", "source"]
}

# ─── Hilfsfunktionen ────────────────────────────────────────────────────────────
def allowed_file(fn):
    return fn.lower().endswith(".pdf")

def extract_text(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, max_len=500):
    paras, chunks, buf = [p.strip() for p in text.split("\n\n") if p.strip()], [], ""
    for p in paras:
        if len(buf) + len(p) < max_len:
            buf += " " + p
        else:
            chunks.append(buf.strip())
            buf = p
    if buf:
        chunks.append(buf.strip())
    return chunks

# ─── Embedder ───────────────────────────────────────────────────────────────────
class Embedder:
    def __init__(self, docs):
        self.docs = docs
        self.vec = CountVectorizer().fit(docs)
        tfidf_matrix = self.vec.transform(docs).toarray()
        sentences = [d.split() for d in docs]
        self.w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=2)
        w2v_dim = self.w2v.vector_size
        w2v_matrix = np.zeros((len(docs), w2v_dim))
        for i, chunk in enumerate(docs):
            words = [w for w in chunk.split() if w in self.w2v.wv]
            if words:
                w2v_matrix[i] = np.mean([self.w2v.wv[w] for w in words], axis=0)
        self.tfidf_matrix = tfidf_matrix
        self.w2v_matrix   = w2v_matrix

    def embed(self, query):
        q_tfidf = self.vec.transform([query]).toarray()[0]
        words = [w for w in query.split() if w in self.w2v.wv]
        q_w2v = np.mean([self.w2v.wv[w] for w in words], axis=0) if words else np.zeros(self.w2v.vector_size)
        return q_tfidf, q_w2v

embedders = {}

# ─── Upload erste PDF ───────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not allowed_file(f.filename):
            return render_template("upload.html", error="Ungültige PDF.")
        fn = secure_filename(f.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], fn)
        f.save(path)

        chunks1 = chunk_text(extract_text(path))
        session["len1"]        = len(chunks1)
        session["filename"]    = fn
        session.pop("filename2", None)
        session["uid"]         = str(uuid.uuid4())
        session["histories"]   = {fn: []}
        session["current_pdf"] = fn
        embedders[session["uid"]] = Embedder(chunks1)

        return redirect(url_for("chat"))
    return render_template("upload.html")

# ─── Upload zweite PDF ──────────────────────────────────────────────────────────
@app.route("/upload2", methods=["POST"])
def upload2():
    f = request.files.get("file")
    if not f or not allowed_file(f.filename):
        return ("", 400)
    fn2 = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], fn2)
    f.save(path)

    chunks2 = chunk_text(extract_text(path))
    session["filename2"]      = fn2
    session["histories"][fn2] = []
    # session["current_pdf"] bleibt unverändert

    uid = session["uid"]
    first_chunks = embedders[uid].docs
    embedders[uid] = Embedder(first_chunks + chunks2)
    return ("", 204)

# ─── PDF-Viewer ────────────────────────────────────────────────────────────────
@app.route("/pdf/<filename>")
def pdf_view(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ─── Chat-Seite ─────────────────────────────────────────────────────────────────
@app.route("/chat")
def chat():
    if "uid" not in session or session["uid"] not in embedders:
        return redirect(url_for("upload"))
    selected = request.args.get("pdf", session.get("current_pdf"))
    session["current_pdf"] = selected
    history = session["histories"].get(selected, [])
    return render_template("chat.html", history=history, current=selected)

# ─── API: Frage beantworten ─────────────────────────────────────────────────────
@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True)
    try:
        validate(instance=data, schema=SCHEMA_ASK)
    except ValidationError as e:
        return jsonify(error=str(e)), 400

    query, source = data["query"], data["source"]
    uid = session["uid"]
    emb = embedders[uid]
    all_chunks = emb.docs
    len1 = session["len1"]

    qc, qw = emb.embed(query)
    def batch_scores(qt, qw_, tf, w2): return tf.dot(qt) + w2.dot(qw_)

    if source == "1":
        tfm, w2m = emb.tfidf_matrix[:len1], emb.w2v_matrix[:len1]
        idxs = np.argsort(-batch_scores(qc, qw, tfm, w2m))[:3]
        ctx_chunks = [all_chunks[i] for i in idxs]
    elif source == "2":
        tfm, w2m = emb.tfidf_matrix[len1:], emb.w2v_matrix[len1:]
        idxs = np.argsort(-batch_scores(qc, qw, tfm, w2m))[:3]
        ctx_chunks = [all_chunks[len1 + i] for i in idxs]
    else:
        mat1_tf, mat1_w2 = emb.tfidf_matrix[:len1], emb.w2v_matrix[:len1]
        mat2_tf, mat2_w2 = emb.tfidf_matrix[len1:], emb.w2v_matrix[len1:]
        idx1 = np.argsort(-batch_scores(qc, qw, mat1_tf, mat1_w2))[:3]
        idx2 = np.argsort(-batch_scores(qc, qw, mat2_tf, mat2_w2))[:3]
        top1 = [all_chunks[i] for i in idx1]
        top2 = [all_chunks[len1 + i] for i in idx2]
        ctx_chunks = ["=== PDF1 ===", *top1, "", "=== PDF2 ===", *top2]

    context = "\n---\n".join(ctx_chunks)
    prompt  = f"{context}\n\nFrage: {query}"

    system_msg = {
        "role": "system",
        "content": (
            "Du beantwortest präzise anhand des Kontextes. "
            "Bei \"both\" vergleiche PDF1 und PDF2."
        )
    }
    user_msg = {"role": "user", "content": prompt}

    resp = client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=[system_msg, user_msg],
        max_tokens=500,
        temperature=0.2
    )
    answer = resp.choices[0].message.content.strip()

    current = session["current_pdf"]
    hist = session["histories"].setdefault(current, [])
    hist += [{"role":"User","text":query},{"role":"Assistant","text":answer}]
    session.modified = True

    return jsonify(chat=hist)

# ─── Download Chat-Log ─────────────────────────────────────────────────────────
@app.route("/download")
def download_chat():
    hist = session["histories"].get(session["current_pdf"], [])
    data = json.dumps(hist, ensure_ascii=False, indent=2)
    return Response(data, mimetype="application/json",
                    headers={"Content-Disposition":"attachment;filename=chat_log.json"})

# ─── NEU: Download Key-Points ──────────────────────────────────────────────────
@app.route("/download_keypoints")
def download_keypoints():
    uid    = session["uid"]
    emb    = embedders[uid]
    chunks = emb.docs
    len1   = session["len1"]
    fn     = session["current_pdf"]

    # nur aktuelle PDF
    relevant = chunks[len1:] if fn==session.get("filename2") else chunks[:len1]
    context  = "\n\n".join(relevant[:5])

    system_msg = {
        "role":"system",
        "content":(
            "Extrahiere exakt diese Felder und gib nur das JSON\n"
            "- name\n- CO2\n- NOX\n- Number_of_Electric_Vehicles\n"
            "- Impact\n- Risks\n- Opportunities\n- Strategy\n"
            "- Actions\n- Targets"
        )
    }
    user_msg = {"role":"user","content":f"{context}\n\nGib nur das JSON zurück."}

    resp = client.chat.completions.create(
        model="meta-llama-3.1-8b-instruct",
        messages=[system_msg,user_msg],
        max_tokens=400,temperature=0
    )
    text = resp.choices[0].message.content.strip()
    try:
        obj = json.loads(text)
        data = json.dumps(obj, ensure_ascii=False, indent=2)
    except:
        data = json.dumps({"raw":text}, ensure_ascii=False, indent=2)

    return Response(data, mimetype="application/json",
                    headers={"Content-Disposition":f"attachment;filename={fn}_keypoints.json"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
