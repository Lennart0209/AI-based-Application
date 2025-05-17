import os
import tempfile
import uuid
from flask import Flask, request, render_template, redirect, url_for, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from jsonschema import validate, ValidationError

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Environment variable OPENAI_API_KEY is not set!")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# In-memory cache for Embedder instances per session
embedders = {}

# Flask App Setup
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}

# JSON Schema for /api/ask
SCHEMA_ASK = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"]
}

# Helpers
def allowed_file(filename):
    return filename.lower().endswith('.pdf')

def extract_text(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, max_len=500):
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks, buffer = [], ''
    for p in paras:
        if len(buffer) + len(p) < max_len:
            buffer += ' ' + p
        else:
            chunks.append(buffer.strip())
            buffer = p
    if buffer:
        chunks.append(buffer.strip())
    return chunks

class Embedder:
    def __init__(self, docs):
        self.docs = docs
        self.vec = CountVectorizer().fit(docs)
        sentences = [d.split() for d in docs]
        self.w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=2)

    def embed(self, doc):
        v1 = self.vec.transform([doc]).toarray()[0]
        words = [w for w in doc.split() if w in self.w2v.wv]
        v2 = sum(self.w2v.wv[w] for w in words) / len(words) if words else [0]*self.w2v.vector_size
        return v1, v2

# Routes
@app.route('/')
def index():
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return render_template('upload.html', error='Bitte lade eine gültige PDF-Datei hoch.')
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        text = extract_text(path)
        chunks = chunk_text(text)
        embedder = Embedder(chunks)

        session_id = str(uuid.uuid4())
        session['uid'] = session_id
        session['filename'] = filename
        session['chat'] = []
        embedders[session_id] = embedder
        return redirect(url_for('chat'))
    return render_template('upload.html')

@app.route('/pdf/<filename>')
def pdf_view(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/chat')
def chat():
    uid = session.get('uid')
    if not uid or uid not in embedders:
        return redirect(url_for('upload'))
    return render_template('chat.html')

@app.route('/api/ask', methods=['POST'])
def api_ask():
    try:
        data = request.get_json(force=True)
        validate(instance=data, schema=SCHEMA_ASK)
    except (ValidationError, Exception) as e:
        return jsonify(error=str(e)), 400

    uid = session.get('uid')
    embedder = embedders.get(uid)
    if not embedder:
        return jsonify(error='Kein Dokument geladen'), 400

    query = data['query']
    qc, qw = embedder.embed(query)
    scores = []
    for chunk in embedder.docs:
        cc = embedder.vec.transform([chunk]).toarray()[0]
        words = [w for w in chunk.split() if w in embedder.w2v.wv]
        cw = sum(embedder.w2v.wv[w] for w in words)/len(words) if words else [0]*embedder.w2v.vector_size
        scores.append(cosine_similarity([qc],[cc])[0][0] + cosine_similarity([qw],[cw])[0][0])

    # Top 3 Chunks verwenden
    top_n = 3
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    top_chunks = [embedder.docs[i] for i in top_idxs]

    # Prompt-Engineering für präzise Antworten
    system_msg = {
        "role": "system",
        "content": (
            "You are a precise, concise assistant. "
            "Answer *only* based on the provided excerpts in no more than 5 sentences."
        )
    }
    user_msg = {
        "role": "user",
        "content": (
            "Use these excerpts to answer the question:\n---\n" +
            "\n---\n".join(top_chunks) +
            f"\n\nQuestion: {query}"
        )
    }

    resp = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[system_msg, user_msg],
        max_tokens=200,
        temperature=0.2
    )
    answer = resp.choices[0].message.content.strip()

    chat = session['chat']
    chat.append({'role': 'User', 'text': query})
    chat.append({'role': 'Assistant', 'text': answer})
    session['chat'] = chat
    return jsonify(chat=chat)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
