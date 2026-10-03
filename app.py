from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from pathlib import Path
import json
import uuid

from document.reader import extract_text
from document.chunker import chunk_text
from ai.generator import LocalAI
from rag.retriever import Retriever

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

SOURCES_FILE = DATA_DIR / "sources.json"

app = Flask(__name__)
app.secret_key = "epoch-local-secret-key"

ai = LocalAI()
retriever = Retriever(DATA_DIR / "index.json")


def load_sources():
    if not SOURCES_FILE.exists():
        return []
    try:
        return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_sources(sources):
    SOURCES_FILE.write_text(
        json.dumps(sources, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# --- Multi-page Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sources")
def sources_page():
    return render_template(
        "sources.html",
        sources=load_sources()
    )


@app.route("/workspace")
def workspace_page():
    return render_template(
        "workspace.html",
        model=ai.model_name
    )


# --- Upload & Document Management Routes ---

@app.route("/upload", methods=["POST"])
def upload():
    uploaded = request.files.get("file")

    if not uploaded or not uploaded.filename:
        flash("Choose a PDF or TXT file first.", "error")
        return redirect(url_for("sources_page"))

    extension = Path(uploaded.filename).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        flash("Epoch currently supports PDF and TXT files.", "error")
        return redirect(url_for("sources_page"))

    source_id = uuid.uuid4().hex
    safe_name = Path(uploaded.filename).name
    stored_name = f"{source_id}{extension}"
    path = UPLOAD_DIR / stored_name
    uploaded.save(path)

    try:
        text = extract_text(path)
        if not text.strip():
            raise ValueError("No readable text was found in this file.")

        chunks = chunk_text(text)

        retriever.add_document(
            source_id=source_id,
            filename=safe_name,
            chunks=chunks
        )

        sources = load_sources()
        sources.append({
            "id": source_id,
            "filename": safe_name,
            "stored_name": stored_name,
            "chunks": len(chunks),
            "characters": len(text)
        })
        save_sources(sources)

        flash(f"Added {safe_name} — {len(chunks)} searchable chunks created.", "success")

    except Exception as exc:
        if path.exists():
            path.unlink()
        flash(f"Could not process the document: {exc}", "error")

    return redirect(url_for("sources_page"))


@app.route("/delete/", methods=["POST"])
def delete_source(source_id):
    sources = load_sources()
    source = next((s for s in sources if s["id"] == source_id), None)

    if source:
        path = UPLOAD_DIR / source["stored_name"]
        if path.exists():
            path.unlink()

        sources = [s for s in sources if s["id"] != source_id]
        save_sources(sources)
        retriever.remove_document(source_id)
        flash("Source removed.", "success")

    return redirect(url_for("sources_page"))


# --- AI Features & Query Routes ---

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()

    if not question:
        return jsonify({"error": "Enter a question."}), 400

    results = retriever.search(question, top_k=5)

    if not results:
        return jsonify({
            "answer": "Upload a source first so Epoch has material to search."
        })

    context = "\n\n".join(
        f"[Source: {item['filename']}]\n{item['text']}"
        for item in results
    )

    answer = ai.answer_question(question, context)

    return jsonify({
        "answer": answer,
        "sources": [
            {
                "filename": item["filename"],
                "score": round(item["score"], 3)
            }
            for item in results
        ]
    })


@app.route("/generate", methods=["POST"])
def generate():
    action = request.form.get("action", "")
    source_ids = request.form.getlist("source_ids")

    if not source_ids:
        source_ids = [s["id"] for s in load_sources()]

    selected = retriever.get_chunks_for_sources(source_ids)

    if not selected:
        return jsonify({"error": "Upload and select at least one source."}), 400

    context = "\n\n".join(
        f"[{item['filename']}]\n{item['text']}"
        for item in selected
    )[:18000]

    try:
        if action == "summary":
            result = ai.summary(context)
        elif action == "notes":
            result = ai.study_notes(context)
        elif action == "flashcards":
            result = ai.flashcards(context)
        elif action == "questions":
            result = ai.practice_questions(context)
        else:
            return jsonify({"error": "Unknown generation type."}), 400

        return jsonify({"result": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)