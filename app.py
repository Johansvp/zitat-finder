import re

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Zitat-Finder", layout="wide")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# APA-Stil Zitatklammer, z.B. (Hamprecht, 2025, S. 119; Kochhan & Cichecki, 2024, S. 73)
CITATION_BLOCK = re.compile(r"\(([^()]*?\d{4}[^()]*?S\.?\s*\d+[^()]*?)\)")
CITATION_ENTRY = re.compile(r"([^,;]+),\s*(\d{4}),\s*S\.?\s*(\d+)")
SENTENCE_BOUNDARY = re.compile(r"[.!?]\s|\)\s")


@st.cache_resource(show_spinner="Lade Sprachmodell (nur beim ersten Mal, danach gecacht)...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


def extract_quotes(pdf_bytes: bytes):
    """Findet APA-Zitatklammern wie (Autor, Jahr, S. Seite) und nimmt den davor
    stehenden Satz als die zitierte Aussage. Eine Klammer mit mehreren durch ';'
    getrennten Quellen ergibt mehrere Eintraege mit demselben Aussage-Text."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = ""
    page_ranges = []  # (start_offset, end_offset, page_num) im hochgeladenen Dokument
    for page_num, page in enumerate(doc, start=1):
        t = re.sub(r"\s+", " ", page.get_text())
        start = len(full_text)
        full_text += t + " "
        page_ranges.append((start, len(full_text), page_num))
    doc.close()

    def doc_page_for_offset(pos):
        for start, end, page_num in page_ranges:
            if start <= pos < end:
                return page_num
        return page_ranges[-1][2] if page_ranges else None

    results = []
    seen = set()
    for block in CITATION_BLOCK.finditer(full_text):
        preceding = full_text[: block.start()]
        boundaries = list(SENTENCE_BOUNDARY.finditer(preceding))
        start_idx = boundaries[-1].end() if boundaries else max(0, len(preceding) - 400)
        claim_text = preceding[start_idx:].strip()
        if len(claim_text.split()) < 4:
            continue

        doc_page = doc_page_for_offset(block.start())

        for entry in CITATION_ENTRY.finditer(block.group(1)):
            author = entry.group(1).strip(" ,")
            year = entry.group(2)
            page_hint = int(entry.group(3))
            key = (claim_text, author, year, page_hint)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "text": claim_text,
                    "author": author,
                    "year": year,
                    "page_hint": page_hint,
                    "doc_page": doc_page,
                }
            )
    return results


def get_book_meta(filename: str, doc: fitz.Document):
    meta = doc.metadata or {}
    first_page_text = doc[0].get_text() if doc.page_count else ""
    haystack = " ".join(
        [meta.get("author", "") or "", meta.get("title", "") or "", filename, first_page_text[:500]]
    ).lower()
    return {"num_pages": doc.page_count, "haystack": haystack}


def chunk_book(filename: str, pdf_bytes: bytes):
    """Zerlegt ein Buch-PDF in Satz-Chunks pro Seite: (filename, page_num, chunk_text)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        buf = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            buf = (buf + " " + s).strip() if buf else s
            if len(buf.split()) >= 8:
                chunks.append((filename, page_num, buf))
                buf = ""
        if buf and len(buf.split()) >= 3:
            chunks.append((filename, page_num, buf))
    meta = get_book_meta(filename, doc)
    doc.close()
    return chunks, meta


@st.cache_data(show_spinner="Indexiere Buecher...")
def build_index(_model, book_files_data):
    """book_files_data: list of (filename, bytes)"""
    all_chunks = []
    book_meta = {}
    for filename, data in book_files_data:
        chunks, meta = chunk_book(filename, data)
        all_chunks.extend(chunks)
        book_meta[filename] = meta

    if not all_chunks:
        return [], None, book_meta

    texts = [c[2] for c in all_chunks]
    embeddings = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
    return all_chunks, embeddings, book_meta


def find_book_for_quote(model, quote, chunks, embeddings, book_meta):
    """Findet das PDF, aus dem ein Zitat stammt. Autor bestimmt das Buch, die im Zitat
    angegebene Seite wird zur Verifikation genutzt (nicht selbst geschaetzt)."""
    query_emb = model.encode([quote["text"]], convert_to_numpy=True)
    scores = util.cos_sim(query_emb, embeddings)[0].numpy()

    author = (quote["author"] or "").lower()
    author_tokens = [t for t in re.split(r"[\s,&]+", author) if len(t) > 2 and t != "und"]
    page_hint = quote["page_hint"]

    candidate_filenames = [
        fn
        for fn, meta in book_meta.items()
        if author_tokens and any(tok in meta.get("haystack", "") for tok in author_tokens)
    ]
    if not candidate_filenames:
        candidate_filenames = list(book_meta.keys())
        author_matched = False
    else:
        author_matched = True

    # unter den Kandidaten-Buechern: bestbewerteten Chunk exakt auf der angegebenen Seite suchen
    best = None
    for fn in candidate_filenames:
        page_chunks = [
            (i, c) for i, c in enumerate(chunks) if c[0] == fn and c[1] == page_hint
        ]
        for i, c in page_chunks:
            score = float(scores[i])
            if best is None or score > best[0]:
                best = (score, fn, c[2])

    if best is not None:
        score, filename, chunk_text = best
        return {
            "filename": filename,
            "page_num": page_hint,
            "chunk_text": chunk_text,
            "score": score,
            "author_matched": author_matched,
            "verified": True,
        }

    # Zitat wurde auf der angegebenen Seite in keinem Kandidaten-Buch gefunden:
    # bestes Buch ueber den Autor waehlen (falls vorhanden), sonst insgesamt bestes semantisches Ergebnis
    if candidate_filenames != list(book_meta.keys()):
        idxs = [i for i, c in enumerate(chunks) if c[0] in candidate_filenames]
        best_idx = max(idxs, key=lambda i: scores[i])
    else:
        best_idx = int(np.argmax(scores))

    return {
        "filename": chunks[best_idx][0],
        "page_num": page_hint,
        "chunk_text": "",
        "score": float(scores[best_idx]),
        "author_matched": author_matched,
        "verified": False,
    }


st.title("Zitat-Finder")
st.caption(
    "Lade ein PDF mit Zitaten hoch und deine Buecher als PDF. Alle Textstellen mit "
    "APA-Zitatklammer, z.B. (Autor, Jahr, S. Seite), werden automatisch den Buechern zugeordnet."
)

col1, col2 = st.columns([1, 1])
with col1:
    quote_pdf_file = st.file_uploader("PDF mit deinem Text (enthaelt die Zitate)", type="pdf", key="quote_pdf")
with col2:
    book_files = st.file_uploader(
        "Buecher (PDF, bis zu 30)",
        type="pdf",
        accept_multiple_files=True,
        key="book_pdfs",
    )

if book_files and len(book_files) > 30:
    st.warning(f"Du hast {len(book_files)} Buecher hochgeladen, es werden nur die ersten 30 verwendet.")
    book_files = book_files[:30]

if quote_pdf_file and book_files:
    model = load_model()

    quote_pdf_bytes = quote_pdf_file.read()
    quotes = extract_quotes(quote_pdf_bytes)

    book_files_data = [(f.name, f.read()) for f in book_files]
    chunks, embeddings, book_meta = build_index(model, book_files_data)

    if not quotes:
        st.info(
            "Es wurden keine Zitatklammern gefunden. Erwartetes Format: "
            "„...Aussage im Text (Autor, Jahr, S. 42; Autor2, Jahr, S. 7).“"
        )
    elif embeddings is None or len(chunks) == 0:
        st.error("Keine durchsuchbaren Inhalte in den hochgeladenen Buechern gefunden.")
    else:
        st.subheader(f"{len(quotes)} Zitate gefunden")

        with st.spinner(f"Ordne {len(quotes)} Zitate den Buechern zu..."):
            rows = []
            for q in quotes:
                result = find_book_for_quote(model, q, chunks, embeddings, book_meta)
                if result["verified"]:
                    status = "bestaetigt"
                elif result["author_matched"]:
                    status = "Autor gefunden, Seite nicht bestaetigt"
                else:
                    status = "nicht sicher gefunden"
                rows.append(
                    {
                        "Zitat": q["text"][:150] + ("..." if len(q["text"]) > 150 else ""),
                        "Seite (dein Dokument)": q["doc_page"],
                        "Autor": q["author"],
                        "Jahr": q["year"],
                        "Seite (Buch)": q["page_hint"],
                        "PDF": result["filename"],
                        "Status": status,
                    }
                )

        st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.info("Bitte lade oben links dein Zitat-PDF und rechts deine Buecher hoch.")
