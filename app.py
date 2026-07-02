import re

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Zitat-Finder", layout="wide")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Zitat in Anfuehrungszeichen, gefolgt optional von einer Klammer mit Autor/Seitenzahl,
# z.B. "Zitat..." (Autor, S. 42)
QUOTE_PATTERN = re.compile(
    r'[„"»]([^„"“”»«]{15,600}?)[”"«]'
    r'(?:\s*\(([^()]*?)\))?',
    re.DOTALL,
)
PAGE_IN_CITATION = re.compile(r"S\.?\s*(\d+)", re.IGNORECASE)


@st.cache_resource(show_spinner="Lade Sprachmodell (nur beim ersten Mal, danach gecacht)...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


def extract_quotes(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()

    results = []
    seen = set()
    for m in QUOTE_PATTERN.finditer(full_text):
        text = re.sub(r"\s+", " ", m.group(1).strip())
        if len(text.split()) < 3 or text in seen:
            continue
        seen.add(text)

        citation = (m.group(2) or "").strip()
        author = None
        page_hint = None
        if citation:
            page_match = PAGE_IN_CITATION.search(citation)
            if page_match:
                page_hint = int(page_match.group(1))
                author = citation[: page_match.start()].strip(" ,")
            else:
                author = citation.strip(" ,")
            author = author or None

        # nur Zitate uebernehmen, die vollstaendig mit Autor UND Seitenzahl gekennzeichnet sind
        if author and page_hint:
            results.append({"text": text, "author": author, "page_hint": page_hint})
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
    author_tokens = [t for t in re.split(r"[\s,]+", author) if len(t) > 2]
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


def render_highlighted_page(pdf_bytes: bytes, page_num: int, search_text: str):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_num - 1]
    if search_text:
        rects = page.search_for(search_text[:200])
        for r in rects:
            page.add_highlight_annot(r)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


st.title("Zitat-Finder")
st.caption(
    "Lade ein PDF mit Zitaten hoch und deine Buecher als PDF. Klicke auf ein Zitat, "
    "um das passende Buch und die Fundstelle zu finden."
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
    book_bytes_by_name = {name: data for name, data in book_files_data}

    chunks, embeddings, book_meta = build_index(model, book_files_data)

    if not quotes:
        st.info(
            "Es wurden keine vollstaendig gekennzeichneten Zitate gefunden. "
            "Zitate muessen im Format „Zitat...“ (Autor, S. 42) vorliegen."
        )
    else:
        st.subheader(f"{len(quotes)} Zitate gefunden — klicke auf eines")

        if "selected_quote_idx" not in st.session_state:
            st.session_state.selected_quote_idx = None

        for i, q in enumerate(quotes):
            label = f"„{q['text'][:110]}{'...' if len(q['text']) > 110 else ''}\""
            hint_parts = []
            if q["author"]:
                hint_parts.append(q["author"])
            if q["page_hint"]:
                hint_parts.append(f"S. {q['page_hint']}")
            if hint_parts:
                label += f"  ({', '.join(hint_parts)})"
            if st.button(label, key=f"quote_{i}"):
                st.session_state.selected_quote_idx = i

        if st.session_state.selected_quote_idx is not None:
            st.divider()
            sel = quotes[st.session_state.selected_quote_idx]
            st.markdown(f"**Ausgewaehltes Zitat:** „{sel['text']}\"")

            if embeddings is None or len(chunks) == 0:
                st.error("Keine durchsuchbaren Inhalte in den hochgeladenen Buechern gefunden.")
            else:
                with st.spinner("Suche PDF..."):
                    result = find_book_for_quote(model, sel, chunks, embeddings, book_meta)

                st.markdown(f"### PDF: {result['filename']}")
                st.caption(f"Seite {result['page_num']} (laut Zitatangabe)")

                if result["verified"]:
                    st.success("Zitat auf dieser Seite in diesem PDF bestaetigt gefunden.")
                elif result["author_matched"]:
                    st.warning(
                        "Autor gefunden, das Zitat konnte auf der angegebenen Seite aber nicht "
                        "exakt bestaetigt werden (evtl. abweichender Text-Layout)."
                    )
                else:
                    st.warning(
                        "Kein PDF mit passendem Autor gefunden — bestes inhaltliches Ergebnis wird angezeigt."
                    )

                img_bytes = render_highlighted_page(
                    book_bytes_by_name[result["filename"]], result["page_num"], result["chunk_text"]
                )
                st.image(
                    img_bytes,
                    caption=f"{result['filename']} — Seite {result['page_num']}",
                    use_container_width=True,
                )
else:
    st.info("Bitte lade oben links dein Zitat-PDF und rechts deine Buecher hoch.")
