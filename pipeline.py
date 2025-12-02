from pathlib import Path
import json
import PyPDF2
import pickle
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"

class PDFIngestor:
    def __init__(self, model_name=EMBED_MODEL, chunk_size=800, chunk_overlap=120):
        self.embedder = SentenceTransformer(model_name)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def extract_text(self, pdf_path: Path) -> str:
        texts = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                texts.append(page.extract_text() or "")
        return "\n".join(texts)

    def create_chunks(self, text: str):
        raw_chunks = self.splitter.split_text(text)
        return [{"id": f"chunk-{i}", "text": c} for i, c in enumerate(raw_chunks)]

    def embed_chunks(self, chunks):
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings

def build_faiss_index(embeddings, metadata, index_path: Path):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    with open(index_path.with_suffix(".meta.pkl"), "wb") as f:
        pickle.dump(metadata, f)

def ingest_pdf_to_faiss(pdf_path: str, output_dir: str):
    pdf = Path(pdf_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ingestor = PDFIngestor()
    raw_text = ingestor.extract_text(pdf)

    if not raw_text.strip():
        print("No text extracted from:", pdf)

    chunks = ingestor.create_chunks(raw_text)
    if not chunks:
        print("No chunks created for:", pdf)
    else:
        embeddings = ingestor.embed_chunks(chunks)
        metadata = [{"source": pdf.name, "excerpt": c["text"][:200], "chunk_id": c["id"]} for c in chunks]
        index_file = out / f"{pdf.stem}.faiss.index"
        build_faiss_index(embeddings, metadata, index_file)
        with open(out / f"{pdf.stem}.chunks.json", "w", encoding="utf8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print("Ingested:", pdf, "->", index_file)
%%writefile build_index_helper.py
from pathlib import Path
from pipeline import ingest_pdf_to_faiss

def ingest_folder(folder="dataset", out="vector_store"):
    folder_path = Path(folder)
    if not folder_path.exists():
        print("Folder not found:", folder)
        return

    pdfs = sorted(folder_path.rglob("*.pdf"))
    if not pdfs:
        print("No PDFs found in folder:", folder)
        return

    for pdf in pdfs:
        print("Processing:", pdf)
        try:
            ingest_pdf_to_faiss(str(pdf), out)
        except Exception as e:
            print("Error ingesting", pdf, e)

    print("Ingestion completed.")
