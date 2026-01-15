from  google import genai 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

EMBED_DIM=768

# Hugging Face embedding model (LOCAL, no API)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

splitter=SentenceSplitter(chunk_size=1000,chunk_overlap=200)

def load_and_chunk_pdf(path:str):
    docs=PDFReader().load_data(path)
    text=[d.text for d in docs if getattr(d,"text",None)]
    chunks=[]
    for t in text:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_text(text:list[str])-> list[list[float]]:
    vectors=embed_model.get_image_embedding_batch(text)
    # 🔒 Safety check (VERY IMPORTANT)
    assert len(vectors[0]) == EMBED_DIM, f"Expected {EMBED_DIM}, got {len(vectors[0])}"
    return vectors
    

