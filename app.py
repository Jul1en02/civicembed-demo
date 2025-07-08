import streamlit as st
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from utils.search import CivicSearcher

# ——— Load models & searcher ——————————————————————————————
sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
searcher = CivicSearcher(
    index_path="base_embeddings.faiss",
    id_list_path="base_id_list.json",
    metadata_path="dataset_info.json"   # or whatever your metadata file is
)

# ——— Streamlit UI ——————————————————————————————————————
st.title("🔍 CivicEmbed Demo (Base SBERT)")

query = st.text_input("Enter your search prompt", "water in Lausanne")
top_k = st.slider("Top K results", 1, 20, 5)

if query:
    # 1) Embed
    with torch.no_grad():
        qv = sbert.encode([query]).astype("float32")
    # 2) Search
    results = searcher.search(qv, top_k=top_k, normalize=True)

    # 3) Display
    for r in results:
        st.markdown(f"### {r.get('title','(no title)')}  ")
        st.write(r.get("description",""))
        st.markdown(f"[Open Dataset]({r.get('link','#')}) — *{r.get('publisher','')}*  ")
        st.write(f"**Score:** {r['score']:.3f}")
        st.markdown("---")
