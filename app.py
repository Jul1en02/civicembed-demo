# app.py — CivicEmbed Demo with Blendable Base & Topical Lenses
# ===========================================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils.search import CivicSearcher

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CivicEmbed Demo",
    page_icon="🔍",
    layout="wide",
)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent
DATA            = ROOT / "data"

TEXT_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
BASE_INDEX      = DATA / "base_embeddings.faiss"
TOPIC_INDEX     = DATA / "topic_embeddings.faiss"
ID_LIST_PATH    = DATA / "base_id_list.json"
TOPIC_ID_LIST   = DATA / "topic_id_list.json"
META_PATH       = DATA / "opendataswiss_metadata_en_with_groups.json"
MAX_DESC_LEN    = 300    # chars before expander
SEARCH_TOP_K    = 400

# Path to the trained topical lens weights:
TOPIC_LENS_WEIGHTS = DATA / "embeddings" / "topical_lens.pt"

# ─── UTILITIES ───────────────────────────────────────────────────────────────
def get_en_group(g):
    return str(g)

# ─── TOPIC LENS MODEL ────────────────────────────────────────────────────────
class LensMLP(nn.Module):
    def __init__(self, dim=384, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim, bias=False)
        )
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)

# ─── CACHING ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_encoder():
    return SentenceTransformer(TEXT_MODEL, device="cpu")

@st.cache_resource(show_spinner=False)
def load_searcher():
    return CivicSearcher(
        index_path=str(BASE_INDEX),
        id_list_path=str(ID_LIST_PATH),
        metadata_path=str(META_PATH),
    )

@st.cache_resource(show_spinner=False)
def load_topic_index():
    idx = faiss.read_index(str(TOPIC_INDEX))
    ids = json.loads(Path(TOPIC_ID_LIST).read_text(encoding="utf-8"))
    return idx, ids

@st.cache_data(show_spinner=False)
def load_metadata():
    return json.loads(META_PATH.read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def embed_query(_model, txt: str):
    with torch.no_grad():
        vec = _model.encode([txt], convert_to_numpy=True)
    return vec.astype("float32")

# ─── INITIALIZATION ─────────────────────────────────────────────────────────
encoder        = load_encoder()
searcher       = load_searcher()
topic_index, topic_ids = load_topic_index()
metadata_list  = load_metadata()

# Build metadata lookup
metadata_map = {rec["id"]: rec for rec in metadata_list}

# Precompute filter options
publishers   = sorted({m.get("publisher", "–") for m in metadata_list})
categories   = sorted({get_en_group(g) for m in metadata_list for g in m.get("groups", [])})
formats_list = sorted({
    fmt.lower()
    for m in metadata_list
    for fmt in (m.get("structured_formats") or m.get("resource_formats", []))
})
years        = sorted({m.get("issued_year") for m in metadata_list if m.get("issued_year")})
min_year, max_year = (min(years), max(years)) if years else (0, 0)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Filters & Blending")
issued_year    = st.sidebar.slider(
    "Issued year range",
    max(min_year, 1900), max_year,
    (max(min_year, 1900), max_year),
    step=1,
)
sel_formats    = st.sidebar.multiselect("Formats", formats_list)
sel_publishers = st.sidebar.multiselect("Publishers", publishers)
sel_categories = st.sidebar.multiselect("Categories", categories)

w_base  = st.sidebar.slider("Base weight",  0.0, 1.0, 0.5, 0.05)
w_topic = st.sidebar.slider("Topic weight", 0.0, 1.0, 0.5, 0.05)
norm    = (w_base + w_topic) or 1e-6

# ─── MAIN ────────────────────────────────────────────────────────────────────
st.title("CivicEmbed Demo")
query = st.text_input("Search prompt", "Mobility in Lausanne")

if query:
    # 1) Embed query
    q_base = embed_query(encoder, query)

    # 2) Base search
    base_hits   = searcher.search(q_base, top_k=SEARCH_TOP_K, normalize=True)
    base_scores = {h["id"]: h["score"] for h in base_hits}

    # 3) Topic lens projection & search
    lens_model = LensMLP(dim=384)
    lens_model.load_state_dict(torch.load(TOPIC_LENS_WEIGHTS, map_location="cpu"))
    lens_model.eval()
    with torch.no_grad():
        q_topic = lens_model(torch.from_numpy(q_base)).detach().numpy()
    Dt, It = topic_index.search(q_topic, SEARCH_TOP_K)
    topic_scores = { topic_ids[i]: dist for dist, i in zip(Dt[0], It[0]) }

    # 4) Combine scores
    combined = {}
    for ds_id, sc in base_scores.items():
        combined[ds_id] = w_base * sc
    for ds_id, sc in topic_scores.items():
        combined[ds_id] = combined.get(ds_id, 0.0) + w_topic * sc
    for ds_id in combined:
        combined[ds_id] /= norm

    # 5) Build sorted hits
    sorted_hits = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    hits = [
        { **metadata_map[ds_id], "id": ds_id, "score": float(score) }
        for ds_id, score in sorted_hits[:SEARCH_TOP_K]
    ]

    # 6) Apply filters
    results = []
    threshold = 0.05
    for h in hits:
        if h["score"] < threshold: continue
        yr = h.get("issued_year")
        if yr is None or not (issued_year[0] <= yr <= issued_year[1]): continue
        fmt_set = {f.lower() for f in (h.get("structured_formats") or h.get("resource_formats", []))}
        if sel_formats and not fmt_set.intersection(sel_formats): continue
        if sel_publishers and h.get("publisher", "–") not in sel_publishers: continue
        grp_set = {get_en_group(g) for g in h.get("groups", [])}
        if sel_categories and not grp_set.intersection(sel_categories): continue
        results.append(h)

    st.write(f"### {len(results)} dataset(s) found")
    if not results:
        st.info("No datasets match. Try relaxing filters or blend weights.")

    # 7) Display result cards
    for h in results:
        title     = h.get("title", "(no title)")
        desc      = h.get("description", "")
        score     = h.get("score", 0.0)
        url       = f"https://opendata.swiss/en/dataset/{h['id']}"
        num_res   = h.get("num_resources", 0)
        issued    = h.get("issued_year", "–")
        publisher = h.get("publisher", "–")
        groups_en = h.get("groups", [])
        fmts      = h.get("structured_formats") or h.get("resource_formats", [])

        c1, c2 = st.columns([8, 2])
        c1.subheader(title)
        c2.markdown(f"**{score:.3f}** [🔗]({url})")

        if len(desc) > MAX_DESC_LEN:
            st.write(desc[:MAX_DESC_LEN].rstrip() + " …")
            with st.expander("Read more"):
                st.write(desc)
        else:
            st.write(desc)

        meta_frags = []
        if num_res:   meta_frags.append(f"**Resources:** {num_res}")
        if issued!="–":meta_frags.append(f"**Issued:** {issued}")
        if fmts:      meta_frags.append(f"**Formats:** {', '.join(fmts)}")
        if publisher: meta_frags.append(f"**Publisher:** {publisher}")
        if groups_en: meta_frags.append(f"**Category:** {', '.join(groups_en)}")

        st.markdown(" • ".join(meta_frags))
        st.markdown("---")
else:
    st.write("Enter a search prompt to begin.")
