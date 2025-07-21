#!/usr/bin/env python3
"""
app.py — CivicEmbed Demo with Blendable Base & Topical Lenses + Conditional Penalties
Normalized penalty distances to [0,1] scale; dynamic topic-weight slider
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import faiss
import numpy as np
import networkx as nx
import unicodedata
import re
from pathlib import Path
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from utils.search import CivicSearcher
from collections import Counter

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="CivicEmbed Demo", page_icon="🔍", layout="wide")

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
ROOT               = Path(__file__).resolve().parent
DATA               = ROOT / "data"

TEXT_MODEL         = "sentence-transformers/all-MiniLM-L6-v2"
BASE_INDEX         = DATA / "base_embeddings.faiss"
TOPIC_INDEX        = DATA / "topic_embeddings.faiss"
ID_LIST_PATH       = DATA / "base_id_list.json"
TOPIC_ID_LIST      = DATA / "topic_id_list.json"
META_PATH          = DATA / "opendataswiss_metadata_en_with_groups.json"
DS_ADMIN_MAP       = DATA / "ds_to_admin.json"

ADMIN_GRAPH        = DATA / "admin_dag_en.pkl"
ADMIN_DIST         = DATA / "admin_wdistances.pkl"
DIST_GRAPH         = DATA / "geo_graph_en.pkl"
DIST_DIST          = DATA / "geo_wdistances.pkl"

TOPIC_LENS_WEIGHTS = DATA / "embeddings" / "topic_lens.pt"

MAX_DESC_LEN       = 1000
SEARCH_TOP_K       = 400

# ─── NORMALIZATION & FUZZY ────────────────────────────────────────────────────
def normalize_str(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = s.encode("ascii", "ignore").decode("utf-8")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

def fuzzy_match(a: str, b: str, threshold: float = 0.95) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= threshold

# ─── LABEL LOOKUP & MATCHING ─────────────────────────────────────────────────
def build_label_lookup(graph: nx.Graph) -> dict[str, str]:
    CANTON_ABBR = {
        "Zürich":"ZH","Bern":"BE","Luzern":"LU","Uri":"UR","Schwyz":"SZ",
        "Obwalden":"OW","Nidwalden":"NW","Glarus":"GL","Zug":"ZG",
        "Fribourg":"FR","Solothurn":"SO","Basel-Stadt":"BS","Basel-Landschaft":"BL",
        "Schaffhausen":"SH","Appenzell Ausserrhoden":"AR","Appenzell Innerrhoden":"AI",
        "St. Gallen":"SG","Graubünden":"GR","Aargau":"AG","Thurgau":"TG",
        "Ticino":"TI","Vaud":"VD","Valais":"VS","Neuchâtel":"NE",
        "Genève":"GE","Jura":"JU"
    }
    labels = {}
    for node in graph.nodes():
        attrs = graph.nodes[node]
        lbl = normalize_str(attrs.get("label", ""))
        labels[node] = lbl
        if attrs.get("level") == "canton":
            abbr = CANTON_ABBR.get(attrs.get("label"))
            if abbr:
                labels[f"{node}_abbr"] = abbr.lower()
    return labels


def find_matches(fields: list[str], labels_lookup: dict[str,str]) -> set[str]:
    combined = " ".join(normalize_str(f) for f in fields if f)
    tokens = combined.split()
    matched = set()
    for key, lbl in labels_lookup.items():
        if key.endswith("_abbr"):
            if lbl in tokens:
                matched.add(key.split("_abbr")[0])
        else:
            if lbl in combined or fuzzy_match(lbl, combined):
                matched.add(key.split("_abbr")[0])
    return matched

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

# ─── CACHE LOADERS ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_encoder(): return SentenceTransformer(TEXT_MODEL, device="cpu")

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
def load_metadata(): return json.loads(Path(META_PATH).read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def embed_query(_model, text: str) -> np.ndarray:
    with torch.no_grad():
        vec = _model.encode([text], convert_to_numpy=True)
    return vec.astype("float32")

@st.cache_resource(show_spinner=False)
def load_graph_and_dist(graph_pkl, dist_pkl):
    import pickle
    with open(graph_pkl, "rb") as gf: G = pickle.load(gf)
    with open(dist_pkl, "rb") as df: D = pickle.load(df)
    return G, D

@st.cache_resource(show_spinner=False)
def load_ds_to_admin(): return json.loads(Path(DS_ADMIN_MAP).read_text(encoding="utf-8"))

# ─── INITIAL SETUP ──────────────────────────────────────────────────────────
encoder, searcher = load_encoder(), load_searcher()
topic_index, topic_ids = load_topic_index()
metadata_list = load_metadata()
metadata_map = {rec["id"]: rec for rec in metadata_list}

# global keyword set
keyword_set = {
    normalize_str(k)
    for rec in metadata_list
    for k in rec.get("keywords", []) + rec.get("tags", [])
    if isinstance(k, str) and k.strip()
}

# occurrence counts (min 50 to qualify)
keyword_counts = Counter()
for rec in metadata_list:
    kws = {normalize_str(k) for k in rec.get("keywords", []) + rec.get("tags", []) if isinstance(k, str) and k.strip()}
    for kw in kws: keyword_counts[kw] += 1

# load graphs & mappings
G_admin, D_admin = load_graph_and_dist(ADMIN_GRAPH, ADMIN_DIST)
G_dist,  D_dist  = load_graph_and_dist(DIST_GRAPH, DIST_DIST)
ds_to_admin     = load_ds_to_admin()

labels_admin = build_label_lookup(G_admin)
labels_geo   = build_label_lookup(G_dist)

# global max distances\

_max_admin = max(d for u in D_admin for d in D_admin[u].values())
_max_geo   = max(d for u in D_dist  for d in D_dist[u].values())

# filters
publishers   = sorted({m.get("publisher","–") for m in metadata_list})
categories   = sorted({str(g) for m in metadata_list for g in m.get("groups",[])})
formats_list = sorted({fmt.lower() for m in metadata_list for fmt in (m.get("structured_formats") or m.get("resource_formats", []))})
years        = sorted({m.get("issued_year") for m in metadata_list if m.get("issued_year")})
min_year, max_year = (min(years), max(years)) if years else (0,0)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Filters & Penalties")
issued_year    = st.sidebar.slider("Issued year range", max(min_year,1900), max_year, (max(min_year,1900), max_year), step=1)
sel_formats    = st.sidebar.multiselect("Formats", formats_list)
sel_publishers = st.sidebar.multiselect("Publishers", publishers)
sel_categories = st.sidebar.multiselect("Categories", categories)

# penalty sliders
w_admin = st.sidebar.slider("Admin penalty", 0.0, 1.0, 0.0, 0.1)
w_dist  = st.sidebar.slider("Distance penalty", 0.0, 1.0, 1.0, 0.1)

# ─── MAIN ────────────────────────────────────────────────────────────────────
st.title("CivicEmbed Demo")
query = st.text_input("Search prompt", "Mobility in Lausanne")

# detect locations & topics
matched_admin = find_matches([query], labels_admin)
matched_geo   = find_matches([query], labels_geo)
query_tokens  = normalize_str(query).split()
matched_topic = {t for t in query_tokens if keyword_counts.get(t, 0) >= 50}

# show detections
admin_labels = [G_admin.nodes[n]["label"] for n in matched_admin] if matched_admin else []
geo_labels   = [G_dist.nodes[n]["label"]   for n in matched_geo]   if matched_geo   else []
if admin_labels or geo_labels:
    st.write(f"**Detected location(s):** {', '.join(sorted(set(admin_labels + geo_labels)))}")
else:
    st.write("**Detected location(s):** None")
if matched_topic:
    st.write(f"**Detected topic(s):** {', '.join([t.capitalize() for t in sorted(matched_topic)])}")
else:
    st.write("**Detected topic(s):** None")

# topic-weight slider (disabled when no topic)
slider_disabled = not bool(matched_topic)
# use session_state to remember last setting
default_blend = st.session_state.get("topic_blend", 0.5)
topic_blend = st.sidebar.slider(
    "Topic weight",
    0.0, 1.0,
    value=default_blend,
    step=0.1,
    key="topic_blend",
    disabled=slider_disabled
)

# compute blend weights
if matched_topic:
    w_topic = topic_blend
    w_base = 1.0 - w_topic
else:
    w_topic = 0.0
    w_base = 1.0
norm = (w_base + w_topic) or 1e-6

if query:
    # 1) Base search
    q_base    = embed_query(encoder, query)
    base_hits = searcher.search(q_base, top_k=SEARCH_TOP_K, normalize=True)
    base_scores = {h["id"]: h["score"] for h in base_hits}

    # 2) Topic lens search
    lens_model  = LensMLP(dim=384)
    lens_model.load_state_dict(torch.load(TOPIC_LENS_WEIGHTS, map_location="cpu"))
    lens_model.eval()
    with torch.no_grad():
        q_topic = lens_model(torch.from_numpy(q_base)).numpy()
    Dt, It = topic_index.search(q_topic, SEARCH_TOP_K)
    topic_scores = {topic_ids[i]: Dt[0][idx] for idx,i in enumerate(It[0])}

    # 3) Blend
    combined = {}
    for ds,sc in base_scores.items(): combined[ds] = w_base*sc
    for ds,sc in topic_scores.items(): combined[ds] = combined.get(ds,0.0) + w_topic*sc
    for ds in combined: combined[ds] /= norm

    # 4–6) Min-distance penalties
    admin_penalties, dist_penalties = {}, {}
    admin_best, dist_best = {}, {}
    if matched_admin or matched_geo:
        for ds in combined:
            rec_nodes = ds_to_admin.get(ds, [])
            if "S00" in rec_nodes and len(rec_nodes) > 1:
                rec_nodes = [n for n in rec_nodes if n != "S00"]
            # admin penalty
            if rec_nodes:
                best_d, best_dn = _max_admin, None
                for qn in matched_admin:
                    for dn in rec_nodes:
                        d = D_admin.get(qn,{}).get(dn, np.inf)
                        if d < best_d: best_d, best_dn = d, dn
                admin_penalties[ds], admin_best[ds] = best_d, best_dn
            else:
                admin_penalties[ds], admin_best[ds] = _max_admin, None
            # geo penalty
            if rec_nodes:
                best_g, best_gn = _max_geo, None
                for qn in matched_geo:
                    for dn in rec_nodes:
                        g = D_dist.get(qn,{}).get(dn, np.inf)
                        if g < best_g: best_g, best_gn = g, dn
                dist_penalties[ds], dist_best[ds] = best_g, best_gn
            else:
                dist_penalties[ds], dist_best[ds] = _max_geo, None
    # apply penalties
    for ds in combined:
        na = admin_penalties.get(ds,0.0)/_max_admin
        ng = dist_penalties.get(ds,0.0)/_max_geo
        combined[ds] -= w_admin * na
        combined[ds] -= w_dist  * ng

    # 8) sort & preprocess
    sorted_ds = sorted(combined.items(), key=lambda x:x[1], reverse=True)
    hits = [{**metadata_map[ds],"id":ds,"score":float(s)} for ds,s in sorted_ds if s>0.0][:SEARCH_TOP_K]

    # 9) apply metadata filters
    results = []
    for h in hits:
        if h["score"]<0.05: continue
        yr = h.get("issued_year")
        if yr is None or not (issued_year[0]<=yr<=issued_year[1]): continue
        fmts = {f.lower() for f in (h.get("structured_formats") or h.get("resource_formats",[]))}
        if sel_formats and not fmts.intersection(sel_formats): continue
        if sel_publishers and h.get("publisher","–") not in sel_publishers: continue
        grps = {str(g) for g in h.get("groups",[])}
        if sel_categories and not grps.intersection(sel_categories): continue
        results.append(h)

    st.write(f"### {len(results)} dataset(s) found")
    if not results: st.info("No datasets match. Try relaxing filters or weights.")

    # 10) render results
    for h in results:
        ds_id = h["id"]
        a_raw = admin_penalties.get(ds_id,0.0)
        d_raw = dist_penalties.get(ds_id,0.0)
        a_norm = a_raw/_max_admin
        d_norm = d_raw/_max_geo
        best_dn = admin_best.get(ds_id)
        best_gn = dist_best.get(ds_id)
        a_lbl = G_admin.nodes[best_dn]["label"] if best_dn else "—"
        g_lbl = G_dist.nodes[best_gn]["label"]   if best_gn else "—"

        st.subheader(h.get("title","(no title)"))
        desc = h.get("description","")
        if len(desc)>MAX_DESC_LEN: desc = desc[:MAX_DESC_LEN].rstrip()+"…"
        st.write(desc)

        meta = []
        if h.get("publisher"): meta.append(f"**Publisher:** {h['publisher']}")
        if h.get("organization"): meta.append(f"**Organization:** {h['organization']}")
        if h.get("groups"): meta.append(f"**Groups:** {', '.join(h['groups'])}")
        issued = h.get("issued") or h.get("issued_year")
        if issued: meta.append(f"**Issued:** {issued}")
        fmts = set(h.get("structured_formats",[])+h.get("resource_formats",[]))
        if fmts: meta.append(f"**Formats:** {', '.join(sorted(fmts))}")
        superset = set(h.get("keywords",[])).union(h.get("tags",[]))
        if superset: meta.append(f"**Keywords & Tags:** {', '.join(sorted(superset))}")
        if meta: st.markdown(" • ".join(meta))

        st.markdown(
            f"**Score:** {h['score']:.3f}  •  "
            f"**Admin penalty:** {a_norm:.2f} ({a_lbl})  •  "
            f"**Distance penalty:** {d_norm:.2f} ({g_lbl})  •  "
            f"[🔗](https://opendata.swiss/en/dataset/{ds_id})"
        )
        st.markdown("---")
else:
    st.write("Enter a search prompt to begin.")
