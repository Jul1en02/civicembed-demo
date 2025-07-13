# civicembed_streamlit.py — revamped UI with the same feature set but smoother UX
# ============================================================================
# Key improvements -----------------------------------------------------------
# • Cleaner layout using a sidebar for filters and a main pane for results.
# • Sliders & filter widgets automatically clamp to realistic ranges.
# • Format filter bug fixed — the card now reflects true dataset formats.
# • Empty metadata fields are skipped in the summary line to reduce clutter.
# • A compact inline “Read more” expander that keeps the description flow.
# • Result counter + early exit message when no datasets match.
# ----------------------------------------------------------------------------

import streamlit as st
import torch
import json
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
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH    = "base_embeddings.faiss"
ID_LIST_PATH  = "base_id_list.json"
#META_PATH     = "dataset_info.json"
META_PATH     = "dataset_info_openai_translated.json"
MAX_DESC_LEN  = 300    # chars before expander kicks in
SEARCH_TOP_K  = 400    # retrieve up to N matches then filter

# ─── CATEGORY MAP (DE→EN) ────────────────────────────────────────────────────
CATEGORY_MAP = {
    "Landwirtschaft, Fischerei, Forstwirtschaft und Nahrungsmittel":
        "Agriculture, fisheries, forestry and food",
    "Regierung und öffentlicher Sektor":
        "Government and public sector",
    "Regionen und Städte":
        "Regions and cities",
    "Umwelt":
        "Environment",
    "Verkehr":
        "Transport",
    "Wirtschaft und Finanzen":
        "Economy and finance",
}

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def get_en_group(g):
    """Return the English label for a group entry (dict | str)."""
    if isinstance(g, dict):
        return g.get("en") or next(iter(g.values()), "")
    if isinstance(g, str):
        try:
            obj = json.loads(g)
            if isinstance(obj, dict):
                return obj.get("en") or next(iter(obj.values()), "")
        except Exception:
            pass
        return CATEGORY_MAP.get(g, g)
    return str(g)

@st.cache_resource(show_spinner=False)
def load_encoder():
    return SentenceTransformer(TEXT_MODEL, device="cpu")

@st.cache_resource(show_spinner=False)
def load_searcher():
    return CivicSearcher(
        index_path=INDEX_PATH,
        id_list_path=ID_LIST_PATH,
        metadata_path=META_PATH,
    )

@st.cache_data(show_spinner=False)
def load_metadata():
    return json.loads(Path(META_PATH).read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def embed_query(_model, txt: str):
    """Embed a query string using the cached model without hashing errors."""
    with torch.no_grad():
        return _model.encode([txt]).astype("float32")

# ─── INITIALISATION ──────────────────────────────────────────────────────────
encoder  = load_encoder()
searcher = load_searcher()
metadata = load_metadata()

# Pre-compute global option lists
publishers     = sorted({m.get("publisher", "–") for m in metadata})
all_group_keys = {get_en_group(g) for m in metadata for g in m.get("groups", [])}
categories     = sorted(all_group_keys)
formats_list   = sorted({
    fmt.lower()  # normalise case
    for m in metadata
    for fmt in (m.get("structured_formats") or m.get("resource_formats", []))
})

years = sorted({m.get("issued_year") for m in metadata if m.get("issued_year")})
min_year, max_year = (min(years), max(years)) if years else (0, 0)

# ─── SIDEBAR — FILTERS ───────────────────────────────────────────────────────
st.sidebar.header("🔧 Filters")

# 1️⃣ Issued year range — clamp to [1900, max_year] for smoother control
issued_year = st.sidebar.slider(
    "Issued year range",
    max(min_year, 1900), max_year,
    (max(min_year, 1900), max_year),
    step=1,
)

# 2️⃣ Other filters — empty by default (opt-in)
sel_formats    = st.sidebar.multiselect("Formats", formats_list)
sel_publishers = st.sidebar.multiselect("Publishers", publishers)
sel_categories = st.sidebar.multiselect("Categories", categories)

# ─── MAIN AREA ───────────────────────────────────────────────────────────────
st.title("CivicEmbed Demo")
query = st.text_input("Search prompt", "Mobility in Lausanne")

if query:
    q_vec   = embed_query(encoder, query)
    hits = searcher.search(q_vec, top_k=2000, normalize=True)
    threshold = 0.50  # or expose this in sidebar
    hits = [h for h in hits if h.get("score", 0.0) >= threshold]

    results = []

    # ——— apply filters ———
    for h in hits:
        yr = h.get("issued_year")
        if yr is None or not (issued_year[0] <= yr <= issued_year[1]):
            continue

        dataset_fmts = {
            fmt.lower() for fmt in (
                h.get("structured_formats") or h.get("resource_formats", [])
            )
        }
        if sel_formats and not dataset_fmts.intersection(sel_formats):
            continue

        if sel_publishers and h.get("publisher", "–") not in sel_publishers:
            continue

        dataset_groups = {get_en_group(g) for g in h.get("groups", [])}
        if sel_categories and not dataset_groups.intersection(sel_categories):
            continue

        results.append(h)

    st.write(f"### {len(results)} dataset(s) found")

    if not results:
        st.info("No datasets match the current filters. Try relaxing them.")

    # ——— display cards ———
    for h in results:
        title       = h.get("title", "(no title)")
        full_desc   = h.get("description", "")
        score       = h.get("score", 0.0)
        ds_id       = h.get("id")
        url         = f"https://opendata.swiss/en/dataset/{ds_id}"
        num_res     = h.get("num_resources", 0)
        issued      = h.get("issued_year", "–")
        publisher   = h.get("publisher", "–")
        groups_en   = [get_en_group(g) for g in h.get("groups", [])]
        dataset_fmts= h.get("structured_formats") or h.get("resource_formats", [])

        # — card header
        header_cols = st.columns([8, 2])
        header_cols[0].subheader(title)
        header_cols[1].markdown(f"**{score:.3f}** [🔗]({url})", unsafe_allow_html=True)

        # — description with inline expander
        if len(full_desc) > MAX_DESC_LEN:
            short = full_desc[:MAX_DESC_LEN].rstrip() + " …"
            st.write(short)
            with st.expander("Read more"):
                st.write(full_desc)
        else:
            st.write(full_desc)

        # — compact metadata line: only include non-empty fields
        meta_fragments = []
        if num_res:
            meta_fragments.append(f"**Resources:** {num_res}")
        if issued != "–":
            meta_fragments.append(f"**Issued:** {issued}")
        if dataset_fmts:
            meta_fragments.append(f"**Formats:** {', '.join(dataset_fmts)}")
        if publisher != "–":
            meta_fragments.append(f"**Publisher:** {publisher}")
        if groups_en:
            meta_fragments.append(f"**Category:** {', '.join(groups_en)}")

        st.markdown(" • ".join(meta_fragments))
        st.markdown("---")
else:
    st.write("Enter a search prompt to begin.")
