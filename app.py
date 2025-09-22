# app.py — Medical History Search (Demo) with PMH section (shown only on button click)

import streamlit as st
import pandas as pd
import numpy as np
import re
import datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Medical History Search (Demo)", layout="wide")

# ---------------- Sidebar: patient context ----------------
st.sidebar.header("Patient")
demo_structured = dict(
    name="Jane Doe", sex="F", dob="1952-08-20",
    problems=["AF (paroxysmal)", "CAD s/p CABG (2018)"],
    meds=["Apixaban 5 mg BID", "Metoprolol 50 mg BID", "Atorvastatin 40 mg nocte"],
    allergies=["Amoxicillin – rash (remote)"],
    vitals=dict(hr=78, bp="128/72", spo2="97% RA"),
)
st.sidebar.write(f"**{demo_structured['name']}**  \n{demo_structured['sex']} • DOB {demo_structured['dob']}")
st.sidebar.write("**Problems**:", "; ".join(demo_structured["problems"]))
st.sidebar.write("**Meds**:", "; ".join(demo_structured["meds"]))
st.sidebar.write("**Allergies**:", "; ".join(demo_structured["allergies"]))
st.sidebar.write("**Vitals**:", f"HR {demo_structured['vitals']['hr']}, BP {demo_structured['vitals']['bp']}, SpO₂ {demo_structured['vitals']['spo2']}")

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload de-identified notes CSV (date,source,text)", type=["csv"])

# ---------------- Load notes (demo set if none uploaded) ----------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.DataFrame([
        dict(date="2024-07-30", source="Radiology",
             text="CTPA with iodinated contrast; no adverse reaction. No PE."),
        dict(date="2023-12-18", source="Primary care",
             text="Doing well on apixaban for paroxysmal AF. CHA2DS2-VASc 4."),
        dict(date="2022-03-14", source="Cardiology consult",
             text="72F with AF. Started apixaban 5 mg BID. TTE EF 55%."),
        dict(date="2021-11-02", source="ED discharge",
             text="CT abdomen/pelvis WITH contrast; tolerated well. Creatinine 1.1."),
        dict(date="2019-05-20", source="GI endoscopy",
             text="Upper GI bleed from gastric ulcer; transfused. Apixaban held 48h then restarted."),
        dict(date="2018-04-09", source="Cardiac surgery op note",
             text="CABG ×3: LIMA-LAD, SVG-OM, SVG-PDA."),
    ])

# ---- PMH content (required items) ----
PMH_ITEMS = [
    "Ischaemic heart disease (IHD)",
    "Heart failure",
    "Hypertension (HBP)",
    "Mitral valve disease",
    "History of binge drinking for the last 10 years",
]
pmh_text = "Previous medical history: " + "; ".join(PMH_ITEMS) + "."

# Add a PMH summary note so generic search can cite it
df = pd.concat([
    df,
    pd.DataFrame([{"date": "2024-01-05", "source": "Problem list / PMH summary", "text": pmh_text}])
], ignore_index=True)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["age_days"] = (dt.datetime.now() - df["date"]).dt.days

# ---------------- Search index ----------------
@st.cache_data(show_spinner=False)
def build_index(texts: pd.Series):
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts.fillna(""))
    return vec, X

vectorizer, X = build_index(df["text"])

def search(query: str, k: int = 6, half_life_days: int = 365):
    q = vectorizer.transform([query])
    sims = linear_kernel(q, X).ravel()
    recency = 0.5 ** (df["age_days"].fillna(3650) / half_life_days)
    score = sims * (0.2 + 0.8 * recency)  # blend content + recency
    out = df.assign(score=score).sort_values("score", ascending=False)
    return out[out["score"] > 0].head(k)

def highlight(text: str, terms):
    if not terms:
        return text
    pattern = r'(' + '|'.join(map(re.escape, terms)) + r')'
    return re.sub(pattern, r'**\1**', text, flags=re.I)

# ---------------- UI ----------------
st.title("Medical History Search (Demo)")

# ===================== PMH SECTION (visible only after button click) =====================
st.subheader("Previous Medical History (PMH)")

if "show_pmh" not in st.session_state:
    st.session_state.show_pmh = False

pmh_cols = st.columns([1, 6])
with pmh_cols[0]:
    if not st.session_state.show_pmh:
        if st.button("Show PMH", type="primary", key="pmh_show"):
            st.session_state.show_pmh = True
            st.rerun()
    else:
        if st.button("Hide PMH", key="pmh_hide"):
            st.session_state.show_pmh = False
            st.rerun()

# Render PMH only when toggled on
if st.session_state.show_pmh:
    st.success("Previous medical history")
    for item in PMH_ITEMS:
        st.markdown(f"- {item}")
    st.caption("_Problem list / PMH summary • 2024-01-05_")

st.markdown("---")

# ===================== Ask the chart (general search) =====================
query = st.text_input(
    "Ask the chart (e.g., 'When was the CABG?', 'Any contrast reactions?', 'Why apixaban?')",
    ""
)

col_ans, col_tl = st.columns([2, 1])

with col_ans:
    if query.strip():
        hits = search(query)
        if hits.empty:
            st.info("No charted evidence found for this query.")
        else:
            st.subheader("Answer (with citations)")
            terms = [t for t in re.findall(r'\w+', query) if len(t) > 2]
            for _, r in hits.iterrows():
                date_str = r['date'].date().isoformat() if pd.notnull(r['date']) else "—"
                st.markdown(f"- {highlight(r['text'], terms)}  \n  _{r['source']} • {date_str}_")
    else:
        st.caption("Type a question above to search the chart.")

with col_tl:
    st.subheader("Timeline")
    tl = df.sort_values("date", ascending=False)
    for _, r in tl.iterrows():
        if pd.isnull(r["date"]):
            continue
        st.write(f"{r['date'].date()} — {r['source']}")

st.markdown("---")
st.caption("Demo only. Verify findings in source notes before clinical decisions.")
