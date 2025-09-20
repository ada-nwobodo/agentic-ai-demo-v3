import streamlit as st
import pandas as pd, numpy as np, re, datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Medical History Search (Demo)", layout="wide")

# ---------------- Sidebar: patient context ----------------
st.sidebar.header("Patient")
demo_structured = dict(
    name="Jane Doe", sex="F", dob="1952-08-20",
    problems=["AF (paroxysmal)", "CAD s/p CABG (2018)"],
    meds=["Apixaban 5 mg BD", "Metoprolol 50 mg BD", "Atorvastatin 40 mg nocte"],
    allergies=["Amoxicillin – rash (remote)"],
    observations=dict(hr=78, bp="128/72", spo2="97% RA"),
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

# ---- Add a PMH summary note so search can cite it ----
PMH_ITEMS = [
    "Ischaemic heart disease (IHD)",
    "Heart failure",
    "Hypertension (HBP)",
    "Mitral valve disease",
    "History of binge drinking for the last 10 years",
]
pmh_text = "Previous medical history: " + "; ".join(PMH_ITEMS) + "."
df = pd.concat([
    df,
    pd.DataFrame([{"date": "2024-01-05", "source": "Problem list / PMH summary", "text": pmh_text}])
], ignore_index=True)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["age_days"] = (dt.datetime.now() - df["date"]).dt.days

# ---------------- Search index ----------------
@st.cache_data(show_spinner=False)
def build_index(texts):
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts.fillna(""))
    return vec, X

vectorizer, X = build_index(df["text"])

def search(query, k=6, half_life_days=365):
    q = vectorizer.transform([query])
    sims = linear_kernel(q, X).ravel()
    recency = 0.5 ** (df["age_days"].fillna(3650) / half_life_days)
    score = sims * (0.2 + 0.8*recency)
    out = df.assign(score=score).sort_values("score", ascending=False)
    return out[out["score"] > 0].head(k)

def highlight(text, terms):
    if not terms: return text
    pattern = r'(' + '|'.join(map(re.escape, terms)) + r')'
    return re.sub(pattern, r'**\1**', text, flags=re.I)

# ---------------- UI ----------------
st.title("Medical History Search (Demo)")

# ---- NEW: Previous Medical History section (appears ABOVE general ask) ----
with st.container():
    st.subheader("Previous Medical History (PMH)")
    pmh_prompt = st.text_input(
        "Ask for previous medical history of the patient",
        placeholder="e.g., 'previous medical history' or 'PMH'",
        key="pmh_query"
    )
    colA, colB = st.columns([1,3])
    with colA:
        show_pmh = st.button("Show PMH")
    with colB:
        st.caption("Includes IHD, heart failure, hypertension, mitral valve disease, and a 10-year history of binge drinking.")

    if pmh_prompt.strip() or show_pmh:
        st.success("Previous medical history")
        st.markdown("\n".join([f"- {item}" for item in PMH_ITEMS]))
        # Also show the PMH summary “citation” entry so it feels like the chart
        st.caption("_Problem list / PMH summary • 2024-01-05_")

st.markdown("---")

# ---- Ask the chart (general search) ----
query = st.text_input("Ask the chart (e.g., 'When was the CABG?', 'Any contrast reactions?', 'Why apixaban?')", "")

col_ans, col_tl = st.columns([2,1])

with col_ans:
    if query.strip():
        hits = search(query)
        if hits.empty:
            st.info("No charted evidence found for this query.")
        else:
            st.subheader("Answer (with citations)")
            terms = [t for t in re.findall(r'\w+', query) if len(t) > 2]
            for _, r in hits.iterrows():
                st.markdown(f"- {highlight(r['text'], terms)}  \n  _{r['source']} • {r['date'].date()}_")
    else:
        st.caption("Type a question above to search the chart.")

with col_tl:
    st.subheader("Timeline")
    tl = df.sort_values("date", ascending=False)
    for _, r in tl.iterrows():
        st.write(f"{r['date'].date()} — {r['source']}")

st.markdown("---")
st.caption("Demo only. Verify findings in source notes before clinical decisions.")



