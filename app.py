# app.py — Medical History Search (Demo) with PMH section (shown only on button click)

import streamlit as st
import pandas as pd
import numpy as np
import re
import datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os, tempfile, io
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk


st.set_page_config(page_title="Medical History Search (Demo)", layout="wide")

# ---- Supabase client (uses your ANON key to match your working RLS) ----
from supabase import create_client, Client
import uuid

SB_URL = st.secrets["SUPABASE_URL"]
SB_KEY = st.secrets["SUPABASE_ANON_KEY"]  # matches your anon RLS policies
sb: Client = create_client(SB_URL, SB_KEY)

# ---- Session id (one per app session) ----
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())





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

# Optional: Upload a lab results file (or use demo labs if not uploaded)
labs_upload = st.sidebar.file_uploader(
    "Upload labs CSV (date,encounter,urea,creatinine,bnp,hgb)",
    type=["csv"],
    key="labs_csv"
)

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

# ---------------- Demo or uploaded labs ----------------
if labs_upload is not None:
    labs = pd.read_csv(labs_upload)
else:
    # Demo labs (simulate last few hospital attendances)
    labs = pd.DataFrame([
        {"date": "2023-10-02", "encounter": "Admission", "urea": 6.0, "creatinine": 82,  "bnp": 180, "hgb": 126},
        {"date": "2023-12-15", "encounter": "ED",        "urea": 6.8, "creatinine": 95,  "bnp": 210, "hgb": 123},
        {"date": "2024-03-28", "encounter": "Admission", "urea": 7.5, "creatinine": 112, "bnp": 260, "hgb": 119},
        {"date": "2024-07-09", "encounter": "ED",        "urea": 8.9, "creatinine": 132, "bnp": 340, "hgb": 115},
        {"date": "2024-11-21", "encounter": "Admission", "urea": 9.6, "creatinine": 145, "bnp": 420, "hgb": 111},
        {"date": "2025-02-04", "encounter": "ED",        "urea": 10.2,"creatinine": 158, "bnp": 510, "hgb": 108},
        {"date": "2025-06-18", "encounter": "Admission", "urea": 11.3,"creatinine": 171, "bnp": 615, "hgb": 104},
    ])
labs["date"] = pd.to_datetime(labs["date"], errors="coerce")
labs = labs.dropna(subset=["date"]).sort_values("date")

# ---- Helpers to compute & render abnormal trends (last 5 attendances) ----
def last5_attendances(labs_df: pd.DataFrame):
    # Take the last 5 rows by date (assumes each row is an attendance)
    return labs_df.sort_values("date").tail(5).reset_index(drop=True)

def trend_delta(series: pd.Series, inverse: bool = False):
    """Return (delta, direction_str). inverse=True => downward is 'worse' (e.g., haemoglobin)."""
    if series.dropna().shape[0] < 2:
        return 0.0, "no change"
    delta = float(series.iloc[-1] - series.iloc[0])
    if inverse:
        direction = "worse" if delta < 0 else ("better" if delta > 0 else "no change")
    else:
        direction = "worse" if delta > 0 else ("better" if delta < 0 else "no change")
    return delta, direction

def render_abnormal_trends(labs_df: pd.DataFrame):
    data5 = last5_attendances(labs_df)
    st.success("Abnormal trends over last 5 hospital attendances")

    # Headline metrics (latest value + delta vs first of the five)
    m1, m2, m3 = st.columns(3)
    d_u, _  = trend_delta(data5["urea"])                   # higher = worse
    d_cr, _ = trend_delta(data5["creatinine"])             # higher = worse
    d_hb, _ = trend_delta(data5["hgb"], inverse=True)      # lower  = worse
    m1.metric("Urea",        f"{data5['urea'].iloc[-1]:.1f}",  f"{d_u:+.1f} vs first")
    m2.metric("Creatinine",  f"{data5['creatinine'].iloc[-1]:.0f}", f"{d_cr:+.0f} vs first")
    m3.metric("Haemoglobin", f"{data5['hgb'].iloc[-1]:.0f}",   f"{d_hb:+.0f} vs first")

    st.caption("Deterioration flagged when urea/creatinine rise, BNP rises, and haemoglobin falls.")

    # Plots
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Renal function")
        st.line_chart(data5.set_index("date")[["urea","creatinine"]])
    with c2:
        st.subheader("BNP (rising is worse)")
        st.line_chart(data5.set_index("date")[["bnp"]])

    st.subheader("Haemoglobin (falling is worse)")
    st.line_chart(data5.set_index("date")[["hgb"]])

    # Compact table of the 5 encounters
    st.dataframe(
        data5[["date","encounter","urea","creatinine","bnp","hgb"]]
           .rename(columns={"hgb":"haemoglobin"})
           .style.format({"urea":"{:.1f}","creatinine":"{:.0f}","bnp":"{:.0f}","haemoglobin":"{:.0f}"}),
        use_container_width=True,
    )


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

#Track UI toggle states
if "show_pmh" not in st.session_state:
    st.session_state.show_pmh = False
if "show_trends" not in st.session_state:
    st.session_state.show_trends = False    

#Two buttons side by side: Show PMH and Show Abnormal Trends
pmh_cols = st.columns([1, 1, 6])

with pmh_cols[0]:
    if not st.session_state.show_pmh:
        if st.button("Show PMH", type="primary", key="pmh_show"):
            # ✅ Log button click to Supabase
            sb.table("demo_events").insert({
                "session_id": st.session_state["session_id"],
                "user_id": "anon",
                "demo_name": "med_history_demo",
                "event_type": "button_click",
                "feature_name": "show_pmh",
                "payload": {"action": "show", "stage_number": 1}
            }).execute()
            
            #Then update the state and rerun
            st.session_state.show_pmh = True
            st.rerun()
    else:
        if st.button("Hide PMH", key="pmh_hide"):
            # ✅ Log "Hide PMH" button click
            sb.table("demo_events").insert({
                "session_id": st.session_state["session_id"],
                "user_id": "anon",
                "demo_name": "med_history_demo",
                "event_type": "button_click",
                "feature_name": "show_pmh",
                "payload": {"action": "hide", "stage_number": 1}
            }).execute()
 
            st.session_state.show_pmh = False
            st.rerun()

# ===================== "Show Abnormal Trends" button =====================
with pmh_cols[1]:
    if not st.session_state.show_trends:
        if st.button("Show Abnormal Trends", key="trends_show"):
            # ✅ Log "Show Abnormal Trends" button click
            sb.table("demo_events").insert({
                "session_id": st.session_state["session_id"],
                "user_id": "anon",
                "demo_name": "med_history_demo",
                "event_type": "button_click",
                "feature_name": "show_abnormal_trends",
                "payload": {"action": "show", "stage_number": 2}
            }).execute()
            
            st.session_state.show_trends = True
            st.rerun()
    else:
        if st.button("Hide Trends", key="trends_hide"):
            # ✅ Log "Hide Abnormal Trends" button click
            sb.table("demo_events").insert({
                "session_id": st.session_state["session_id"],
                "user_id": "anon",
                "demo_name": "med_history_demo",
                "event_type": "button_click",
                "feature_name": "show_abnormal_trends",
                "payload": {"action": "hide", "stage_number": 2}
            }).execute()
            
            st.session_state.show_trends = False
            st.rerun()

# ---------------- Render the PMH and Trends sections ----------------
if st.session_state.show_pmh:
    st.success("Previous medical history")
    for item in PMH_ITEMS:
        st.markdown(f"- {item}")
    st.caption("_Problem list / PMH summary • 2024-01-05_")

# Render abnormal trends only when toggled on
if st.session_state.show_trends:
    render_abnormal_trends(labs)

st.markdown("---")


# ===================== Ask the chart (general search) =====================

query = st.text_input(
    "Ask the chart (e.g., 'When was the CABG?', 'Any contrast reactions?', 'Why apixaban?')",
    ""
)

col_ans, col_tl = st.columns([2, 1])

with col_ans:
    if query.strip():
        # ---- run the search & time it ----
        t0 = dt.datetime.utcnow()
        hits = search(query)
        latency_ms = int((dt.datetime.utcnow() - t0).total_seconds() * 1000)
        k_hits = int(hits.shape[0])

        # ---- log the search event (stage 3) ----
        try:
            sb.table("demo_events").insert({
                "session_id": st.session_state["session_id"],
                "user_id": "anon",
                "demo_name": "med_history_demo",
                "event_type": "search",
                "feature_name": "ask_the_chart",
                "payload": {
                    "stage_number": 3,
                    "q": query,
                    "k": k_hits,
                    "latency_ms": latency_ms,
                    # include a tiny sample of returned text (truncated, de-identified)
                    "hits_sample": [str(t)[:120] for t in hits["text"].head(3).fillna("").tolist()]
                }
            }).execute()
        except Exception:
            pass

        # ---- render results (unchanged) ----
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


# ===================== 12-lead ECG Rhythm Check (upload & vet) =====================

st.subheader("12-lead ECG rhythm check")

st.caption("Upload a WFDB record (.hea + .dat). We'll render a 3×4 ECG with a full lead II rhythm strip, "
           "estimate the rhythm, and (if present) compare to a trusted label in the header (e.g., '#Dx:' from PTB-XL).")

col_u1, col_u2 = st.columns(2)
with col_u1:
    f_hea = st.file_uploader("Header (.hea)", type=["hea"], key="ecg_hea")
with col_u2:
    f_dat = st.file_uploader("Signal (.dat)", type=["dat"], key="ecg_dat")

# ---- INSERT A: log ECG file uploads (minimal) ----
if "last_ecg_upload" not in st.session_state:
    st.session_state.last_ecg_upload = (None, None)

_curr = (getattr(f_hea, "name", None), getattr(f_dat, "name", None))
if _curr != st.session_state.last_ecg_upload and (f_hea or f_dat):
    try:
        sb.table("demo_events").insert({
            "session_id": st.session_state["session_id"],
            "user_id": "anon",
            "demo_name": "med_history_demo",
            "event_type": "file_upload",
            "feature_name": "ecg_wfdb",
            "payload": {
                "stage_number": 4,
                "has_header": bool(f_hea),
                "has_signal": bool(f_dat),
                "header_name": _curr[0],
                "signal_name": _curr[1],
            }
        }).execute()
    except Exception:
        pass
    st.session_state.last_ecg_upload = _curr
# ---- END INSERT A ----


def _save_temp_pair(fhea, fdat):
    """Save uploaded WFDB pair to a temp folder and return (tmpdir, record_name_base)."""
    tmpdir = tempfile.mkdtemp()
    # Attempt to derive record_name from .hea filename
    rec_base = os.path.splitext(fhea.name)[0] if fhea else "uploaded"
    hea_path = os.path.join(tmpdir, f"{rec_base}.hea")
    dat_path = os.path.join(tmpdir, f"{rec_base}.dat")
    with open(hea_path, "wb") as out:
        out.write(fhea.read())
    with open(dat_path, "wb") as out:
        out.write(fdat.read())
    return tmpdir, rec_base

def _units_to_mV(signal, units):
    """Convert per-channel units to mV."""
    scale = np.ones(signal.shape[1], dtype=float)
    if units:
        for i, u in enumerate(units):
            if not u:
                continue
            ul = u.replace("µ","u").lower()
            if ul == "uv": scale[i] = 1e-3
            elif ul == "mv": scale[i] = 1.0
            elif ul == "v": scale[i] = 1e3
            else: scale[i] = 1.0
    return signal * scale

def _lead_index(sig_names, target):
    # robust caseless matching for I, II, III, aVR/AVL/AVF, V1–V6
    aliases = {
        "I": ["i","lead i","leadi"], "II": ["ii","lead ii","leadii"], "III": ["iii","lead iii","leadiii"],
        "aVR": ["avr","vr"], "aVL": ["avl","vl"], "aVF": ["avf","vf"],
        "V1":["v1"],"V2":["v2"],"V3":["v3"],"V4":["v4"],"V5":["v5"],"V6":["v6"]
    }
    names = [s.lower().strip() for s in sig_names]
    for cand in [target] + aliases.get(target, []):
        c = cand.lower()
        for i, n in enumerate(names):
            if n == c:
                return i
    return None

def _plot_3x4_with_rhythm(rec, sig_mV):
    fs = rec.fs
    sig_names = rec.sig_name
    layout = [
        ["I","aVR","V1","V4"],
        ["II","aVL","V2","V5"],
        ["III","aVF","V3","V6"],
    ]
    mm_per_mV = 10
    seg_len = 2.5
    samples_seg = int(seg_len * fs)
    t_seg = np.arange(samples_seg) / fs
    row_h = 2.5 * mm_per_mV
    total_w = seg_len * 4

    fig = plt.figure(figsize=(12, 7))
    ax = plt.gca()
    ax.axis("off")

    # grid (full page)
    ymin = -(3 * row_h + 1.5 * mm_per_mV)
    ymax = 3 * mm_per_mV
    def grid(xmax, ymin, ymax):
        for x in np.arange(0, total_w+1e-6, 0.04): ax.axvline(x, color="pink", lw=0.3, zorder=0)
        for y in np.arange(ymin, ymax+1e-6, 0.1*mm_per_mV): ax.axhline(y, color="pink", lw=0.3, zorder=0)
        for x in np.arange(0, total_w+1e-6, 0.2): ax.axvline(x, color="red", lw=0.6, zorder=0)
        for y in np.arange(ymin, ymax+1e-6, 0.5*mm_per_mV): ax.axhline(y, color="red", lw=0.6, zorder=0)
    grid(total_w, ymin, ymax)

    # 3×4 leads (consecutive non-overlapping 2.5 s per column)
    for r in range(3):
        for c in range(4):
            lead = layout[r][c]
            idx = _lead_index(sig_names, lead)
            if idx is None: continue
            start = c * samples_seg
            end = min(start + samples_seg, sig_mV.shape[0])
            x = t_seg[:(end-start)] + c * seg_len
            y = sig_mV[start:end, idx] * mm_per_mV
            yoff = -(r * row_h)
            ax.plot(x, y + yoff, lw=1)
            if c == 0:
                ax.text(-0.25, yoff, lead, fontsize=9, ha="right", va="center", weight="bold")
            else:
                ax.text(c * seg_len + 0.08, yoff + 0.8*mm_per_mV, lead, fontsize=9, ha="left", va="center", weight="bold")

    # rhythm strip (lead II, 10 s)
    idx_ii = _lead_index(sig_names, "II")
    if idx_ii is not None:
        n_r = min(int(10 * fs), sig_mV.shape[0])
        xr = np.arange(n_r) / fs
        yr = sig_mV[:n_r, idx_ii] * mm_per_mV
        yoff = -(3 * row_h)
        ax.plot(xr, yr + yoff, lw=1)
        ax.text(-0.25, yoff, "II", fontsize=9, ha="right", va="center", weight="bold")

    # calibration pulse
    cal_x = [0, 0, 0.2, 0.2]
    cal_y = [0, 1*mm_per_mV, 1*mm_per_mV, 0]
    ax.plot(cal_x, np.array(cal_y) + 1.6*mm_per_mV, lw=1)

    ax.set_xlim(0, total_w); ax.set_ylim(ymin, ymax)
    ax.set_title("12-Lead ECG – 3×4 layout + Lead II rhythm (25 mm/s, 10 mm/mV)", fontsize=12)
    return fig

def _rhythm_inference(lead_ii, fs):
    """
    Lightweight rhythm suggestion:
      - R-peaks → HR, regularity (SDNN, RMSSD, CV)
      - QRS width (from delineation) for 'narrow' vs 'wide'
      - Simple rules → sinus / sinus tachy / probable AF / indeterminate
    """
    cleaned = nk.ecg_clean(lead_ii, sampling_rate=fs, method="neurokit")
    # R-peaks
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    rpeaks = info.get("ECG_R_Peaks", [])
    if len(rpeaks) < 6:
        return dict(label="Indeterminate", details="Insufficient R-peaks detected")

    rr = np.diff(rpeaks) / fs
    hr = 60.0 / np.clip(rr, 1e-6, None)
    mean_hr = float(np.nanmean(hr))
    sdnn = float(np.nanstd(rr))
    cv_rr = float(np.nanstd(rr) / (np.nanmean(rr) + 1e-6))
    rmssd = float(np.sqrt(np.nanmean(np.square(np.diff(rr)))))

    # Delineation (rough QRS width)
    try:
        sigdict = nk.ecg_delineate(cleaned, rpeaks=info, sampling_rate=fs, method="dwt")
        q_on = sigdict["ECG_Q_Peaks"]
        s_off = sigdict["ECG_S_Peaks"]
        widths = []
        for q, s in zip(q_on, s_off):
            if q is not None and s is not None and s > q:
                widths.append((s - q) / fs)
        qrs_ms = float(np.nanmedian(widths) * 1000) if widths else np.nan
    except Exception:
        qrs_ms = np.nan

    # Simple rules (demo only, not diagnostic)
    label = "Indeterminate"
    if not np.isnan(mean_hr):
        if (cv_rr < 0.10) and (60 <= mean_hr <= 100) and (np.isnan(qrs_ms) or qrs_ms < 120):
            label = "Sinus rhythm (likely)"
        elif (cv_rr < 0.10) and (mean_hr > 100) and (np.isnan(qrs_ms) or qrs_ms < 120):
            label = "Sinus tachycardia (likely)"
        elif (cv_rr >= 0.12) and (np.isnan(qrs_ms) or qrs_ms < 120):
            label = "Atrial fibrillation (probable)"
        # you can add more rules (e.g., wide QRS + regular → VT vs SVT with aberrancy)

    details = (f"HR≈{mean_hr:.0f} bpm, RR-CV={cv_rr:.2f}, SDNN={sdnn:.3f}s, RMSSD={rmssd:.3f}s, "
               f"QRS≈{qrs_ms:.0f} ms")
    return dict(label=label, details=details)

# ----- Run if both files provided -----
if f_hea and f_dat:
    try:
        # ---- INSERT B: mark ECG feature used (minimal) ----
        try:
            sb.table("demo_events").insert({
                "session_id": st.session_state["session_id"],
                "user_id": "anon",
                "demo_name": "med_history_demo",
                "event_type": "button_click",
                "feature_name": "ecg_check",
                "payload": {"stage_number": 4, "action": "run"}
            }).execute()
        except Exception:
            pass
        # ---- END INSERT B ----
        
        tmpdir, rec_base = _save_temp_pair(f_hea, f_dat)
        rec = wfdb.rdrecord(os.path.join(tmpdir, rec_base))
        sig = rec.p_signal
        sig_mV = _units_to_mV(sig, getattr(rec, "sig_units", None))

        # Plot ECG
        fig = _plot_3x4_with_rhythm(rec, sig_mV)
        st.pyplot(fig, use_container_width=True)

        # Rhythm suggestion from lead II
        idx_ii = _lead_index(rec.sig_name, "II")
        if idx_ii is not None:
            rr_res = _rhythm_inference(sig_mV[:, idx_ii], rec.fs)
            st.success(f"Rhythm suggestion: **{rr_res['label']}**")
            st.caption(rr_res["details"])
        else:
            st.info("Lead II not found; rhythm suggestion skipped.")

        # Trusted source (header) — show Dx/diagnosis if present
        trusted = None
        hdr = wfdb.rdheader(os.path.join(tmpdir, rec_base))
        # Many PhysioNet/PTB-XL headers place labels in comment lines
        if hasattr(hdr, "comments") and hdr.comments:
            for line in hdr.comments:
                if "dx" in line.lower() or "diagnos" in line.lower():
                    trusted = line.strip()
                    break
        with st.expander("Trusted source (from header)"):
            if trusted:
                st.write(trusted)
            else:
                st.write("No diagnosis label found in header comments.")

        # Agreement (very naive string match for demo)
        if trusted:
            tl = trusted.lower()
            guess = rr_res["label"].lower()
            if ("fib" in tl and "fibrillation" in guess) or \
               ("sinus" in tl and "sinus" in guess):
                st.success("Agreement with trusted label (coarse match).")
            else:
                st.warning("Prediction and trusted label may differ—review visually.")

            # ---- INSERT E: log "compare_with_trusted_source" (stage 5) ----
            try:
                agrees = (("fib" in tl and "fibrillation" in guess) or ("sinus" in tl and "sinus" in guess))
                sb.table("demo_events").insert({
                    "session_id": st.session_state["session_id"],
                    "user_id": "anon",
                    "demo_name": "med_history_demo",
                    "event_type": "view",
                    "feature_name": "compare_with_trusted_source",
                    "payload": {
                    "stage_number": 5,
                    "trusted": trusted[:160],
                    "pred": rr_res["label"],
                    "agrees": bool(agrees)
                    }
                }).execute()
            except Exception:
                pass
    # ---- END INSERT E ----

        st.caption("Demo only — not for clinical use. Validate against source ECG and clinical context.")
    except Exception as e:
        st.error(f"Failed to process ECG: {e}")
else:
    st.info("Upload both .hea and .dat files to run the rhythm check.")




# ---------------------------------------------------------------------
# Request Imaging – summary of clinical information + relevant guidelines
# ---------------------------------------------------------------------
def _latest_labs_summary(labs_df):
    try:
        if labs_df is None or labs_df.empty:
            return "Latest labs: not available."
        row = labs_df.sort_values("date").iloc[-1]
        parts = []
        if "urea" in row: parts.append(f"Urea {row['urea']:.1f} mmol/L")
        if "creatinine" in row: parts.append(f"Creatinine {row['creatinine']:.0f} µmol/L")
        if "bnp" in row: parts.append(f"BNP {row['bnp']:.0f} ng/L")
        if "hgb" in row: parts.append(f"Hb {row['hgb']:.0f} g/L")
        return "Latest labs: " + ", ".join(parts) if parts else "Latest labs: not available."
    except Exception:
        return "Latest labs: not available."

with st.expander("Request Imaging – summary of clinical information + relevant guidelines", expanded=False):
    # Pull a few structured bits if available
    name = demo_structured.get("name", "Patient")
    sex  = demo_structured.get("sex", "—")
    dob  = demo_structured.get("dob", "—")
    problems = "; ".join(demo_structured.get("problems", [])) or "—"
    meds     = "; ".join(demo_structured.get("meds", [])) or "—"
    allergies = "; ".join(demo_structured.get("allergies", [])) or "—"

    labs_line = _latest_labs_summary(labs if "labs" in locals() else None)

    st.markdown(f"""
**Clinical Summary (auto-generated draft)**  
"75 year old female with intermittent chest pain + new onset paroxysmal AF. Previous history of CAD with CABG in 2018".  
- Intended request: **Please review for appropriate imaging based on current presentation and PMH.**
""")

    st.markdown("""
**Relevant guideline reminders (for the requester to verify)**  
- **AF (NICE NG185):** Imaging particularly if there’s a change in rhythm, LV function, or suspected complications.  
- **Stable chest pain/CAD (NICE NG17):** CT coronary angiography is first-line; post-CABG graft evaluation may be indicated if clinically relevant.  
- **Iodinated contrast safety:** Check renal function (e.g., eGFR/creatinine) and prior contrast reactions; ensure hydration and medication review.
""")

if st.button("Request Imaging", type="primary", key="req_img"):
        try:
            sb.table("demo_events").insert({
                "session_id": st.session_state["session_id"],
                "user_id": st.session_state.get("user_id", "anon"),
                "demo_name": "med_history_demo",
                "event_type": "button_click",
                "feature_name": "request_imaging",
                "payload": {"stage_number": 6, "action": "submit"}
            }).execute()
        except Exception:
            pass

        try:
            if "timestamp_start" not in st.session_state:
                st.session_state["timestamp_start"] = dt.datetime.utcnow().isoformat()
            ts_end = dt.datetime.utcnow()
            ts_start = dt.datetime.fromisoformat(st.session_state["timestamp_start"])
            duration_sec = int((ts_end - ts_start).total_seconds())

            sb.table("demo_events").insert({
                "session_id": st.session_state["session_id"],
                "user_id": st.session_state.get("user_id", "anon"),
                "demo_name": "med_history_demo",
                "event_type": "session_end",
                "feature_name": None,
                "payload": {
                    "reason": "completed",
                    "timestamp_end": ts_end.isoformat(),
                    "duration_sec": duration_sec,
                    "last_info": "request_imaging_clicked"
                }
            }).execute()
        except Exception:
            pass

        st.success("Imaging request noted. Step marked complete.")


# ===================== USER FEEDBACK & FEATURE VALUE PANEL (Enhanced) =====================

st.markdown("---")
st.header("💬 Tell us what's most valuable")

st.caption("Please rank the usefulness of each feature compared with your current workflow, "
           "and estimate whether you’d adopt it and whether it could help you avoid errors or omissions.")

# Define feature list
features = [
    "Show PMH",
    "Abnormal Bloods",
    "Detailed Search Records",
    "12-Lead ECG Check",
    "PhysioNet",
    "Imaging + Guidelines",
]

# Build feedback containers
feedback_data = []
for feature in features:
    with st.expander(f"⭐ {feature}", expanded=False):
        st.markdown(f"**{feature}** – How valuable is this feature compared to your current way of working?")
        
        usefulness = st.slider(
            f"Usefulness of {feature} (1 = No benefit, 7 = Transformative)",
            1, 7, 4, key=f"usefulness_{feature}"
        )

        time_saved = st.number_input(
            f"Estimated minutes saved using {feature}",
            0, 120, 0, key=f"time_saved_{feature}"
        )

        adoption = st.radio(
            f"Would you use {feature} if integrated into your EHR?",
            ["Yes", "Maybe", "No"], horizontal=True, key=f"adoption_{feature}"
        )

        error_avoidance = st.slider(
            f"Would {feature} help you avoid omissions or errors? (1 = No, 5 = Absolutely)",
            1, 5, 3, key=f"error_avoidance_{feature}"
        )

        feedback_data.append({
            "feature": feature,
            "usefulness": usefulness,
            "time_saved": time_saved,
            "adoption": adoption,
            "error_avoidance": error_avoidance
        })

# Overall experience & comments
st.subheader("Overall experience")
overall_rating = st.slider("How valuable was this demo overall?", 1, 10, 7, key="overall_rating")
comments = st.text_area("Any additional comments or suggestions?")

# 👇 NEW FIELD — participant identifier
participant_id = st.text_input(
    "Your initials or email (to link feedback)",
    key="participant_id"
)

# Submit feedback
if st.button("Submit Feedback", type="primary", key="submit_feedback"):
    try:
        sb.table("feature_feedback").insert({
            "user_id": st.session_state.get("user_id", "anon"),
            "session_id": st.session_state.get("session_id"),
            "demo_name": "med_history_demo",
            "ranking": {
                "features": feedback_data,
                "overall_rating": overall_rating
            },
            "comments": comments
        }).execute()
        st.success("✅ Thank you for your feedback! It’s been securely recorded.")
    except Exception as e:
        st.error(f"⚠️ Failed to submit feedback: {e}")

        st.info("This section is a non-diagnostic draft. Please verify clinical details and local guideline applicability before submitting an imaging request.")
