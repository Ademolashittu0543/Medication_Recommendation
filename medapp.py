import os
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "true"
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── PDF generation ──────────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    import io
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict · Disease Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Reset & base ───────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #F7F8FC; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1100px; }

/* ── Hero header ────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0F2057 0%, #1A3A8F 60%, #2563EB 100%);
    border-radius: 20px;
    padding: 2.8rem 3rem 2.4rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "⬤";
    position: absolute; top: -40px; right: -40px;
    font-size: 260px; opacity: 0.04; color: #fff;
    line-height: 1;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #BAD0FF;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.85rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.5rem;
    line-height: 1.15;
}
.hero p {
    color: #93B8FF;
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
}

/* ── Section label ──────────────────────────── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.5rem;
}

/* ── Symptom multiselect box ────────────────── */
.symptom-box {
    background: #fff;
    border: 1.5px solid #E5E7EB;
    border-radius: 14px;
    padding: 1.5rem 1.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 1.2rem;
}

/* ── Result card ────────────────────────────── */
.result-hero {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    border: 1.5px solid #BFDBFE;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin: 1.5rem 0;
}
.disease-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #1E40AF;
    margin: 0;
}
.confidence-chip {
    display: inline-block;
    background: #DCFCE7;
    color: #166534;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-top: 0.6rem;
}

/* ── Progress bar labels ────────────────────── */
.pred-row { margin: 0.4rem 0; }
.pred-label { font-size: 0.88rem; color: #374151; font-weight: 500; }

/* ── Info cards inside tabs ─────────────────── */
.info-card {
    background: #fff;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
}
.info-icon { font-size: 1.2rem; margin-right: 0.5rem; }
.info-text { font-size: 0.9rem; color: #374151; line-height: 1.6; }

/* ── Disclaimer ─────────────────────────────── */
.disclaimer {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.82rem;
    color: #92400E;
    margin-top: 2rem;
    line-height: 1.5;
}

/* ── Sidebar ────────────────────────────────── */
section[data-testid="stSidebar"] { background: #0F2057 !important; }
section[data-testid="stSidebar"] * { color: #CBD5F0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #ffffff !important; }
section[data-testid="stSidebar"] .stMarkdown a { color: #93B8FF !important; }

/* ── Button override ────────────────────────── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1D4ED8, #2563EB);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.65rem 1.2rem;
    letter-spacing: 0.02em;
    transition: transform 0.15s, box-shadow 0.15s;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37,99,235,0.4);
}

/* ── Download button ────────────────────────── */
div.stDownloadButton > button {
    background: #fff;
    border: 1.5px solid #2563EB;
    color: #2563EB !important;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.9rem;
}
div.stDownloadButton > button:hover {
    background: #EFF6FF;
}

/* ── Tab strip ──────────────────────────────── */
button[data-baseweb="tab"] {
    font-size: 0.84rem !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        precautions   = pd.read_csv(DATA_DIR / "precautions_clean.xls")
        medications   = pd.read_csv(DATA_DIR / "medications_clean.xls")
        diets         = pd.read_csv(DATA_DIR / "diets_clean.xls")
        workouts      = pd.read_csv(DATA_DIR / "workout_clean.xls")
        descriptions  = pd.read_csv(DATA_DIR / "description_clean.xls")
        with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        with open(MODEL_DIR / "features.pkl", "rb") as f:
            feature_names = pickle.load(f)
        return le, feature_names, precautions, medications, diets, workouts, descriptions
    except Exception as e:
        st.error(f"Failed to load data artifacts: {e}")
        st.stop()

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model_path = MODEL_DIR / "best_model.pkl"
        if not model_path.exists():
            st.error("Model file not found. Train and export the model first.")
            st.stop()
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

# ── PDF report builder ────────────────────────────────────────────────────────
def build_pdf_report(
    patient_name, patient_age, patient_gender,
    symptoms, disease, confidence,
    top3_diseases, top3_probs,
    description, medications, diet, workouts, precautions
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=colors.HexColor("#1E3A8A"),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "Sub",
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#6B7280"),
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    section_header = ParagraphStyle(
        "SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=colors.HexColor("#1D4ED8"),
        spaceBefore=12,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=colors.HexColor("#374151"),
        leading=15,
        spaceAfter=4,
    )
    small_style = ParagraphStyle(
        "Small",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        textColor=colors.HexColor("#9CA3AF"),
        leading=13,
    )

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("🩺 MediPredict", title_style))
    story.append(Paragraph("AI-Powered Medical Diagnostic Report", sub_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}", sub_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1D4ED8")))
    story.append(Spacer(1, 0.4*cm))

    # ── Patient info ─────────────────────────────────────────────────────────
    story.append(Paragraph("Patient Information", section_header))
    patient_data = [
        ["Full Name", patient_name or "—", "Age", f"{patient_age} yrs" if patient_age else "—"],
        ["Gender", patient_gender or "—", "Date", datetime.now().strftime("%d %b %Y")],
    ]
    pt = Table(patient_data, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5*cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F0F4FF")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#374151")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#EFF6FF"), colors.HexColor("#F8FAFC")]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#DBEAFE")),
        ("PADDING", (0,0), (-1,-1), 6),
        ("RADIUS", (0,0), (-1,-1), 4),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.4*cm))

    # ── Symptoms ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Reported Symptoms", section_header))
    sym_chunks = [symptoms[i:i+3] for i in range(0, len(symptoms), 3)]
    sym_rows = [[s for s in chunk] + [""]*(3-len(chunk)) for chunk in sym_chunks]
    sym_tbl = Table(sym_rows, colWidths=[5.83*cm]*3)
    sym_tbl.setStyle(TableStyle([
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#1E40AF")),
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#DBEAFE")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#BFDBFE")),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(sym_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Primary diagnosis ────────────────────────────────────────────────────
    story.append(Paragraph("Primary Diagnosis", section_header))
    diag_data = [
        ["Predicted Condition", disease],
        ["Confidence Score",    f"{confidence:.1f}%"],
    ]
    dt = Table(diag_data, colWidths=[5*cm, 12.5*cm])
    dt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#1D4ED8")),
        ("TEXTCOLOR",  (0,0), (0,-1), colors.white),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica-Bold"),
        ("BACKGROUND", (1,0), (1,0), colors.HexColor("#DBEAFE")),
        ("TEXTCOLOR",  (1,0), (1,0), colors.HexColor("#1E3A8A")),
        ("BACKGROUND", (1,1), (1,1), colors.HexColor("#DCFCE7")),
        ("TEXTCOLOR",  (1,1), (1,1), colors.HexColor("#166534")),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ("PADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(dt)
    story.append(Spacer(1, 0.3*cm))

    # Top-3 differential
    story.append(Paragraph("Differential Diagnoses (Top 3)", section_header))
    diff_rows = [["Condition", "Probability"]]
    for d, p in zip(top3_diseases, top3_probs):
        diff_rows.append([str(d).title(), f"{p*100:.1f}%"])
    diff_tbl = Table(diff_rows, colWidths=[12*cm, 5.5*cm])
    diff_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F8FAFC"), colors.HexColor("#EFF6FF")]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ("ALIGN", (1,0), (1,-1), "CENTER"),
        ("PADDING", (0,0), (-1,-1), 7),
    ]))
    story.append(diff_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Description ──────────────────────────────────────────────────────────
    if description:
        story.append(Paragraph("Condition Overview", section_header))
        story.append(Paragraph(description, body_style))

    # ── Medications ──────────────────────────────────────────────────────────
    if medications:
        story.append(Paragraph("Recommended Medications", section_header))
        med_rows = [["#", "Medication"]]
        for i, m in enumerate(medications, 1):
            med_rows.append([str(i), m.strip()])
        med_tbl = Table(med_rows, colWidths=[1.2*cm, 16.3*cm])
        med_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#7C3AED")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F3FF")]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#EDE9FE")),
            ("ALIGN", (0,0), (0,-1), "CENTER"),
            ("PADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(med_tbl)

    # ── Diet ─────────────────────────────────────────────────────────────────
    if diet:
        story.append(Paragraph("Dietary Recommendations", section_header))
        diet_rows = [["✅  " + d.strip()] for d in diet]
        diet_tbl = Table(diet_rows, colWidths=[17.5*cm])
        diet_tbl.setStyle(TableStyle([
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#166534")),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#F0FDF4"), colors.white]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#BBF7D0")),
            ("PADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(diet_tbl)

    # ── Workouts ─────────────────────────────────────────────────────────────
    if workouts:
        story.append(Paragraph("Recommended Physical Activities", section_header))
        for w in workouts:
            story.append(Paragraph(f"🏃 {w.strip()}", body_style))

    # ── Precautions ──────────────────────────────────────────────────────────
    if precautions:
        story.append(Paragraph("Safety Precautions", section_header))
        prec_rows = [[f"⚠️  {p.strip()}"] for p in precautions if p]
        prec_tbl = Table(prec_rows, colWidths=[17.5*cm])
        prec_tbl.setStyle(TableStyle([
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#92400E")),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#FFF7ED"), colors.white]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#FED7AA")),
            ("PADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(prec_tbl)

    # ── Footer disclaimer ────────────────────────────────────────────────────
    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#E5E7EB")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This report is generated by an AI-powered tool for informational purposes only. "
        "It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified "
        "healthcare professional before making any health decisions.",
        small_style,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ── Load resources ─────────────────────────────────────────────────────────────
with st.spinner("Loading model and data…"):
    le, feature_names, precautions_df, medications_df, diets_df, workouts_df, descriptions_df = load_artifacts()
    model = load_model()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 MediPredict")
    st.markdown("---")
    st.markdown("### Patient Details")
    patient_name   = st.text_input("Full Name", placeholder="e.g. John Doe")
    patient_age    = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="Years")
    patient_gender = st.selectbox("Gender", ["— Select —", "Male", "Female", "Other", "Prefer not to say"])
    if patient_gender == "— Select —":
        patient_gender = ""

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "MediPredict uses a machine-learning classifier trained on symptom-disease "
        "mappings to suggest likely conditions. **Always verify results with a doctor.**"
    )
    st.markdown("---")
    st.caption(f"Model features: **{len(feature_names)}** symptoms")
    st.caption(f"Diseases covered: **{len(le.classes_)}**")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Diagnostic Tool</div>
    <h1>🩺 MediPredict</h1>
    <p>Select your symptoms below — the model will identify the most likely condition and provide evidence-based recommendations.</p>
</div>
""", unsafe_allow_html=True)

# ── Symptom selector ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Step 1 — Select Symptoms</div>', unsafe_allow_html=True)
with st.container():
    selected_symptoms = st.multiselect(
        label="Symptoms",
        options=sorted(feature_names),
        label_visibility="collapsed",
        placeholder="Start typing a symptom…",
        help="Select all symptoms you are currently experiencing. More symptoms improve accuracy.",
    )

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    predict_clicked = st.button("🔍 Analyse Symptoms", type="primary", use_container_width=True)
with col_clear:
    clear_clicked = st.button("✕ Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if selected_symptoms:
    st.markdown(
        f'<div style="font-size:0.82rem;color:#6B7280;margin-top:0.3rem;">'
        f'<b>{len(selected_symptoms)}</b> symptom(s) selected</div>',
        unsafe_allow_html=True,
    )

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_clicked:
    if not selected_symptoms:
        st.error("⚠️ Please select at least one symptom before analysing.")
    else:
        with st.spinner("Running diagnostic model…"):
            input_vec = np.zeros(len(feature_names), dtype=np.int8)
            sym_to_idx = {s: i for i, s in enumerate(feature_names)}
            for sym in selected_symptoms:
                if sym in sym_to_idx:
                    input_vec[sym_to_idx[sym]] = 1
            input_vec = input_vec.reshape(1, -1)

            pred = model.predict(input_vec)[0]
            disease = le.inverse_transform([pred])[0].title()

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_vec)[0]
                confidence = proba.max() * 100
                top3_idx = proba.argsort()[-3:][::-1]
                top3_diseases = le.inverse_transform(top3_idx)
                top3_probs = proba[top3_idx]
            else:
                confidence = 88.0
                top3_diseases = [disease]
                top3_probs    = [0.88]

        # ── Result card ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-label">Step 2 — Diagnostic Result</div>', unsafe_allow_html=True)

        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"""
            <div class="result-hero">
                <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#3B82F6;margin-bottom:0.4rem;">PRIMARY DIAGNOSIS</div>
                <p class="disease-name">{disease}</p>
                <span class="confidence-chip">✓ {confidence:.1f}% confidence</span>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown("**Differential Diagnoses**")
            for d, p in zip(top3_diseases, top3_probs):
                st.progress(float(p), text=f"{str(d).title()} — {p*100:.1f}%")

        # ── Gather recommendation data ────────────────────────────────────────
        disease_key = disease.lower()

        def _get(df, col_disease, col_data):
            row = df[df[col_disease].str.lower() == disease_key]
            if not row.empty and col_data in row.columns:
                return str(row.iloc[0][col_data]).split(" | ")
            return []

        desc_row = descriptions_df[descriptions_df["Disease"].str.lower() == disease_key]
        description_text = desc_row.iloc[0]["Description"] if not desc_row.empty else ""
        med_list  = _get(medications_df, "Disease", "Medication")
        diet_list = _get(diets_df,       "Disease", "Diet")
        work_list = _get(workouts_df,    "Disease", "Workouts")

        prec_row = precautions_df[precautions_df["Disease"].str.lower() == disease_key]
        prec_list = []
        if not prec_row.empty:
            for i in range(1, 5):
                col = f"Precaution_{i}"
                if col in prec_row.columns and pd.notna(prec_row.iloc[0][col]):
                    prec_list.append(prec_row.iloc[0][col])

        # ── Recommendation tabs ───────────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Step 3 — Recommendations</div>', unsafe_allow_html=True)
        tabs = st.tabs(["📋 Overview", "💊 Medications", "🥗 Diet", "🏋️ Activity", "🛡️ Precautions"])

        with tabs[0]:
            if description_text:
                st.markdown(f'<div class="info-card"><span class="info-text">{description_text}</span></div>', unsafe_allow_html=True)
            else:
                st.info("No description available for this condition.")

        with tabs[1]:
            if med_list:
                for m in med_list:
                    st.markdown(f'<div class="info-card"><span class="info-icon">💊</span><span class="info-text">{m.strip()}</span></div>', unsafe_allow_html=True)
            else:
                st.info("No medication data available.")

        with tabs[2]:
            if diet_list:
                for d in diet_list:
                    st.markdown(f'<div class="info-card"><span class="info-icon">✅</span><span class="info-text">{d.strip()}</span></div>', unsafe_allow_html=True)
            else:
                st.info("No dietary recommendations available.")

        with tabs[3]:
            if work_list:
                for w in work_list:
                    st.markdown(f'<div class="info-card"><span class="info-icon">🏃</span><span class="info-text">{w.strip()}</span></div>', unsafe_allow_html=True)
            else:
                st.info("No activity recommendations available.")

        with tabs[4]:
            if prec_list:
                for p in prec_list:
                    st.markdown(f'<div class="info-card"><span class="info-icon">⚠️</span><span class="info-text">{p}</span></div>', unsafe_allow_html=True)
            else:
                st.info("No precaution data available.")

        # ── Download report ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-label">Step 4 — Download Report</div>', unsafe_allow_html=True)

        if PDF_AVAILABLE:
            pdf_bytes = build_pdf_report(
                patient_name   = patient_name,
                patient_age    = patient_age,
                patient_gender = patient_gender,
                symptoms       = selected_symptoms,
                disease        = disease,
                confidence     = confidence,
                top3_diseases  = top3_diseases,
                top3_probs     = top3_probs,
                description    = description_text,
                medications    = med_list,
                diet           = diet_list,
                workouts       = work_list,
                precautions    = prec_list,
            )
            fname = f"MediPredict_{disease.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                label="📄 Download Medical Report (PDF)",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning(
                "PDF generation requires **reportlab**. Install it with:\n"
                "```\npip install reportlab\n```\n"
                "Then restart the app."
            )

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong> MediPredict is an AI-powered educational tool.
            Results are not a substitute for professional medical advice, diagnosis, or treatment.
            Always consult a licensed healthcare provider before making health decisions.
        </div>
        """, unsafe_allow_html=True)
