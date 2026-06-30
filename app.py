import gradio as gr
import numpy as np
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ================= LOAD MODEL =================
model = joblib.load("model.pkl")


# ================= HEALTH STATUS =================
def health_plan(risk):
    if risk >= 70:
        return "🔴 HIGH RISK", "#DC2626"
    elif risk >= 30:
        return "🟠 MODERATE RISK", "#F59E0B"
    else:
        return "🟢 LOW RISK", "#22C55E"


# ================= PDF GENERATION =================
def create_pdf(content_text):
    file_path = "/tmp/heart_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []
    for line in content_text.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 5))

    doc.build(content)
    return file_path


# ================= PREDICT FUNCTION =================
def predict(name, pid, age, sex,
            cp, trestbps, chol, fbs,
            restecg, thalach, exang,
            oldpeak, slope, ca, thal):

    sex_val = 1 if sex == "Male" else 0

    data = np.array([[age, sex_val, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak,
                      slope, ca, thal]])

    prob = model.predict_proba(data)[0][1]
    risk = float(prob * 100)

    status, color = health_plan(risk)

    # ================= HTML OUTPUT =================
    html = f"""
    <div style="
        background:white;
        padding:25px;
        border-radius:20px;
        border-left:10px solid {color};
        box-shadow:0 10px 25px rgba(0,0,0,0.12);
        font-family:Arial;
    ">
    <h2 style="color:{color};">{status}</h2>
    <h3>📊 Risk Score: {risk:.2f}%</h3>
    <hr>
    <h3>👤 Patient Details</h3>
    <p><b>Name:</b> {name}</p>
    <p><b>ID:</b> {pid}</p>
    <p><b>Age:</b> {age}</p>
    <p><b>Gender:</b> {sex}</p>
    <hr>
    <h3>🥗 Health Advice</h3>
    <p>Vegetables, Fruits, Nuts, Fish<br>
    Avoid fried food, sugar, processed food</p>
    <h3>🏃 Exercise</h3>
    <p>Walking 30–40 min/day, Yoga, Cycling</p>
    <h3>😴 Sleep</h3>
    <p>7–8 hours daily</p>
    <h3>⚠ Medical Advice</h3>
    <p>Consult doctor if symptoms persist</p>
    </div>
    """

    # ================= PDF CONTENT =================
    pdf_text = f"""
HEART ATTACK RISK REPORT
PATIENT DETAILS
Name: {name}
ID: {pid}
Age: {age}
Gender: {sex}
RESULT
Status: {status}
Risk Score: {risk:.2f}%
RECOMMENDATIONS
- Eat healthy diet
- Exercise daily (30-40 min)
- Sleep 7-8 hours
- Avoid junk food
NOTE: AI generated report
"""

    pdf_path = create_pdf(pdf_text)

    return html, pdf_path


# ================= UI =================
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan")
) as app:

    gr.Markdown("""
    # 🫀 Heart Attack Prediction AI  
    ### Professional Medical Risk Dashboard
    """)

    # -------- PATIENT DETAILS --------
    gr.Markdown("## 👤 Patient Details")

    with gr.Row():
        name = gr.Textbox(label="Patient Name")
        pid = gr.Textbox(label="Patient ID")
        age = gr.Number(label="Age", value=30)

    sex = gr.Dropdown(["Male", "Female"], label="Gender")

    # -------- MEDICAL INPUTS --------
    gr.Markdown("## ❤️ Medical Parameters")

    with gr.Row():
        cp = gr.Number(label="Chest Pain (0-3)", value=1)
        trestbps = gr.Number(label="BP", value=120)
        chol = gr.Number(label="Cholesterol", value=200)

    with gr.Row():
        fbs = gr.Number(label="Sugar (0/1)", value=0)
        restecg = gr.Number(label="ECG (0-2)", value=1)
        thalach = gr.Number(label="Max Heart Rate", value=150)

    with gr.Row():
        exang = gr.Number(label="Angina (0/1)", value=0)
        oldpeak = gr.Number(label="Oldpeak", value=1.0)
        slope = gr.Number(label="Slope (0-2)", value=1)

    with gr.Row():
        ca = gr.Number(label="Vessels (0-3)", value=0)
        thal = gr.Number(label="Thal (0-3)", value=2)

    # -------- BUTTON --------
    btn = gr.Button("🔍 Analyze Risk", variant="primary")

    output_html = gr.HTML()
    output_pdf = gr.File(label="📄 Download Medical Report")

    # -------- FUNCTION CALL --------
    btn.click(
        predict,
        inputs=[name, pid, age, sex,
                cp, trestbps, chol, fbs,
                restecg, thalach, exang,
                oldpeak, slope, ca, thal],
        outputs=[output_html, output_pdf]
    )

app.launch()
