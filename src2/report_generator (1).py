from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from gtts import gTTS
import io

def generate_pdf_report(info, confidence, nutrient_status, nutrient_conf):
    """Creates a detailed PDF report in memory from all info."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    story.append(Paragraph("Plant Health Analysis Report", styles['h1']))
    story.append(Spacer(1, 12))
    
    # --- Disease Section ---
    story.append(Paragraph("<b>Disease Analysis (from RGB Image)</b>", styles['h2']))
    story.append(Paragraph(f"<b>Prediction:</b> {info['name']}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>Recommendation:</b>", styles['h3']))
    story.append(Paragraph(info['recommendation'], styles['Normal']))
    story.append(Spacer(1, 12))

    # --- Nutrient Section ---
    story.append(Paragraph("<b>Nutrient Analysis (from Hyperspectral)</b>", styles['h2']))
    story.append(Paragraph(f"<b>Fertilizer Level:</b> {nutrient_status}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {nutrient_conf:.2f}%", styles['Normal']))
    story.append(Spacer(1, 12))
    
    if "Deficient" in nutrient_status:
        rec = "Nutrient levels appear low. A balanced NPK (Nitrogen, Phosphorus, Potassium) fertilizer is recommended."
    else:
        rec = "Nutrient levels appear sufficient. No immediate fertilizer application is needed."
    story.append(Paragraph(f"<b>Recommendation:</b> {rec}", styles['Normal']))

    
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_audio_report(info, nutrient_status, lang='en'):
    """Creates an MP3 audio report for both results."""
    if lang == 'hi':
        text_to_speak = f"Rog ka natija: {info['name']}. Poshan ka natija: Fertilizer {nutrient_status} hai."
    else:
        text_to_speak = f"Disease Prediction: {info['name']}. Nutrient Prediction: Fertilizer level is {nutrient_status}."
        
    tts = gTTS(text=text_to_speak, lang=lang)
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer