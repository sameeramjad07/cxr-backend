import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

def markdown_to_html(text):
    # Convert **bold** to <b>...</b>
    def repl(match):
        return f"<b>{match.group(1)}</b>"
    return re.sub(r"\*\*(.*?)\*\*", repl, text)

def _call_gemini(prompt, llm_model):
    return llm_model.generate_content(prompt).text

async def generate_report(predictions, class_labels):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDl-nEytKjUk8hyHcoQPvlOrbsKDmt9JUk")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Enhanced prompt for structured and formatted output
    condition_data = ', '.join([f'{label}: {prob:.2f}' for label, prob in zip(class_labels, predictions)])
    prompt = f"""
You are a medical AI assistant generating a detailed chest X-ray analysis report.

Structure the report with:

- **Summary**: Key conditions with probabilities > 0.5
- **Detailed Findings**: All conditions with probabilities and interpretations
- **Recommendations**: Suggested next steps

Here are the probabilities:
{condition_data}

Format with Markdown-style headings (e.g., **Summary**) and bullet points.
"""

    loop = asyncio.get_event_loop()
    try:
        with ThreadPoolExecutor() as pool:
            response_text = await asyncio.wait_for(
                loop.run_in_executor(pool, _call_gemini, prompt, llm_model),
                timeout=20
            )

            # Sanity check
            if "**Summary**" not in response_text or "**Detailed Findings**" not in response_text:
                raise Exception("Gemini response is incomplete or malformed")

            return response_text
    except asyncio.TimeoutError:
        raise Exception("Gemini API call timed out")
    except Exception as e:
        raise Exception(f"Gemini API failed: {str(e)}")

def create_pdf_report(report_text, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(name='CustomHeading1', fontName='Helvetica-Bold', fontSize=18, textColor=colors.darkblue, spaceAfter=12))
    styles.add(ParagraphStyle(name='CustomHeading2', fontName='Helvetica-Bold', fontSize=14, textColor=colors.black, spaceAfter=6))
    styles.add(ParagraphStyle(name='CustomBodyText', fontName='Helvetica', fontSize=12, textColor=colors.black, leading=14))
    styles.add(ParagraphStyle(name='CustomBoldText', fontName='Helvetica-Bold', fontSize=12, textColor=colors.black, leading=14))
    styles.add(ParagraphStyle(name='CustomDisclaimer', fontName='Helvetica-Oblique', fontSize=10, textColor=colors.grey, spaceBefore=12))

    story = []

    # Header with logo placeholder
    # Note: Replace with actual image path if available
    try:
        story.append(Paragraph("<b>Chest X-ray Analysis Report</b>", styles['CustomHeading1']))
        story.append(Paragraph("Patient ID: CXR-2025-001 | Generated: May 19, 2025", styles['CustomBodyText']))
        story.append(Spacer(1, 24))
    except Exception as e:
        print(f"Warning: Header setup failed: {str(e)}")

    # Parse the report text and format it
    lines = report_text.split('\n')
    current_section = None
    findings_table_data = [['Condition', 'Probability', 'Interpretation']]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect sections
        if line.startswith("**Summary**"):
            current_section = 'summary'
            story.append(Paragraph("Summary", styles['CustomHeading2']))
            story.append(Spacer(1, 6))
        elif line.startswith("**Detailed Findings**"):
            current_section = 'findings'
            story.append(Paragraph("Detailed Findings", styles['CustomHeading2']))
            story.append(Spacer(1, 6))
        elif line.startswith("**Recommendations**"):
            current_section = 'recommendations'
            if len(findings_table_data) > 1:
                table = Table(findings_table_data, colWidths=[2*inch, 1*inch, 2.5*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ]))
                story.append(table)
                story.append(Spacer(1, 12))
            story.append(Paragraph("Recommendations", styles['CustomHeading2']))
            story.append(Spacer(1, 6))
        elif line.startswith("- ") and current_section == 'findings':
            try:
                label_part = line[2:]
                label, prob_str = label_part.split(':')
                prob = float(prob_str.strip())
                interpretation = "⚠️ Significant finding" if prob > 0.5 else "Normal"
                findings_table_data.append([label.strip(), f"{prob:.2%}", interpretation])
            except ValueError:
                continue  # skip malformed lines
        elif current_section in ['summary', 'recommendations']:
            safe_line = markdown_to_html(line.strip())
            story.append(Paragraph(safe_line, styles['CustomBodyText']))
            story.append(Spacer(1, 6))


    # Add disclaimer footer
    story.append(PageBreak())
    story.append(Paragraph("Disclaimer: This report is AI-generated and intended for informational purposes only. Consult a qualified healthcare professional for diagnosis and treatment.", styles['CustomDisclaimer']))

    # Build the document
    doc.build(story)