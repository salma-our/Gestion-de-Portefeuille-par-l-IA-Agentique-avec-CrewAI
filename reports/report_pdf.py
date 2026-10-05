import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def clean_text(text: str) -> str:
    return (
        text.replace("**", "")
        .replace("###", "")
        .replace("##", "")
        .replace("#", "")
        .replace("€", "EUR")
    )


def markdown_to_pdf(markdown_text: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        spaceAfter=16,
    )

    h1_style = ParagraphStyle(
        "CustomH1",
        parent=styles["Heading1"],
        fontSize=15,
        leading=18,
        spaceBefore=14,
        spaceAfter=8,
    )

    h2_style = ParagraphStyle(
        "CustomH2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=6,
    )

    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=5,
    )

    story = []

    lines = markdown_text.splitlines()
    table_buffer = []

    def flush_table():
        nonlocal table_buffer
        if not table_buffer:
            return

        table_data = []
        for row in table_buffer:
            if "|" in row and "---" not in row:
                cells = [clean_text(c.strip()) for c in row.split("|") if c.strip()]
                if cells:
                    table_data.append(cells)

        if table_data:
            table = Table(table_data, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 8))

        table_buffer = []

    for line in lines:
        line = line.strip()

        if not line:
            flush_table()
            story.append(Spacer(1, 6))
            continue

        if "|" in line:
            table_buffer.append(line)
            continue

        flush_table()

        if line.startswith("# "):
            story.append(Paragraph(clean_text(line), title_style))
        elif line.startswith("## "):
            story.append(Paragraph(clean_text(line), h1_style))
        elif line.startswith("### "):
            story.append(Paragraph(clean_text(line), h2_style))
        elif line.startswith("- "):
            story.append(Paragraph("• " + clean_text(line[2:]), normal_style))
        else:
            story.append(Paragraph(clean_text(line), normal_style))

    flush_table()

    doc.build(story)
