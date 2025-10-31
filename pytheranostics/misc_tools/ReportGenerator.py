"""Report generation utilities for dosimetry analysis."""

import glob
import json
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def signature_block(person, styles, width=2.5 * inch, height=0.6 * inch):
    """Create a stable signature block for PDF reports.

    Placeholder for signature, line, and text.
    Returns a small table that can be inserted side-by-side with others.
    """
    # Empty row for signature space
    sig_space = Spacer(1, height)

    # Line row
    line = Table([[""]], colWidths=[width])
    line.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, -1), 1, colors.black)]))

    # Text row
    text = Paragraph(
        f"<para align=center><b>{person['name']}</b><br/>{person['title']}<br/>{person['affiliation']}</para>",
        styles["Normal"],
    )

    # Wrap everything in a column table (1 col, 3 rows)
    block = Table([[sig_space], [line], [text]], colWidths=[width])
    block.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return block


def create_dosimetry_pdf(json_file, output_file, calculated_by=None, approved_by=None):
    """Generate a dosimetry report PDF from patient JSON data.

    Parameters
    ----------
    json_file : str or Path
        Path to the patient's JSON file.
    output_file : str or Path
        Path to save the generated PDF report.
    calculated_by : list of dict, optional
        List of dictionaries with keys 'name', 'title', 'affiliation' for those who calculated the doses.
    approved_by : list of dict, optional
        List of dictionaries with keys 'name', 'title', 'affiliation' for those who approved the report.
    """
    # Load JSON data
    with open(json_file, "r") as file:
        data = json.load(file)

    # Create PDF document
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph(
        "<para align=center><b>DOSIMETRY REPORT</b></para>", styles["Title"]
    )
    elements.append(title)
    elements.append(Spacer(1, 0.5 * inch))

    # Subject Information Section
    elements.append(Paragraph("<b>Subject Information</b>", styles["Heading2"]))

    # Subject Information Table
    subject_data = [
        ["Clinical Trial", "PR.21"],
        ["ID", data.get("PatientID")],
        ["Sex", data.get("Gender")],
        ["Number of cycles ", data.get("No_of_completed_cycles")],
    ]

    subject_table = Table(subject_data, colWidths=[1.5 * inch, 4 * inch])
    subject_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 12),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ]
        )
    )

    elements.append(subject_table)
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(
        Paragraph("<b>Maximum Intensity Projection</b>", styles["Heading3"])
    )

    calling_folder = Path().absolute()  # notebook folder
    mip_images = []
    max_width = 8 * inch
    max_height = 6 * inch

    for i in range(1, data.get("No_of_completed_cycles") + 1):
        pattern = calling_folder / f"TestDoseDB/MIP_tp*_Cycle_0{i}.png"
        matches = sorted(glob.glob(str(pattern)))  # find all matches for this cycle

        for match in matches:
            img = Image(match)
            scale = min(max_width / img.imageWidth, max_height / img.imageHeight) / 2
            img.drawWidth = img.imageWidth * scale
            img.drawHeight = img.imageHeight * scale
            mip_images.append(img)

    # Put all images in one row using a Table
    if mip_images:
        mip_table = Table([mip_images])  # single row
        mip_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        elements.append(mip_table)

        # Add caption below
        elements.append(Spacer(1, 0.2 * inch))
        caption = Paragraph(
            "<para align=center><i>Figure 1: Maximum Intensity Projection images of the patient across cycles. "
            "The regions show the segmented organs at risk including the kidneys and the salivary glands. </i></para>",
            styles["Normal"],
        )
        elements.append(caption)
        elements.append(Spacer(1, 0.2 * inch))

    for i in range(1, data.get("No_of_completed_cycles") + 1):
        cycle_info(i, elements, styles, data)

    elements.append(Paragraph("<b>Patient Summary</b>", styles["Heading2"]))

    # Paths to your three images
    image_paths = [
        calling_folder / "TestDoseDB/Gy_cummulative.png",
        calling_folder / "TestDoseDB/Gy_per_cycle.png",
        calling_folder / "TestDoseDB/Gy_per_GBq_per_cycle.png",
    ]

    # Load and scale images
    imgs = []
    for path in image_paths:
        img = Image(str(path))
        scale = min(max_width / img.imageWidth, max_height / img.imageHeight) / 2.8
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        imgs.append(img)

    # Create a table with 1 row and 3 columns
    table = Table([imgs], colWidths=[max_width / 3] * 3)

    # Optional styling
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(table)

    # Add caption
    elements.append(Spacer(1, 0.2 * inch))
    caption = Paragraph(
        "<para align=center><i>Figure 2: Cumulative absorbed dose, absorbed dose per cycle, and absorbed dose per GBq per cycle for target organs.</i></para>",
        styles["Normal"],
    )
    elements.append(caption)
    elements.append(Spacer(1, 0.2 * inch))

    # Paths to trend plots
    trend_paths = [
        calling_folder / "TestDoseDB/Hemoglobin_trend.png",
        calling_folder / "TestDoseDB/Platelets_trend.png",
        calling_folder / "TestDoseDB/eGFR_trend.png",
        calling_folder / "TestDoseDB/PSA_trend.png",
    ]

    trend_imgs = []
    for path in trend_paths:
        img = Image(str(path))
        # Scale to fit 2×2 layout
        scale = min(
            (max_width / 2.2) / img.imageWidth, (max_height / 2.2) / img.imageHeight
        )
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        trend_imgs.append(img)

    # Arrange in 2×2 structure
    trend_table_data = [[trend_imgs[0], trend_imgs[1]], [trend_imgs[2], trend_imgs[3]]]

    trend_table = Table(trend_table_data, colWidths=[max_width / 2.2, max_width / 2.2])

    trend_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(trend_table)

    # Add caption
    elements.append(Spacer(1, 0.2 * inch))
    caption = Paragraph(
        "<para align=center><i>Figure 3: Trends of hematological and renal function, and PSA.</i></para>",
        styles["Normal"],
    )
    elements.append(caption)
    elements.append(Spacer(1, 0.2 * inch))

    # ===============================
    # Signatures Section
    # ===============================
    elements.append(Spacer(1, 0.5 * inch))
    team_title = Paragraph("<b>Signature Section</b>", styles["Heading2"])
    elements.append(team_title)

    # --- Calculated by ---
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(
        Paragraph("The absorbed doses were calculated by:", styles["Normal"])
    )

    if calculated_by:
        calc_blocks = [signature_block(p, styles) for p in calculated_by]
        calc_table = Table(
            [calc_blocks], colWidths=[doc.width / len(calc_blocks)] * len(calc_blocks)
        )
        elements.append(calc_table)

    # --- Approved by ---
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(
        Paragraph("The results were reviewed and approved by:", styles["Normal"])
    )

    if approved_by:
        app_blocks = [signature_block(p, styles) for p in approved_by]
        app_table = Table(
            [app_blocks], colWidths=[doc.width / len(app_blocks)] * len(app_blocks)
        )
        elements.append(app_table)

    # Build PDF
    doc.build(elements)


def cycle_info(cycle_n, elements, styles, data):
    """Add cycle information to the PDF report elements.

    Parameters
    ----------
    cycle_n : int
        The cycle number.
    elements : list
        List of reportlab elements to append to.
    styles : dict
        ReportLab styles dictionary.
    data : dict
        Patient data dictionary.
    """
    # Therapy Information Section
    therapy_title = Paragraph(f"<b>Cycle {cycle_n}</b>", styles["Heading2"])
    elements.append(therapy_title)

    # Therapy Information Table
    therapy_info = data.get(f"Cycle_0{cycle_n}", {})
    therapy_data = [
        ["Radiopharmaceutical", "177Lu-PSMA-617"],
        ["Mode of administration", "I.V."],
        ["Administered Activity (MBq)", therapy_info[0].get("InjectedActivity", "")],
        [
            "Date of injection",
            (
                datetime.strptime(
                    therapy_info[0].get("InjectionDate", ""), "%Y%m%d"
                ).strftime("%Y-%m-%d")
                if therapy_info[0].get("InjectionDate", "")
                else ""
            ),
        ],
    ]

    therapy_table = Table(therapy_data, colWidths=[2.5 * inch, 3 * inch])
    therapy_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 12),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ]
        )
    )

    elements.append(therapy_table)

    page_width, page_height = letter
    max_width = page_width - 2 * 72  # 1-inch margins
    max_height = page_height - 2 * 72

    fig_title = Paragraph(
        "<b>Time-activity curves, fit functions, and fit parameters</b>",
        styles["Heading3"],
    )
    elements.append(fig_title)

    calling_folder = Path().absolute()  # notebook folder
    pattern = calling_folder / f"TestDoseDB/*_fit_Cycle_0{cycle_n}.png"

    # glob returns a list of matching files
    image_paths = glob.glob(str(pattern))

    for image_path in image_paths:
        img = Image(image_path)

        # Compute scaling factor to fit inside page
        scale = min(max_width / img.imageWidth, max_height / img.imageHeight)

        # Apply scaling (preserve aspect ratio)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale

        elements.append(Spacer(1, 0.3 * inch))
        elements.append(img)

    fig_title = Paragraph(
        "<b>Absorbed dose results for the organs at risk</b>", styles["Heading3"]
    )
    elements.append(fig_title)

    organ_data_Gy_GBq = [
        ["Organ", "TIA (h)", "AD (Gy/GBq)", "AD(Gy)", "BED (Gy)"],
        [
            "Kidneys",
            round(
                (
                    therapy_info[0]["rois"]["Kidney_Left"]["TIA_h"]
                    + therapy_info[0]["rois"]["Kidney_Right"]["TIA_h"]
                ),
                2,
            ),
            round(therapy_info[0]["Organ-level_AD"]["Kidneys"]["AD[Gy/GBq]"], 2),
            round(therapy_info[0]["Organ-level_AD"]["Kidneys"]["AD[Gy]"], 2),
            round(therapy_info[0]["Organ-level_AD"]["Kidneys"]["BED[Gy]"], 2),
        ],
        [
            "Red Marrow",
            round((therapy_info[0]["rois"]["BoneMarrow"]["TIA_h"]), 2),
            round(therapy_info[0]["Organ-level_AD"]["Red Marrow"]["AD[Gy/GBq]"], 2),
            round(therapy_info[0]["Organ-level_AD"]["Red Marrow"]["AD[Gy]"], 2),
            "-",
        ],
        [
            "Salivary glands",
            round(
                (
                    therapy_info[0]["rois"]["ParotidGland_Left"]["TIA_h"]
                    + therapy_info[0]["rois"]["ParotidGland_Right"]["TIA_h"]
                    + therapy_info[0]["rois"]["SubmandibularGland_Left"]["TIA_h"]
                    + therapy_info[0]["rois"]["SubmandibularGland_Right"]["TIA_h"]
                ),
                2,
            ),
            round(
                therapy_info[0]["Organ-level_AD"]["Salivary Glands"]["AD[Gy/GBq]"], 2
            ),
            round(therapy_info[0]["Organ-level_AD"]["Salivary Glands"]["AD[Gy]"], 2),
            "-",
        ],
    ]
    organ_table_Gy_GBq = Table(organ_data_Gy_GBq, colWidths=[1.5 * inch, 1.2 * inch])
    organ_table_Gy_GBq.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 12),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ]
        )
    )

    elements.append(organ_table_Gy_GBq)
    elements.append(Spacer(1, 0.3 * inch))
