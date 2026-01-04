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
    PageBreak,
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
    elements = []

    # If a signature path is provided, insert the image
    if person.get("signature"):
        sig_img = Image(person["signature"])
        # Scale the image to fit width and maintain aspect ratio
        sig_img.drawHeight = height
        sig_img.drawWidth = width
        elements.append(sig_img)
    else:
        # Empty space if no signature image
        elements.append(Spacer(1, height))

    # Line row
    line = Table([[""]], colWidths=[width])
    line.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, -1), 1, colors.black)]))
    elements.append(line)

    # Text row
    text = Paragraph(
        f"<para align=center><b>{person['name']}</b><br/>{person['title']}<br/>{person['affiliation']}</para>",
        styles["Normal"],
    )
    elements.append(text)

    # Wrap everything in a column table (1 col, stacked)
    block = Table([[e] for e in elements], colWidths=[width])
    block.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return block


def create_dosimetry_pdf(
    image_bar, json_file, output_file, calculated_by=None, approved_by=None
):
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
        "<para align=center><b>DOSIMETRY ASSESSMENT</b></para>", styles["Title"]
    )
    elements.append(title)
    elements.append(Spacer(1, 0.5 * inch))

    # Subject Information Section
    elements.append(Paragraph("<b>Subject Information</b>", styles["Heading2"]))
    # Subject Information Table
    subject_data = [
        ["Clinical Trial", data.get("ClinicalTrial")],
        ["Radiopharmaceutical", "177Lu-PSMA-617"],
        ["Mode of administration", "I.V."],
        ["ID", data.get("PatientID")],
        ["Sex", data.get("Gender")],
        ["Weight kg", data.get("Cycle_01", {})[0].get("Weight_g", "") / 1000],
        ["Height cm", data.get("Cycle_01", {})[0].get("Height_cm", "")],
        ["Number of cycles ", data.get("No_of_completed_cycles")],
    ]

    subject_table = Table(subject_data, colWidths=[2 * inch, 3.5 * inch])
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

        # Add colorbar as the last "image" in the same row
    colorbar_path = calling_folder / "TestDoseDB/bar.png"

    if colorbar_path.exists():
        bar_img = Image(str(colorbar_path))

        # Make the bar much narrower (thin horizontal line)
        bar_max_width = 2.7 * inch
        bar_max_height = 2.4 * inch

        bar_scale = min(
            bar_max_width / bar_img.imageWidth, bar_max_height / bar_img.imageHeight
        )

        bar_img.drawWidth = bar_img.imageWidth * bar_scale
        bar_img.drawHeight = bar_img.imageHeight * bar_scale

    mip_images.append(bar_img)  # <-- add as last image

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
            "The regions show the segmented organs at risk including the kidneys and the salivary glands. "
            f"The maximum value threshold set in all images at {image_bar/1000} kBq/ml. </i></para>",
            styles["Normal"],
        )
        elements.append(caption)
        elements.append(Spacer(1, 0.2 * inch))

    elements.append(PageBreak())

    for i in range(1, data.get("No_of_completed_cycles") + 1):
        cycle_info(i, elements, styles, data)

    elements.append(PageBreak())

    elements.append(Paragraph("<b>Patient Summary</b>", styles["Heading2"]))

    # ===============================
    # CUMULATIVE ORGAN TABLE
    # ===============================
    elements.append(
        Paragraph("<b>Cumulative Absorbed Dose Summary</b>", styles["Heading3"])
    )

    total_tia_kidneys = 0
    total_tia_salivary = 0
    total_tia_marrow = 0

    total_ad_kidneys = 0
    total_ad_salivary = 0
    total_ad_marrow = 0

    total_bed_kidneys = 0

    for i in range(1, data.get("No_of_completed_cycles") + 1):
        therapy_info = data.get(f"Cycle_0{i}", {})[0]

        # Kidneys
        total_tia_kidneys += (
            therapy_info["VOIs"]["Kidney_Left"]["TIA_h"]
            + therapy_info["VOIs"]["Kidney_Right"]["TIA_h"]
        )
        total_ad_kidneys += therapy_info["Organ-level_AD"]["Kidneys"]["AD[Gy]"]
        total_bed_kidneys += therapy_info["Organ-level_AD"]["Kidneys"]["BED[Gy]"]

        # Red marrow
        total_tia_marrow += therapy_info["VOIs"]["BoneMarrow"]["TIA_h"]
        total_ad_marrow += therapy_info["Organ-level_AD"]["Red Marrow"]["AD[Gy]"]

        # Salivary glands
        total_tia_salivary += (
            therapy_info["VOIs"]["ParotidGland_Left"]["TIA_h"]
            + therapy_info["VOIs"]["ParotidGland_Right"]["TIA_h"]
            + therapy_info["VOIs"]["SubmandibularGland_Left"]["TIA_h"]
            + therapy_info["VOIs"]["SubmandibularGland_Right"]["TIA_h"]
        )
        total_ad_salivary += therapy_info["Organ-level_AD"]["Salivary Glands"]["AD[Gy]"]

    # Build the cumulative table
    cumulative_data = [
        ["Organ", "Cumulative TIA (h)", "Cumulative AD (Gy)", "Cumulative BED (Gy)"],
        [
            "Kidneys",
            round(total_tia_kidneys, 2),
            round(total_ad_kidneys, 2),
            round(total_bed_kidneys, 2),
        ],
        ["Red Marrow", round(total_tia_marrow, 2), round(total_ad_marrow, 2), "-"],
        [
            "Salivary glands",
            round(total_tia_salivary, 2),
            round(total_ad_salivary, 2),
            "-",
        ],
    ]

    cumulative_table = Table(cumulative_data, colWidths=[1.5 * inch, 1.7 * inch])
    cumulative_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 12),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightblue),
            ]
        )
    )

    elements.append(cumulative_table)
    # Paths to your three images
    image_paths = [
        calling_folder / "TestDoseDB/Gy_per_cycle.png",
        calling_folder / "TestDoseDB/Gy_per_GBq_per_cycle.png",
        calling_folder / "TestDoseDB/Gy_cumulative.png",
    ]

    # Load and scale images
    imgs = []
    for path in image_paths:
        img = Image(str(path))
        scale = min(max_width / img.imageWidth, max_height / img.imageHeight) / 3
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
        "<para align=center><i>Figure 2: Absorbed dose per cycle reported in units of Gy and Gy/GBq, and cumulative absorbed dose in Gy for target organs and total tumor burden (TTB).</i></para>",
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
            (max_width / 2.5) / img.imageWidth, (max_height / 2.5) / img.imageHeight
        )
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        trend_imgs.append(img)

    # Arrange in 2×2 structure
    trend_table_data = [[trend_imgs[0], trend_imgs[1]], [trend_imgs[2], trend_imgs[3]]]

    trend_table = Table(trend_table_data, colWidths=[max_width / 2.5, max_width / 2.5])

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
    elements.append(trend_table)

    # Add caption
    caption = Paragraph(
        "<para align=center><i>Figure 3: Trends of hematological and renal function, and PSA.</i></para>",
        styles["Normal"],
    )
    elements.append(caption)

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

    # ===============================
    # Appendix
    # ===============================

    elements.append(PageBreak())
    title = Paragraph("<b>Appendix</b>", styles["Heading2"])
    elements.append(title)
    fig_title = Paragraph(
        "<b>Biodistribution and kinetic analysis</b>",
        styles["Heading3"],
    )
    elements.append(fig_title)

    for i in range(1, data.get("No_of_completed_cycles") + 1):
        biodistribution_per_cycle(i, elements, styles, data)

    # Build PDF
    doc.build(elements)


def biodistribution_per_cycle(cycle_n, elements, styles, data):
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
    if cycle_n == 1:
        pass
    else:
        elements.append(PageBreak())

    therapy_title = Paragraph(f"<b>Cycle {cycle_n}</b>", styles["Heading2"])
    elements.append(therapy_title)

    page_width, page_height = letter
    max_width = page_width - 2 * 72  # 1-inch margins
    max_height = page_height - 2 * 72

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

    # Format injection date nicely
    inj_date_raw = therapy_info[0].get("InjectionDate", "")
    inj_date = (
        datetime.strptime(inj_date_raw, "%Y%m%d").strftime("%Y-%m-%d")
        if inj_date_raw
        else ""
    )

    # Build a clean paragraph instead of a table
    therapy_info_para = Paragraph(
        f"<para>"
        f"<b>Administered activity:</b> {therapy_info[0].get('InjectedActivity', '')} MBq<br/>"
        f"<b>Date of injection:</b> {inj_date}"
        f"</para>",
        styles["Normal"],
    )

    # Add to document
    elements.append(therapy_info_para)
    elements.append(Spacer(1, 0.089 * inch))

    fig_title = Paragraph(
        "<b>Absorbed dose results for the organs at risk</b>", styles["Heading3"]
    )
    elements.append(fig_title)

    organ_data_Gy_GBq = [
        ["Organ", "TIA (h)", "Mass (g)", "AD (Gy/GBq)", "AD (Gy)", "BED (Gy)"],
        [
            "Kidneys",
            round(
                (
                    therapy_info[0]["VOIs"]["Kidney_Left"]["TIA_h"]
                    + therapy_info[0]["VOIs"]["Kidney_Right"]["TIA_h"]
                ),
                2,
            ),
            round(
                (
                    therapy_info[0]["VOIs"]["Kidney_Left"]["volumes_mL"]["mean"]
                    + therapy_info[0]["VOIs"]["Kidney_Right"]["volumes_mL"]["mean"]
                ),
                2,
            ),
            round(therapy_info[0]["Organ-level_AD"]["Kidneys"]["AD[Gy/GBq]"], 2),
            round(therapy_info[0]["Organ-level_AD"]["Kidneys"]["AD[Gy]"], 2),
            round(therapy_info[0]["Organ-level_AD"]["Kidneys"]["BED[Gy]"], 2),
        ],
        [
            "Red Marrow",
            round((therapy_info[0]["VOIs"]["BoneMarrow"]["TIA_h"]), 2),
            round(therapy_info[0]["VOIs"]["BoneMarrow"]["volumes_mL"]["mean"], 2),
            round(therapy_info[0]["Organ-level_AD"]["Red Marrow"]["AD[Gy/GBq]"], 2),
            round(therapy_info[0]["Organ-level_AD"]["Red Marrow"]["AD[Gy]"], 2),
            "-",
        ],
        [
            "Salivary glands",
            round(
                (
                    therapy_info[0]["VOIs"]["ParotidGland_Left"]["TIA_h"]
                    + therapy_info[0]["VOIs"]["ParotidGland_Right"]["TIA_h"]
                    + therapy_info[0]["VOIs"]["SubmandibularGland_Left"]["TIA_h"]
                    + therapy_info[0]["VOIs"]["SubmandibularGland_Right"]["TIA_h"]
                ),
                2,
            ),
            round(
                (
                    therapy_info[0]["VOIs"]["ParotidGland_Left"]["volumes_mL"]["mean"]
                    + therapy_info[0]["VOIs"]["ParotidGland_Right"]["volumes_mL"][
                        "mean"
                    ]
                    + therapy_info[0]["VOIs"]["SubmandibularGland_Left"]["volumes_mL"][
                        "mean"
                    ]
                    + therapy_info[0]["VOIs"]["SubmandibularGland_Right"]["volumes_mL"][
                        "mean"
                    ]
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
    organ_table_Gy_GBq = Table(organ_data_Gy_GBq, colWidths=[1.5 * inch, 1.15 * inch])
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
