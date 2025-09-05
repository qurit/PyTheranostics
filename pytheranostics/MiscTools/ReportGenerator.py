import json
import glob
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image


def create_dosimetry_pdf(json_file, output_file):
    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Create PDF document
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph("<para align=center><b>DOSIMETRY REPORT</b></para>", 
                     styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 0.5*inch))
    
    # Subject Information Section
    subject_title = Paragraph("<b>Subject Information</b>", styles['Heading2'])
    elements.append(subject_title)
    
    # Subject Information Table
    subject_data = [
        ['Clinical Trial', 'PR.21'],
        ['ID', data.get('PatientID')],
        ['Sex', data.get('Gender')],
        ['Number of cycles ', data.get('No_of_completed_cycles')]
    ]
    
    subject_table = Table(subject_data, colWidths=[1.5*inch, 4*inch])
    subject_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
    ]))
    
    elements.append(subject_table)
    elements.append(Spacer(1, 0.3*inch))

    for i in range(1,data.get("No_of_completed_cycles")+1):
        cycle = data[f"Cycle_0{i}"]
        cycle_info(i, elements, styles)
        
    fig_title = Paragraph(f"<b>Patient Summary</b>", styles['Heading2'])
    elements.append(fig_title)
    
    # Build PDF
    doc.build(elements)

def cycle_info(cycle_n, elements, styles):
    # Therapy Information Section
    therapy_title = Paragraph(f"<b>Cycle {cycle_n}</b>", styles['Heading2'])
    elements.append(therapy_title)
    
    # Therapy Information Table
    therapy_info = data.get(f'Cycle_0{cycle_n}', {})
    therapy_data = [
        ['Radiopharmaceutical', '177Lu-PSMA-617'],
        ['Mode of administration', 'I.V.'],
        ['Administered Activity (MBq)', therapy_info[0].get('InjectedActivity', '')],
        ['Date of injection', datetime.strptime(therapy_info[0].get('InjectionDate', ''), '%Y%m%d').strftime('%Y-%m-%d') if therapy_info[0].get('InjectionDate', '') else ''],
    ]
    
    therapy_table = Table(therapy_data, colWidths=[2.5*inch, 3*inch])
    therapy_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
    ]))
    
    elements.append(therapy_table)

    fig_title = Paragraph(f"<b>Maximum Intensity Projection</b>", styles['Heading3'])
    elements.append(fig_title)

    page_width, page_height = letter
    max_width = page_width - 2*72   # 1-inch margins
    max_height = page_height - 2*72
    
    # Add image to elements list
    image_path = "TestDoseDB/mip.PNG"
    img = Image(image_path)  # Adjust size as needed
    scale = min(max_width / img.imageWidth, max_height / img.imageHeight) / 2

    # Apply scaling (preserve aspect ratio)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    elements.append(Spacer(1, 0.3*inch))
    elements.append(img)

    # Add caption immediately after the image
    caption = Paragraph("<para align=center><i>Figure 1: Maximum Intensity Projection images of the patient at the different SPECT/CT scan time points post injection. The regions show the segmented organs at risk including the kidneys and the salivary glands. </i></para>", 
                       styles['Normal'])
    elements.append(caption)
    elements.append(Spacer(1, 0.2*inch)) 
    
    
    fig_title = Paragraph(f"<b>Time-activity curves, fit functions, and fit parameters</b>", styles['Heading3'])
    elements.append(fig_title)
    
    image_paths = glob.glob("TestDoseDB/*Cycle_01.png")

    for image_path in image_paths:
        img = Image(image_path)

        # Compute scaling factor to fit inside page
        scale = min(max_width / img.imageWidth, max_height / img.imageHeight)

        # Apply scaling (preserve aspect ratio)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale

        elements.append(Spacer(1, 0.3*inch))
        elements.append(img)
    
    fig_title = Paragraph(f"<b>Absorbed dose results for the organs at risk</b>", styles['Heading3'])
    elements.append(fig_title)
    
    organ_data_Gy_GBq = [
        ['Organ', 'TIA (h)', 'AD (Gy/GBq)', 'AD(Gy)', 'BED (Gy)'],
        ['Kidneys', round((therapy_info[0]['rois']['Kidney_Left']['TIA_h'] + therapy_info[0]['rois']['Kidney_Right']['TIA_h']),2), round(therapy_info[0]['Organ-level_AD']['Kidneys']['AD[Gy/GBq]'], 2), round(therapy_info[0]['Organ-level_AD']['Kidneys']['AD[Gy]'], 2), round(therapy_info[0]['Organ-level_AD']['Kidneys']['BED[Gy]'], 2)],
        ['Red Marrow',  round((therapy_info[0]['rois']['BoneMarrow']['TIA_h']),2), round(therapy_info[0]['Organ-level_AD']['Red Marrow']['AD[Gy/GBq]'], 2), round(therapy_info[0]['Organ-level_AD']['Red Marrow']['AD[Gy]'], 2), "-"],
        ['Salivary glands', round((therapy_info[0]['rois']['ParotidGland_Left']['TIA_h'] + therapy_info[0]['rois']['ParotidGland_Right']['TIA_h'] + therapy_info[0]['rois']['SubmandibularGland_Left']['TIA_h'] +therapy_info[0]['rois']['SubmandibularGland_Right']['TIA_h']),2), round(therapy_info[0]['Organ-level_AD']['Salivary Glands']['AD[Gy/GBq]'], 2),    round(therapy_info[0]['Organ-level_AD']['Salivary Glands']['AD[Gy]'], 2), "-"], 
    ]
    organ_table_Gy_GBq = Table(organ_data_Gy_GBq, colWidths=[1.5*inch, 1.2*inch])
    organ_table_Gy_GBq.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
    ]))
    
    elements.append(organ_table_Gy_GBq)
    elements.append(Spacer(1, 0.3*inch))
    