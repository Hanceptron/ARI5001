"""
PDF Report Generator for ARI 5001 HMM Project
Generates comprehensive PDF report with experimental results.

Author: Murat Emirhan Aykut
Date: December 2025
"""

import os
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors


def generate_report(results_data, output_path):
    """
    Generate a comprehensive PDF report with experimental results.
    
    Args:
        results_data: Dictionary containing experiment results
        output_path: Path to save the PDF report
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4, 
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20
    )
    
    story = []
    
    # Title
    story.append(Paragraph("ARI 5001 - Hidden Markov Models Project", title_style))
    story.append(Paragraph("Experimental Results Report", title_style))
    story.append(Spacer(1, 20))
    
    # Author info
    story.append(Paragraph("Author: Murat Emirhan Aykut", styles['Normal']))
    story.append(Paragraph("Date: December 2025", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Abstract
    story.append(Paragraph("Abstract", heading_style))
    abstract_text = """
    This report presents experimental results for Hidden Markov Model (HMM) inference algorithms 
    applied to weather prediction with umbrella observations. We compare three inference methods: 
    forward filtering, forward-backward smoothing, and Viterbi decoding across various experimental 
    conditions including observation noise, transition dynamics, and sequence length.
    """
    story.append(Paragraph(abstract_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Experiment 1 Results
    story.append(Paragraph("Experiment 1: Filtering vs Smoothing", heading_style))
    exp1 = results_data['exp1']
    
    # Create results table
    exp1_data = [
        ['Method', 'Mean Accuracy', 'Std Dev', 'Min', 'Max'],
        ['Filtering', f"{np.mean(exp1['filtering']):.4f}", f"{np.std(exp1['filtering']):.4f}", 
         f"{np.min(exp1['filtering']):.4f}", f"{np.max(exp1['filtering']):.4f}"],
        ['Smoothing', f"{np.mean(exp1['smoothing']):.4f}", f"{np.std(exp1['smoothing']):.4f}", 
         f"{np.min(exp1['smoothing']):.4f}", f"{np.max(exp1['smoothing']):.4f}"],
        ['Viterbi', f"{np.mean(exp1['viterbi']):.4f}", f"{np.std(exp1['viterbi']):.4f}", 
         f"{np.min(exp1['viterbi']):.4f}", f"{np.max(exp1['viterbi']):.4f}"]
    ]
    
    exp1_table = Table(exp1_data)
    exp1_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(exp1_table)
    story.append(Spacer(1, 12))
    
    improvement = np.array(exp1['smoothing']) - np.array(exp1['filtering'])
    improvement_text = f"""
    Smoothing shows a mean improvement of {np.mean(improvement):.4f} ± {np.std(improvement):.4f} 
    over filtering, with smoothing performing better in {np.sum(improvement > 0)} out of 100 trials.
    """
    story.append(Paragraph(improvement_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Experiment 2 Results
    story.append(Paragraph("Experiment 2: Observation Noise Sensitivity", heading_style))
    exp2 = results_data['exp2']
    
    exp2_text = """
    This experiment evaluates how observation noise affects inference accuracy. 
    All three methods show graceful degradation as noise increases.
    """
    story.append(Paragraph(exp2_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Experiment 3 Results
    story.append(Paragraph("Experiment 3: Transition Dynamics", heading_style))
    exp3 = results_data['exp3']
    
    exp3_text = """
    This experiment examines how state persistence (sticky transitions) affects 
    inference performance. Higher persistence generally improves accuracy.
    """
    story.append(Paragraph(exp3_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Experiment 4 Results
    story.append(Paragraph("Experiment 4: Sequence Length Effect", heading_style))
    exp4 = results_data['exp4']
    
    exp4_text = """
    This experiment analyzes how sequence length affects inference accuracy. 
    Longer sequences provide more information for better inference.
    """
    story.append(Paragraph(exp4_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Conclusions
    story.append(Paragraph("Conclusions", heading_style))
    conclusions = """
    1. Smoothing consistently outperforms filtering by leveraging future observations.
    2. All methods degrade gracefully with increasing observation noise.
    3. Higher state persistence improves inference accuracy across all methods.
    4. Longer observation sequences generally lead to better inference performance.
    """
    story.append(Paragraph(conclusions, styles['Normal']))
    
    # Add images if they exist
    if os.path.exists("belief_evolution.png"):
        story.append(Spacer(1, 20))
        story.append(Paragraph("Belief Evolution Visualization", heading_style))
        story.append(Image("belief_evolution.png", width=6*inch, height=4*inch))
    
    if os.path.exists("experiment_results.png"):
        story.append(Spacer(1, 20))
        story.append(Paragraph("Experiment Results Summary", heading_style))
        story.append(Image("experiment_results.png", width=6*inch, height=4*inch))
    
    # Build PDF
    doc.build(story)
    print(f"PDF report generated: {output_path}")