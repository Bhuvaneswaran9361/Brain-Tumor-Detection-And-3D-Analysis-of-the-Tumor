"""
report_generator.py — Clinical PDF Report  (alignment-fixed v2)
================================================================
All alignment issues fixed:
  1. Probability table   — fixed total width = usable page width
  2. Imaging panel       — 4-up grid with equal cell sizes, full-width table
  3. Measurements table  — two balanced columns spanning page width
  4. Shape metrics table — three balanced columns spanning page width
  5. 3D views            — equal-width 3-column table, centred on page
  6. Grad-CAM heatmaps   — equal-width 3-column table, centred on page
  7. Caption rows        — same colWidths as image rows (critical for alignment)
  8. All tables use LEFT/VALIGN consistently
"""
import io, base64, logging, datetime
from pathlib import Path
import numpy as np
import cv2

logger = logging.getLogger(__name__)

DARK_BG    = (13/255,  17/255,  23/255)
CYAN       = (0/255,  212/255, 255/255)
AMBER      = (255/255, 179/255,   0/255)
GREEN      = (57/255,  211/255,  83/255)
RED        = (255/255,  68/255,  68/255)
LIGHT_GRAY = (0.85, 0.85, 0.85)
MID_GRAY   = (0.55, 0.55, 0.55)

CLASS_COLORS_HEX = {
    "Glioma":     "#FF4444",
    "Meningioma": "#FF8C00",
    "Pituitary":  "#FFD700",
    "No Tumor":   "#44CC44",
}

def _hex_to_rl(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))

def _b64_to_pil(b64):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64)))

def _b64_to_np(b64):
    data = base64.b64decode(b64)
    arr  = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def generate_report(
    predict_result: dict,
    patient_id:  str = "Unknown",
    patient_name: str = "N/A",
    scan_date:   str = None,
    radiologist: str = "AI System",
    output_path: str = "/tmp/brain_tumor_report.pdf",
) -> str:

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units     import mm
        from reportlab.lib           import colors
        from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums     import TA_CENTER, TA_LEFT
        from reportlab.platypus      import (SimpleDocTemplate, Paragraph,
                                              Spacer, Table, TableStyle,
                                              Image as RLImage, HRFlowable)
        from reportlab.graphics.shapes import Drawing, Rect
    except ImportError:
        raise ImportError("pip install reportlab")

    # ── Page geometry ─────────────────────────────────────────────────────────
    PAGE_W, PAGE_H = A4
    L_MAR = R_MAR = 18 * mm
    USABLE_W = PAGE_W - L_MAR - R_MAR   # ≈ 174 mm

    scan_date = scan_date or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    cls     = predict_result.get("classification", {})
    seg     = predict_result.get("segmentation",   {})
    t3d     = predict_result.get("tumor_3d",       {})
    demo    = predict_result.get("demo_mode", False)

    tumor_class = cls.get("class_name", "Unknown")
    confidence  = cls.get("confidence",  0.0)
    has_tumor   = seg.get("tumor_detected", False)
    area_mm2    = t3d.get("volume_mm2",  0.0)
    probs       = cls.get("probabilities", {})
    shape_m     = t3d.get("shape_metrics") or {}
    dims        = t3d.get("dimensions")    or {}

    cls_color_rl = colors.Color(*_hex_to_rl(
        CLASS_COLORS_HEX.get(tumor_class, "#888888")))

    # ── Doc ───────────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=L_MAR, rightMargin=R_MAR,
        topMargin=16*mm,  bottomMargin=16*mm,
        title=f"Brain Tumour Report — {patient_id}",
        author="Brain Tumour Detection System",
    )

    # ── Styles ────────────────────────────────────────────────────────────────
    def S(name, **kw): return ParagraphStyle(name, **kw)

    TITLE   = S("RPT_TITLE",  fontSize=20, fontName="Helvetica-Bold",
                 textColor=colors.Color(*CYAN), alignment=TA_CENTER, spaceAfter=4)
    SUB     = S("RPT_SUB",    fontSize=10, fontName="Helvetica",
                 textColor=colors.Color(*MID_GRAY), alignment=TA_CENTER, spaceAfter=8)
    H1      = S("RPT_H1",     fontSize=13, fontName="Helvetica-Bold",
                 textColor=colors.Color(*CYAN), spaceBefore=8, spaceAfter=4)
    H2      = S("RPT_H2",     fontSize=11, fontName="Helvetica-Bold",
                 textColor=colors.Color(*AMBER), spaceBefore=6, spaceAfter=3)
    BODY    = S("RPT_BODY",   fontSize=9,  fontName="Helvetica",
                 textColor=colors.Color(*LIGHT_GRAY), spaceAfter=4, leading=13)
    SMALL   = S("RPT_SMALL",  fontSize=7,  fontName="Helvetica",
                 textColor=colors.Color(*MID_GRAY))
    RESULT  = S("RPT_RESULT", fontSize=22, fontName="Helvetica-Bold",
                 textColor=cls_color_rl, alignment=TA_CENTER, spaceAfter=2)
    CONF_S  = S("RPT_CONF",   fontSize=13, fontName="Helvetica",
                 textColor=colors.Color(*LIGHT_GRAY), alignment=TA_CENTER, spaceAfter=4)
    STATUS_C = RED if has_tumor else GREEN
    STAT_S  = S("RPT_STAT",   fontSize=11, fontName="Helvetica-Bold",
                 textColor=colors.Color(*STATUS_C), alignment=TA_CENTER, spaceAfter=4)
    CAP     = S("RPT_CAP",    fontSize=8,  fontName="Helvetica-Oblique",
                 textColor=colors.Color(*MID_GRAY), alignment=TA_CENTER,
                 spaceAfter=2, spaceBefore=2)
    WARN    = S("RPT_WARN",   fontSize=8,  fontName="Helvetica-Oblique",
                 textColor=colors.Color(*AMBER), alignment=TA_CENTER)

    def hr(thick=0.5):
        return HRFlowable(width="100%", thickness=thick,
                          color=colors.Color(*CYAN),
                          spaceBefore=4, spaceAfter=4)

    def section_hdr(txt):
        return [hr(), Paragraph(txt, H1)]

    # ── Common table style helper ─────────────────────────────────────────────
    def _tbl_style(extra=None):
        base = [
            ("BACKGROUND",     (0,0), (-1,0),  colors.Color(0.08,0.10,0.14)),
            ("ROWBACKGROUNDS",  (0,1), (-1,-1),
             [colors.Color(0.08,0.10,0.14), colors.Color(0.10,0.12,0.16)]),
            ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTNAME",       (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",       (0,0), (-1,-1), 8),
            ("TEXTCOLOR",      (0,0), (-1,0),  colors.Color(*AMBER)),
            ("TEXTCOLOR",      (0,1), (-1,-1), colors.Color(*LIGHT_GRAY)),
            ("INNERGRID",      (0,0), (-1,-1), 0.3, colors.Color(0.2,0.2,0.3)),
            ("BOX",            (0,0), (-1,-1), 0.5, colors.Color(*CYAN)),
            ("TOPPADDING",     (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
            ("LEFTPADDING",    (0,0), (-1,-1), 7),
            ("RIGHTPADDING",   (0,0), (-1,-1), 7),
            ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
        ]
        if extra:
            base.extend(extra)
        return TableStyle(base)

    # ── Image table helper ────────────────────────────────────────────────────
    def _img_tbl_style():
        return TableStyle([
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("INNERGRID",  (0,0), (-1,-1), 0.3, colors.Color(0.2,0.2,0.3)),
            ("BOX",        (0,0), (-1,-1), 0.5, colors.Color(*CYAN)),
            ("BACKGROUND", (0,0), (-1,-1), colors.Color(0.05,0.07,0.11)),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ])

    def _cap_tbl_style():
        return TableStyle([
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("TOPPADDING",    (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ])

    # ── STORY ─────────────────────────────────────────────────────────────────
    story = []

    # ── 1. Header ─────────────────────────────────────────────────────────────
    story.append(Paragraph("BRAIN TUMOUR DETECTION REPORT", TITLE))
    story.append(Paragraph("AI-Assisted Clinical Decision Support", SUB))
    story.append(hr(1.5))

    # Patient info — 4 equal columns
    CW4 = USABLE_W / 4
    info_rows = [
        ["Patient ID",   patient_id,   "Scan Date",    scan_date],
        ["Patient Name", patient_name, "Radiologist",  radiologist],
        ["Report Mode",  "DEMO" if demo else "LIVE",
         "Model Version","YOLO-NAS + En-DeNet v3"],
    ]
    it = Table(info_rows, colWidths=[CW4]*4)
    it.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",      (2,0), (2,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,0), (0,-1), colors.Color(*AMBER)),
        ("TEXTCOLOR",     (2,0), (2,-1), colors.Color(*AMBER)),
        ("TEXTCOLOR",     (1,0), (1,-1), colors.Color(*LIGHT_GRAY)),
        ("TEXTCOLOR",     (3,0), (3,-1), colors.Color(*LIGHT_GRAY)),
        ("ROWBACKGROUNDS",(0,0), (-1,-1),
         [colors.Color(0.08,0.10,0.14), colors.Color(0.10,0.12,0.16)]),
        ("BOX",           (0,0), (-1,-1), 0.5, colors.Color(*CYAN)),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.Color(0.2,0.2,0.3)),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(it)
    story.append(Spacer(1, 7*mm))

    # ── 2. Primary diagnosis ──────────────────────────────────────────────────
    story += section_hdr("PRIMARY DIAGNOSIS")
    story.append(Spacer(1, 3*mm))

    # Each diagnosis line is its own row in a single-column table so they
    # stack vertically with no overlap. A list-in-one-cell causes ReportLab
    # to render all paragraphs at the same vertical position.
    diag_rows = [
        [Paragraph(f"Confidence: {confidence:.1f}%", CONF_S)],
        [Paragraph(tumor_class.upper(), RESULT)],
        [Paragraph(
            "[ TUMOUR DETECTED ]" if has_tumor else "[ NO TUMOUR DETECTED ]",
            STAT_S)],
    ]
    if demo:
        diag_rows.append(
            [Paragraph("⚠ DEMO MODE — Results are synthetic.", WARN)])
    diag_tbl = Table(diag_rows, colWidths=[USABLE_W])
    diag_tbl.setStyle(TableStyle([
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 0),
        ("RIGHTPADDING",  (0,0), (-1,-1), 0),
        ("BACKGROUND",    (0,0), (-1,-1), colors.Color(0.07,0.09,0.13)),
        ("BOX",           (0,0), (-1,-1), 0.5, colors.Color(*CYAN)),
    ]))
    story.append(diag_tbl)
    story.append(Spacer(1, 5*mm))

    # ── 3. Probability table ───────────────────────────────────────────────────
    # Columns: Class | Probability | Bar
    # Widths must sum exactly to USABLE_W
    CLS_W  = 42*mm
    PROB_W = 28*mm
    BAR_W  = USABLE_W - CLS_W - PROB_W   # column width
    BAR_H  = 8
    # The table cell has 7pt left+right padding each side (from _tbl_style).
    # Drawing must be smaller than (BAR_W - total cell padding) or it overflows.
    CELL_PAD = 14   # 7pt left + 7pt right from _tbl_style
    DRAW_W   = BAR_W - CELL_PAD   # drawing canvas is narrower than the column

    CLS_COLORS_RL = {
        "Glioma":     colors.Color(1.0, 0.27, 0.27),
        "Meningioma": colors.Color(1.0, 0.55, 0.00),
        "Pituitary":  colors.Color(1.0, 0.85, 0.00),
        "No Tumor":   colors.Color(0.27, 0.80, 0.33),
    }

    prob_rows = []
    for name, prob in (probs or {}).items():
        bar_c  = CLS_COLORS_RL.get(name, colors.Color(*CYAN))
        fill_w = max(2, min(DRAW_W, DRAW_W * prob / 100))
        d = Drawing(DRAW_W, BAR_H + 6)
        # Track background — full drawing width
        d.add(Rect(0, 2, DRAW_W, BAR_H,
                   fillColor=colors.Color(0.13,0.15,0.20),
                   strokeColor=colors.Color(0.22,0.24,0.32), strokeWidth=0.4))
        # Filled bar — capped at DRAW_W
        d.add(Rect(0, 2, fill_w, BAR_H,
                   fillColor=bar_c, strokeColor=None))
        lbl_s = ParagraphStyle(f"PL_{name}", fontSize=9,
                                fontName="Helvetica-Bold", textColor=bar_c)
        prob_rows.append([
            Paragraph(name, BODY),
            Paragraph(f"<b>{prob:.2f}%</b>", lbl_s),
            d,
        ])

    story.append(Paragraph("Classification Probabilities", H2))
    story.append(Spacer(1, 2*mm))
    pt = Table(
        [["Class", "Probability", "Distribution"]] + prob_rows,
        colWidths=[CLS_W, PROB_W, BAR_W],
    )
    pt.setStyle(_tbl_style())
    story.append(pt)
    story.append(Spacer(1, 6*mm))

    # ── 4. Imaging panel ──────────────────────────────────────────────────────
    story += section_hdr("IMAGING ANALYSIS")

    # Collect images in fixed order
    img_keys = [
        ("hadf_image_b64",           "HADF Preprocessed"),
        ("overlay_image_b64",        "Detection Overlay"),
        ("gradcam_overlay_b64",      "Grad-CAM (Classifier)"),
        ("seg_gradcam_overlay_b64",  "SegGrad-CAM (Segmenter)"),
    ]
    seg_keys = [
        ("segmentation", "overlay_base64", "Segmentation Overlay"),
        ("segmentation", "mask_base64",    "Segmentation Mask"),
    ]
    panel_imgs, panel_caps = [], []
    for key, cap in img_keys:
        b64 = predict_result.get(key)
        if b64:
            try: panel_imgs.append(_b64_to_pil(b64)); panel_caps.append(cap)
            except Exception: pass
    for sk, ik, cap in seg_keys:
        b64 = (predict_result.get(sk) or {}).get(ik)
        if b64 and len(panel_imgs) < 8:
            try: panel_imgs.append(_b64_to_pil(b64)); panel_caps.append(cap)
            except Exception: pass

    if panel_imgs:
        COLS       = 4
        # Cell width = USABLE_W / COLS, height = same for square cells
        CELL_W     = USABLE_W / COLS
        CELL_H     = CELL_W          # square
        IMG_PX     = 160             # resize all images to this square
        col_ws     = [CELL_W] * COLS

        for start in range(0, len(panel_imgs), COLS):
            chunk = panel_imgs[start:start+COLS]
            caps  = panel_caps[start:start+COLS]
            # Pad to full row
            while len(chunk) < COLS:
                chunk.append(None); caps.append("")

            img_row, cap_row = [], []
            for pil, c in zip(chunk, caps):
                if pil:
                    buf = io.BytesIO()
                    pil.convert("RGB").resize((IMG_PX, IMG_PX)).save(buf, "PNG")
                    buf.seek(0)
                    img_row.append(RLImage(buf, width=CELL_W, height=CELL_H))
                else:
                    img_row.append("")
                cap_row.append(Paragraph(c, CAP))

            it = Table([img_row], colWidths=col_ws)
            it.setStyle(_img_tbl_style())
            ct = Table([cap_row], colWidths=col_ws)
            ct.setStyle(_cap_tbl_style())
            story.append(it)
            story.append(ct)
            story.append(Spacer(1, 3*mm))

    # ── 5. Tumour measurements ────────────────────────────────────────────────
    story += section_hdr("TUMOUR MEASUREMENTS")
    story.append(Spacer(1, 2*mm))

    # Two-column table: Metric | Value  (full width)
    M_COL  = USABLE_W * 0.40
    V_COL  = USABLE_W - M_COL
    meas   = [
        ["Metric",         "Value"],
        ["Tumour Present",  "Yes" if has_tumor else "No"],
        ["Estimated Area",  f"{area_mm2:.2f} sq mm"],
        ["Width",           f"{dims.get('width_mm',0):.2f} mm"],
        ["Height",          f"{dims.get('height_mm',0):.2f} mm"],
        ["Width (px)",      str(dims.get('width_px',0))],
        ["Height (px)",     str(dims.get('height_px',0))],
    ]
    if t3d.get("centroid"):
        cx, cy = t3d["centroid"]
        meas.append(["Centroid", f"({cx}, {cy}) px"])

    mt = Table(meas, colWidths=[M_COL, V_COL])
    mt.setStyle(_tbl_style())
    story.append(mt)
    story.append(Spacer(1, 4*mm))

    # ── 6. Shape metrics ──────────────────────────────────────────────────────
    if shape_m:
        story.append(Paragraph("Shape Metrics", H2))
        # Three columns: Metric | Value | Interpretation  (full width)
        S1 = USABLE_W * 0.30
        S2 = USABLE_W * 0.20
        S3 = USABLE_W - S1 - S2
        interp_map = {
            "sphericity":  (0.80, "Circular (typical meningioma)",
                                  "Irregular (typical glioma)"),
            "circularity": (0.75, "Well-defined border",
                                  "Irregular border"),
            "compactness": (0.60, "Compact mass",
                                  "Diffuse infiltration"),
            "elongation":  (0.70, "Round shape",
                                  "Elongated shape"),
        }
        sd = [["Metric", "Value", "Interpretation"]]
        for k, v in shape_m.items():
            thresh, good, bad = interp_map.get(k, (0.5,"",""))
            sd.append([k.replace("_"," ").title(), f"{v:.4f}",
                        good if float(v) >= thresh else bad])
        st = Table(sd, colWidths=[S1, S2, S3])
        st.setStyle(_tbl_style())
        story.append(st)

    story.append(Spacer(1, 6*mm))

    # ── 7. 3D Reconstruction ──────────────────────────────────────────────────
    story += section_hdr("3D TUMOUR RECONSTRUCTION")

    if has_tumor and (predict_result.get("segmentation") or {}).get("mask_base64"):
        try:
            mask_np = _b64_to_np(
                predict_result["segmentation"]["mask_base64"])
            if mask_np is not None:
                recon  = TumourReconstructor3DForReport()
                views  = recon.generate_report_views(mask_np, tumor_class)
                if views:
                    # Always 3 views — each gets exactly 1/3 of usable width
                    N      = len(views)
                    V_W    = USABLE_W / N
                    V_H    = V_W          # square
                    v_ws   = [V_W] * N
                    v_imgs, v_caps = [], []
                    for pil_v, cap_v in views:
                        buf = io.BytesIO()
                        pil_v.save(buf, "PNG"); buf.seek(0)
                        v_imgs.append(RLImage(buf, width=V_W, height=V_H))
                        v_caps.append(Paragraph(cap_v, CAP))
                    vt = Table([v_imgs], colWidths=v_ws)
                    vt.setStyle(_img_tbl_style())
                    vc = Table([v_caps], colWidths=v_ws)
                    vc.setStyle(_cap_tbl_style())
                    story.append(vt)
                    story.append(vc)
        except Exception as e:
            logger.warning("3D preview failed: %s", e)
            story.append(Paragraph(
                "3D reconstruction available in the interactive web viewer.",
                BODY))
    else:
        story.append(Paragraph(
            "No tumour detected — 3D reconstruction not applicable.", BODY))

    story.append(Spacer(1, 6*mm))

    # ── 8. Model Explainability ───────────────────────────────────────────────
    story += section_hdr("MODEL EXPLAINABILITY (Grad-CAM)")
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Grad-CAM highlights the regions of the MRI scan that most influenced "
        "the model's decision. Warmer colours (red/yellow) = higher attention.",
        BODY))

    cam_keys = [
        ("gradcam_b64",     "Grad-CAM (Classifier)"),
        ("seg_gradcam_b64", "SegGrad-CAM (Segmenter)"),
        ("eigencam_b64",    "EigenCAM (Gradient-free)"),
    ]
    cam_imgs, cam_caps = [], []
    for key, cap in cam_keys:
        b64 = predict_result.get(key)
        if b64:
            try:
                pil = _b64_to_pil(b64).convert("RGB")
                cam_imgs.append(pil)
                cam_caps.append(cap)
            except Exception:
                pass

    if cam_imgs:
        story.append(Spacer(1, 2*mm))
        N     = len(cam_imgs)
        C_W   = USABLE_W / N
        C_H   = C_W * 0.6        # landscape ratio — avoids oversized squares
        c_ws  = [C_W] * N
        ci_row, cc_row = [], []
        for pil_c, cap_c in zip(cam_imgs, cam_caps):
            buf = io.BytesIO()
            pil_c.resize((180,180)).save(buf, "PNG"); buf.seek(0)
            ci_row.append(RLImage(buf, width=C_W, height=C_H))
            cc_row.append(Paragraph(cap_c, CAP))
        ct_img = Table([ci_row], colWidths=c_ws)
        ct_img.setStyle(_img_tbl_style())
        ct_cap = Table([cc_row], colWidths=c_ws)
        ct_cap.setStyle(_cap_tbl_style())
        story.append(ct_img)
        story.append(ct_cap)

    story.append(Spacer(1, 4*mm))

    # ── 9. Clinical disclaimer ────────────────────────────────────────────────
    story += section_hdr("CLINICAL DISCLAIMER")
    story.append(Paragraph(
        "This report is generated by an AI-assisted system for research and "
        "decision-support purposes ONLY. It is NOT a substitute for professional "
        "medical diagnosis. All findings must be reviewed and confirmed by a "
        "qualified radiologist or neuro-oncologist before any clinical decision "
        "is made. The system uses YOLO-NAS for classification and En-DeNet v3 "
        "for segmentation, trained on the BraTS 2020 dataset.",
        BODY))

    story.append(Spacer(1, 3*mm))
    proc_time = predict_result.get("processing_time", 0)
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
        f"|  Processing time: {proc_time:.3f}s  "
        f"|  System: Brain Tumour Detection v3",
        SMALL))

    # ── Page background + footer ──────────────────────────────────────────────
    def _page_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColorRGB(*DARK_BG)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.setStrokeColorRGB(*CYAN)
        canvas.setLineWidth(0.5)
        canvas.line(L_MAR, 12*mm, A4[0]-R_MAR, 12*mm)
        canvas.setFillColorRGB(*MID_GRAY)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(L_MAR, 8*mm,
            f"Brain Tumour Detection Report  |  Patient: {patient_id}"
            f"  |  Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_page_bg, onLaterPages=_page_bg)
    logger.info("Report saved: %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 3D views for report (matplotlib static renders)
# ─────────────────────────────────────────────────────────────────────────────

class TumourReconstructor3DForReport:
    def generate_report_views(self, mask_img, tumor_class):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from PIL import Image
        try:
            from scipy.ndimage  import gaussian_filter
            from skimage.measure import marching_cubes
        except ImportError:
            return []

        gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY) \
               if mask_img.ndim==3 else mask_img.copy()
        gray_s = cv2.resize(gray, (64,64))
        binary = (gray_s > 127).astype(np.float32)
        if binary.sum() < 10: return []

        dist = cv2.distanceTransform(
            (binary*255).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        if dist.max() > 0: dist /= dist.max()

        depth  = 32
        center = depth // 2
        volume = np.zeros((depth, 64, 64), np.float32)
        for z in range(depth):
            t = abs(z-center)/(depth/2)
            volume[z] = dist * np.exp(-3*t**2)
        volume = (volume > 0.3).astype(np.float32)
        if volume.sum() < 5: return []

        smooth = gaussian_filter(volume, sigma=1.0)
        try:
            verts, faces, _, _ = marching_cubes(smooth, level=0.4)
        except Exception:
            return []

        col_map = {
            "Glioma":     ("#CC3333","darkred"),
            "Meningioma": ("#CC6600","darkorange"),
            "Pituitary":  ("#CCAA00","goldenrod"),
        }
        face_c, edge_c = col_map.get(tumor_class, ("#CC3333","darkred"))

        results = []
        for elev, azim, cap in [(30,45,"Frontal View"),
                                 (15,135,"Lateral View"),
                                 (60,90,"Superior View")]:
            fig = plt.figure(figsize=(4,4), facecolor="#0d1117")
            ax  = fig.add_subplot(111, projection="3d", facecolor="#0d1117")
            mesh = Poly3DCollection(verts[faces], alpha=0.82,
                                     facecolor=face_c, edgecolor=edge_c,
                                     linewidth=0.15)
            ax.add_collection3d(mesh)
            ax.set_xlim(verts[:,0].min(), verts[:,0].max())
            ax.set_ylim(verts[:,1].min(), verts[:,1].max())
            ax.set_zlim(verts[:,2].min(), verts[:,2].max())
            for axis_lbl, lbl in [("x","X"),("y","Y"),("z","Z")]:
                getattr(ax, f"set_{axis_lbl}label")(lbl, color="white", fontsize=7)
            ax.tick_params(colors="white", labelsize=6)
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False
                pane.set_edgecolor("#1f2937")
            ax.grid(color="#1f2937", linewidth=0.5)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(tumor_class, color="#00d4ff", fontsize=9, pad=4)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120,
                        bbox_inches="tight", facecolor="#0d1117")
            plt.close(fig)
            buf.seek(0)
            results.append((Image.open(buf).copy(), cap))
        return results


# ── backward-compat import ────────────────────────────────────────────────────
try:
    from .reconstruction_3d import TumourReconstructor3D
except ImportError:
    try:
        from reconstruction_3d import TumourReconstructor3D
    except ImportError:
        TumourReconstructor3D = None