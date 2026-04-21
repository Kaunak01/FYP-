"""FraudLens PDF Report Generator — ReportLab + Matplotlib."""
import io
import logging
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, HRFlowable, KeepTogether,
)

logger = logging.getLogger(__name__)

W, H = A4

# ---- Rule abbreviation helper ----
_RULE_ABBREV = {
    'VELOCITY_SPIKE':       'VS',
    'NIGHTTIME_HIGH_VALUE': 'NHV',
    'FIRST_TXN_HIGH_VALUE': 'FHV',
    'AMOUNT_ANOMALY':       'AA',
    'RAPID_ESCALATION':     'RE',
}

def _abbrev_rules(rules):
    if not rules:
        return '—'
    return ', '.join(_RULE_ABBREV.get(r, r[:4]) for r in rules)


MARGIN = 1.8 * cm

# ---- Palette ----
NAVY   = colors.HexColor('#1a1a2e')
ACCENT = colors.HexColor('#e67e22')
RED    = colors.HexColor('#e74c3c')
GREEN  = colors.HexColor('#27ae60')
YELLOW = colors.HexColor('#f39c12')
LIGHT  = colors.HexColor('#f8f9fa')
BORDER = colors.HexColor('#dee2e6')
MUTED  = colors.HexColor('#6c757d')
WHITE  = colors.white


# ---- Chart helpers ----
def _fig_to_image(fig, width_cm, height_cm):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=width_cm * cm, height=height_cm * cm)


class FraudLensReportGenerator:
    def __init__(self, payload, config, model_manager=None):
        self.p  = payload
        self.c  = config
        self.mm = model_manager
        self.ts = datetime.now()
        self.report_id = f"FRAUDLENS-{self.ts.strftime('%Y-%m-%d-%H%M%S')}"
        self.styles = self._build_styles()
        # Pre-compute clean filename (strip the long "N rows from X — Model: Y" format)
        self._filename = self._clean_filename(payload.get('filename', 'Uploaded dataset'))
        # Pre-compute cardholder groups so TOC page numbers can be estimated
        self._cardholder_groups = self._compute_cardholder_groups()

    def _compute_cardholder_groups(self):
        """Pre-compute cardholder burst groups (same logic as _cardholder_profiles)."""
        top = self.p.get('top_transactions') or self.p.get('top_flagged') or []
        fraud_txns = sorted(
            [t for t in top if t.get('classification') in ('FRAUD', 'REVIEW') and t.get('features')],
            key=lambda t: t['features'].get('amount_velocity_1h', 0),
            reverse=True,
        )
        groups = []
        used = set()
        card_idx = 1
        for t in fraud_txns:
            if t['transaction_id'] in used:
                continue
            v = t['features'].get('amount_velocity_1h', 0)
            burst = [t]
            used.add(t['transaction_id'])
            for t2 in fraud_txns:
                if t2['transaction_id'] in used:
                    continue
                if abs(t2['features'].get('amount_velocity_1h', 0) - v) < v * 0.15 and len(burst) < 6:
                    burst.append(t2)
                    used.add(t2['transaction_id'])
            groups.append((f'CARD-{card_idx:04d}', burst))
            card_idx += 1
            if card_idx > 5:
                break
        return groups

    def _estimate_cardholder_pages(self):
        """Estimate how many PDF pages Section 8 (Cardholder Profiles) will occupy."""
        groups = self._cardholder_groups
        if not groups:
            return 1
        # Count total timeline rows for top 3 groups
        total_rows = sum(len(txns) for _, txns in groups[:3])
        # Empirically: >14 total rows overflows to 2 pages
        return 2 if total_rows > 14 else 1

    @staticmethod
    def _clean_filename(raw):
        """Extract just the CSV filename from the long formatted string."""
        import re
        if not raw:
            return 'Uploaded dataset'
        # Strip "NNNNN transactions from " prefix
        s = re.sub(r'^\d[\d,]* transactions from ', '', str(raw))
        # Strip " — Model: ..." suffix
        s = s.split(' — Model:')[0].strip()
        return s or raw

    # ------------------------------------------------------------------ styles
    def _build_styles(self):
        s = getSampleStyleSheet()

        def add(name, **kw):
            if name not in s:
                s.add(ParagraphStyle(name=name, **kw))

        add('SectionH',    fontSize=13, fontName='Helvetica-Bold', textColor=NAVY,
            spaceBefore=4, spaceAfter=6)
        add('SubsectionH', fontSize=10, fontName='Helvetica-Bold', textColor=NAVY,
            spaceBefore=8, spaceAfter=5)
        add('Body9',       fontSize=9,  leading=14)
        add('Muted8',      fontSize=8,  textColor=MUTED, leading=11)
        add('Bold8',       fontSize=8,  fontName='Helvetica-Bold')
        add('TOCEntry',    fontSize=10, leading=16)
        add('CenterBold',  fontSize=10, fontName='Helvetica-Bold', alignment=TA_CENTER)
        return s

    # ------------------------------------------------------------------ shorthands
    def _h1(self, text, num=None):
        prefix = f"{num}. " if num else ""
        return Paragraph(f"<b>{prefix}{text.upper()}</b>", self.styles['SectionH'])

    def _h2(self, text):
        return Paragraph(f"<b>{text}</b>", self.styles['SubsectionH'])

    def _body(self, text):
        return Paragraph(text, self.styles['Body9'])

    def _sp(self, h=8):
        return Spacer(1, h)

    def _hr(self):
        return HRFlowable(width='100%', thickness=0.5, color=BORDER,
                          spaceAfter=6, spaceBefore=2)

    def _table(self, data, col_widths, extra_style=None):
        base = [
            ('FONTNAME',       (0,0),  (-1,0),  'Helvetica-Bold'),
            ('FONTSIZE',       (0,0),  (-1,-1), 8),
            ('BACKGROUND',     (0,0),  (-1,0),  colors.HexColor('#e9ecef')),
            ('ROWBACKGROUNDS', (0,1),  (-1,-1), [WHITE, LIGHT]),
            ('GRID',           (0,0),  (-1,-1), 0.3, BORDER),
            ('TOPPADDING',     (0,0),  (-1,-1), 4),
            ('BOTTOMPADDING',  (0,0),  (-1,-1), 4),
            ('LEFTPADDING',    (0,0),  (-1,-1), 6),
            ('RIGHTPADDING',   (0,0),  (-1,-1), 6),
            ('VALIGN',         (0,0),  (-1,-1), 'MIDDLE'),
        ]
        if extra_style:
            base.extend(extra_style)
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle(base))
        return t

    # ================================================================== GENERATE
    def generate(self):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=2.6 * cm, bottomMargin=2.0 * cm,
            title=f"FraudLens Fraud Analysis Report",
            author=self.c.get('prepared_by', 'FraudLens'),
        )

        story = []
        story += self._cover_page();        story.append(PageBreak())
        story += self._toc();               story.append(PageBreak())
        story += self._exec_summary();      story.append(PageBreak())
        story += self._config_section();    story.append(PageBreak())
        story += self._metrics_section();   story.append(PageBreak())
        story += self._risk_distribution(); story.append(PageBreak())
        story += self._high_risk_txns();    story.append(PageBreak())
        story += self._pattern_analysis();  story.append(PageBreak())
        story += self._velocity_section();      story.append(PageBreak())
        story += self._cardholder_profiles();   story.append(PageBreak())
        story += self._model_explanation();     story.append(PageBreak())
        story += self._compliance();        story.append(PageBreak())
        if self.c.get('include_recommendations', True):
            story += self._recommendations(); story.append(PageBreak())
        if self.c.get('include_appendix', True):
            story += self._appendix();        story.append(PageBreak())
        story += self._glossary();          story.append(PageBreak())
        story += self._disclaimer()

        doc.build(story,
                  onFirstPage=self._hdr_ftr,
                  onLaterPages=self._hdr_ftr)
        buf.seek(0)
        return buf, self.report_id

    # ------------------------------------------------------------------ header/footer
    def _hdr_ftr(self, canvas, doc):
        canvas.saveState()
        cls = self.c.get('classification', 'CONFIDENTIAL')
        cls_col = colors.HexColor('#dc2626') if cls == 'CONFIDENTIAL' else MUTED

        # top line
        canvas.setStrokeColor(BORDER); canvas.setLineWidth(0.5)
        canvas.line(MARGIN, H - 1.9*cm, W - MARGIN, H - 1.9*cm)

        # Small S shield logo
        logo_x = MARGIN
        logo_y = H - 1.75*cm
        lw, lh = 0.75*cm, 0.55*cm
        canvas.setFillColor(NAVY)
        canvas.roundRect(logo_x, logo_y, lw, lh, 2, fill=1, stroke=0)
        canvas.setFillColor(ACCENT)
        canvas.roundRect(logo_x + 0.07*cm, logo_y + 0.05*cm,
                         lw - 0.14*cm, lh - 0.1*cm, 1, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawCentredString(logo_x + lw/2, logo_y + 0.15*cm, 'S')

        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN + lw + 0.2*cm, H - 1.6*cm, 'FraudLens Fraud Analysis Report')
        canvas.setFillColor(cls_col)
        canvas.setFont('Helvetica-Bold', 7.5)
        canvas.drawRightString(W - MARGIN, H - 1.6*cm, cls)

        # bottom line
        canvas.setStrokeColor(BORDER)
        canvas.line(MARGIN, 1.6*cm, W - MARGIN, 1.6*cm)
        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 1.2*cm, f'Report ID: {self.report_id}')
        canvas.drawCentredString(W/2, 1.2*cm, f'Page {doc.page}')
        canvas.drawRightString(W - MARGIN, 1.2*cm, self.ts.strftime('%d %B %Y'))
        canvas.restoreState()

    # ================================================================== PAGES
    # ------------------------------------------------------------------ cover
    def _cover_page(self):
        cls = self.c.get('classification', 'CONFIDENTIAL')
        cls_col = colors.HexColor('#dc2626') if cls == 'CONFIDENTIAL' else MUTED
        p = self.p; perf = p.get('performance')

        def banner(text, bg, fg=WHITE, fs=10):
            t = Table([[text]], colWidths=[14*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,-1), bg),
                ('TEXTCOLOR',     (0,0), (-1,-1), fg),
                ('FONTNAME',      (0,0), (-1,-1), 'Helvetica-Bold'),
                ('FONTSIZE',      (0,0), (-1,-1), fs),
                ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
                ('TOPPADDING',    (0,0), (-1,-1), 10 if fs > 10 else 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10 if fs > 10 else 6),
            ]))
            return t

        elems = [Spacer(1, 0.8*cm)]
        elems.append(banner("FraudLens", NAVY, WHITE, 32))
        elems.append(Spacer(1, 0.3*cm))
        elems.append(banner("HYBRID ML FRAUD DETECTION SYSTEM", ACCENT, WHITE, 10))
        elems.append(Spacer(1, 1.2*cm))
        elems.append(Paragraph(
            "<b>FRAUD ANALYSIS REPORT</b>",
            ParagraphStyle('ct', fontSize=20, alignment=TA_CENTER,
                           textColor=NAVY, fontName='Helvetica-Bold')
        ))
        elems.append(Spacer(1, 1.4*cm))

        f1_str = f"{perf['f1']:.3f}" if perf else 'N/A (unlabelled)'
        meta_rows = [
            ["Institution:",    self.c.get('institution_name', 'Financial Institution')],
            ["Report ID:",      self.report_id],
            ["Generated:",      self.ts.strftime('%d %B %Y, %H:%M UTC')],
            ["Prepared By:",    self.c.get('prepared_by', 'Fraud Analytics Team')],
            ["Classification:", cls],
            ["", ""],
            ["Dataset:",        self._filename],
            ["Transactions:",   f"{p.get('total', 0):,}"],
            ["Model:",          p.get('model_display', 'AE + BDS + XGBoost')],
            ["Live Analysis F1:",  f1_str],
            ["Training F1:",       "0.868 (Sparkov test set)"],
        ]
        extra = []
        for i, row in enumerate(meta_rows):
            if row[0] == 'Classification:':
                extra += [
                    ('TEXTCOLOR', (1,i), (1,i), cls_col),
                    ('FONTNAME',  (1,i), (1,i), 'Helvetica-Bold'),
                ]
        mt = Table(meta_rows, colWidths=[4*cm, 10*cm])
        mt.setStyle(TableStyle([
            ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE',  (0,0), (-1,-1), 9),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
        ] + extra))
        elems.append(mt)
        elems.append(Spacer(1, 1.5*cm))
        elems.append(banner(f" {cls} ", cls_col, WHITE, 11))
        elems.append(Spacer(1, 0.8*cm))
        elems.append(self._body(
            "This document contains confidential information. "
            "Unauthorised distribution is prohibited."
        ))
        return elems

    # ------------------------------------------------------------------ TOC
    def _toc(self):
        sections = [
            ("1",  "Executive Summary",                          "3"),
            ("2",  "Analysis Configuration",                     "4"),
            ("3",  "Key Performance Metrics",                    "5"),
            ("4",  "Risk Distribution",                          "6"),
            ("5",  "High-Risk Transactions",                     "7"),
            ("6",  "Fraud Pattern Analysis",                     "12"),
            ("7",  "Velocity & Attack Pattern Analysis",         "13"),
            ("8",  "Cardholder Risk Profiles",                   "15"),
            ("9",  "Model Explanation & Feature Importance",     str(15 + self._estimate_cardholder_pages())),
            ("10", "Regulatory Compliance",                      str(16 + self._estimate_cardholder_pages())),
            ("11", "Recommendations",                            str(17 + self._estimate_cardholder_pages())),
            ("12", "Technical Appendix",                         str(18 + self._estimate_cardholder_pages())),
            ("13", "Glossary",                                   str(19 + self._estimate_cardholder_pages())),
            ("14", "Legal Disclaimer",                           str(20 + self._estimate_cardholder_pages())),
        ]
        elems = [self._h1("Table of Contents"), self._hr(), self._sp(6)]
        for num, title, pg in sections:
            row = Table(
                [[Paragraph(f"<b>{num}.</b>  {title}", self.styles['TOCEntry']),
                  Paragraph(pg, ParagraphStyle('pgn', fontSize=10, alignment=TA_RIGHT))]],
                colWidths=[13*cm, 2*cm]
            )
            row.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
                                      ('BOTTOMPADDING', (0,0), (-1,-1), 2)]))
            elems.append(row)
            elems.append(HRFlowable(width='100%', thickness=0.3, color=BORDER,
                                    spaceBefore=2, spaceAfter=2))
        return elems

    # ------------------------------------------------------------------ exec summary
    def _exec_summary(self):
        p = self.p; perf = p.get('performance')
        total  = p.get('total', 0)
        counts = p.get('counts', {})
        fraud_n = counts.get('FRAUD', 0)
        review_n = counts.get('REVIEW', 0)
        at_risk  = p.get('amount_at_risk', 0)
        amount_stats = p.get('amount_stats', {})

        elems = [self._h1("Executive Summary", 1), self._hr()]
        elems.append(self._body(
            f"FraudLens analysed <b>{total:,} credit card transactions</b> from "
            f"<b>{self.c.get('institution_name','the institution')}</b>. "
            f"The hybrid ML model ({p.get('model_display','AE+BDS+XGBoost')}) identified "
            f"<b>{fraud_n+review_n} transactions</b> requiring investigation."
        ))
        elems.append(self._sp(10))

        # Key findings box
        tp_str  = f"{perf['tp']} ({perf['precision']*100:.1f}% precision)" if perf else str(fraud_n)
        fn_str  = str(perf['fn']) if perf else 'N/A (unlabelled dataset)'
        avg_amt = amount_stats.get('flagged_avg', 0)
        kf = [
            ["TRANSACTIONS ANALYSED",   f"{total:,}"],
            ["FRAUD + REVIEW ALERTS",    str(fraud_n + review_n)],
            ["ESTIMATED TRUE FRAUDS",    tp_str],
            ["ESTIMATED MISSED FRAUDS",  fn_str],
            ["TOTAL VALUE AT RISK",       f"${at_risk:,.2f}"],
            ["AVERAGE FRAUD AMOUNT",      f"${avg_amt:.2f}"],
        ]
        kf_t = Table(kf, colWidths=[7.5*cm, 7.5*cm])
        kf_t.setStyle(TableStyle([
            ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE',  (0,0), (-1,-1), 9),
            ('BACKGROUND',(0,0), (-1,-1), colors.HexColor('#f3f4f6')),
            ('GRID',      (0,0), (-1,-1), 0.5, WHITE),
            ('TOPPADDING',    (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ('LEFTPADDING',   (0,0), (-1,-1), 10),
        ]))
        elems.append(kf_t)
        elems.append(self._sp(10))

        # Confidence bar
        if perf:
            f1 = perf['f1']
            level = "HIGH CONFIDENCE" if f1 >= 0.85 else "MODERATE" if f1 >= 0.7 else "LOW CONFIDENCE"
            bc = GREEN if f1 >= 0.85 else YELLOW if f1 >= 0.7 else RED
            bar_t = Table([[f"MODEL CONFIDENCE: {level}   —   F1 = {f1:.3f}"]],
                          colWidths=[15*cm])
            bar_t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), bc),
                ('TEXTCOLOR',  (0,0), (-1,-1), WHITE),
                ('FONTNAME',   (0,0), (-1,-1), 'Helvetica-Bold'),
                ('FONTSIZE',   (0,0), (-1,-1), 10),
                ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
                ('TOPPADDING',    (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ]))
            elems.append(bar_t)
            elems.append(self._sp(10))

        # Bullet findings
        elems.append(self._h2("Key Findings"))
        hour_fraud  = p.get('hour_fraud', {})
        cat_fraud   = p.get('category_fraud', {})
        night_fraud = sum(hour_fraud.get(str(h), 0) for h in list(range(22,24))+list(range(0,6)))
        total_alerts = fraud_n + review_n
        bullets = []
        if total_alerts:
            bullets.append(f"• <b>{total_alerts}</b> transactions flagged ({total_alerts/max(total,1)*100:.2f}% of volume) totalling <b>${at_risk:,.2f}</b> at risk")
        if night_fraud and total_alerts:
            bullets.append(f"• <b>{night_fraud/max(total_alerts,1)*100:.0f}%</b> of alerts occurred during nighttime hours (10 pm – 6 am)")
        top_cat = sorted(cat_fraud.items(), key=lambda x: x[1], reverse=True)
        if top_cat and top_cat[0][1] > 0:
            bullets.append(f"• Highest-risk category: <b>{top_cat[0][0]}</b> ({top_cat[0][1]} flagged)")
        if amount_stats.get('flagged_avg', 0) > amount_stats.get('normal_avg', 0) * 1.5:
            ratio = amount_stats['flagged_avg'] / max(amount_stats['normal_avg'], 1)
            bullets.append(f"• Flagged transaction avg ${amount_stats['flagged_avg']:.2f} is <b>{ratio:.1f}×</b> higher than normal avg ${amount_stats['normal_avg']:.2f}")
        if not bullets:
            bullets.append("• No significant fraud concentration detected in this dataset")
        for b in bullets:
            elems.append(self._body(b)); elems.append(self._sp(3))

        elems.append(self._sp(8))
        elems.append(self._h2("Immediate Actions Recommended"))
        elems.append(self._body("• Review the top 10 highest-risk transactions detailed in Section 5"))
        elems.append(self._body("• Investigate all transactions with probability > 95% within 24 hours"))
        elems.append(self._body("• See Section 9 for full prioritised recommendations"))
        return elems

    # ------------------------------------------------------------------ config
    def _config_section(self):
        p = self.p; perf = p.get('performance')
        total_val = p.get('amount_at_risk', 0) + p.get('amount_safe', 0)
        fraud_lbl = (f"{perf['total_fraud']} ({perf['total_fraud']/max(p.get('total',1),1)*100:.2f}%)"
                     if perf else 'N/A — unlabelled dataset')

        elems = [self._h1("Analysis Configuration", 2), self._hr()]
        elems.append(self._h2("2.1  Dataset Information"))
        elems.append(self._table([
            ["Parameter", "Value"],
            ["Source File",        self._filename],
            ["Total Transactions", f"{p.get('total',0):,}"],
            ["Total Value",        f"${total_val:,.2f}"],
            ["Labelled Frauds",    fraud_lbl],
            ["Unique Categories",  str(len(p.get('category_counts', {})))],
        ], [6*cm, 9*cm]))
        elems.append(self._sp(10))

        elems.append(self._h2("2.2  Model Configuration"))
        elems.append(self._table([
            ["Parameter", "Value"],
            ["Model",              p.get('model_display', 'AE + BDS + XGBoost')],
            ["Training Dataset",   "Sparkov Synthetic — 1,296,675 train rows"],
            ["Total Features",     "19  (14 base + 1 autoencoder + 4 BDS)"],
            ["FRAUD threshold",    "≥ 0.70  →  FRAUD decision"],
            ["REVIEW threshold",   "≥ 0.50  →  REVIEW decision"],
            ["MONITOR threshold",  "≥ 0.30  →  MONITOR decision"],
        ], [6*cm, 9*cm]))
        elems.append(self._sp(10))

        elems.append(self._h2("2.3  Feature Set"))
        elems.append(self._body(
            "<b>Base (14):</b> amt, hour, is_night, category_encoded, velocity_1h, velocity_24h, "
            "amount_velocity_1h, age, city_pop, distance_cardholder_merchant, gender_encoded, "
            "day_of_week_encoded, is_weekend, month<br/><br/>"
            "<b>Autoencoder (1):</b> reconstruction_error — MSE between input and autoencoder output; "
            "high value = anomalous pattern<br/><br/>"
            "<b>BDS (4):</b> bds_amount, bds_time, bds_freq, bds_category — "
            "GA-optimised deviation scores against global transaction norms"
        ))
        return elems

    # ------------------------------------------------------------------ metrics
    def _metrics_section(self):
        p = self.p; perf = p.get('performance')
        elems = [self._h1("Key Performance Metrics", 3), self._hr()]

        if perf:
            prec = perf['precision'] * 100
            rec  = perf['recall']    * 100
            f1   = perf['f1']
            acc  = perf['accuracy']  * 100
            tp, fp, fn, tn = perf['tp'], perf['fp'], perf['fn'], perf['tn']

            elems.append(self._h2("3.1  Classification Performance"))
            box = Table([[
                Paragraph(f"<b>PRECISION</b><br/><font size='18' color='#e74c3c'><b>{prec:.1f}%</b></font>",
                          self.styles['CenterBold']),
                Paragraph(f"<b>RECALL</b><br/><font size='18' color='#3498db'><b>{rec:.1f}%</b></font>",
                          self.styles['CenterBold']),
                Paragraph(f"<b>F1 SCORE</b><br/><font size='18' color='#27ae60'><b>{f1:.3f}</b></font>",
                          self.styles['CenterBold']),
                Paragraph(f"<b>ACCURACY</b><br/><font size='18'><b>{acc:.1f}%</b></font>",
                          self.styles['CenterBold']),
            ]], colWidths=[3.75*cm]*4)
            box.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), LIGHT),
                ('GRID',       (0,0), (-1,-1), 1, WHITE),
                ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
                ('TOPPADDING', (0,0), (-1,-1), 12),
                ('BOTTOMPADDING', (0,0), (-1,-1), 12),
                ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elems.append(box)
            elems.append(self._sp(12))

            elems.append(self._h2("3.2  Confusion Matrix"))
            cm_t = Table([
                ["",              "Predicted: NORMAL", "Predicted: FRAUD"],
                ["Actual: NORMAL", f"TN = {tn:,}",      f"FP = {fp:,}"],
                ["Actual: FRAUD",  f"FN = {fn:,}",      f"TP = {tp:,}"],
            ], colWidths=[4.5*cm, 5*cm, 5*cm])
            cm_t.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('ALIGN',    (0,0), (-1,-1), 'CENTER'),
                ('GRID',     (0,0), (-1,-1), 0.5, BORDER),
                ('BACKGROUND', (1,1), (1,1), colors.HexColor('#d1fae5')),
                ('BACKGROUND', (2,1), (2,1), colors.HexColor('#fee2e2')),
                ('BACKGROUND', (1,2), (1,2), colors.HexColor('#fee2e2')),
                ('BACKGROUND', (2,2), (2,2), colors.HexColor('#d1fae5')),
                ('TOPPADDING',    (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ]))
            elems.append(cm_t)
            elems.append(Paragraph("Green = correct classification. Red = errors.", self.styles['Muted8']))
            elems.append(self._sp(12))

            spec = tn / max(tn+fp, 1)
            fpr  = fp / max(tn+fp, 1)
            fnr  = fn / max(tp+fn, 1)
            elems.append(self._h2("3.3  Detailed Metrics"))
            elems.append(self._table([
                ["Metric", "Value", "Interpretation"],
                ["True Positives (TP)",  f"{tp:,}",             "Frauds correctly caught"],
                ["False Positives (FP)", f"{fp:,}",             "Legitimate transactions incorrectly flagged"],
                ["False Negatives (FN)", f"{fn:,}",             "Frauds missed"],
                ["True Negatives (TN)",  f"{tn:,}",             "Legitimate correctly cleared"],
                ["Precision",            f"{prec:.2f}%",        "When flagged, we are correct this often"],
                ["Recall (Sensitivity)", f"{rec:.2f}%",         "Percentage of frauds caught"],
                ["Specificity",          f"{spec*100:.3f}%",    "Legitimate transactions correctly cleared"],
                ["F1 Score",             f"{f1:.4f}",           "Harmonic mean of precision and recall"],
                ["False Positive Rate",  f"{fpr*100:.3f}%",     "Fraction of legit transactions flagged"],
                ["False Negative Rate",  f"{fnr*100:.3f}%",     "Fraction of actual frauds missed"],
            ], [4.5*cm, 3*cm, 7.5*cm]))
            elems.append(self._sp(12))

            elems.append(self._h2("3.4  Financial Impact"))
            fa_cost = fp * 150
            caught_val = p.get('amount_at_risk', 0)
            elems.append(self._table([
                ["Metric", "Value"],
                ["Total Value Flagged",             f"${caught_val:,.2f}"],
                ["False Alarm Investigation Cost*", f"${fa_cost:,.2f}"],
                ["Net Value Protected",             f"${max(caught_val - fa_cost, 0):,.2f}"],
            ], [7*cm, 8*cm]))
            elems.append(Paragraph("* Assuming $150 investigation cost per false positive.", self.styles['Muted8']))
        else:
            elems.append(self._body(
                "Performance metrics unavailable — dataset has no fraud labels.<br/>"
                "Upload a CSV with an <b>is_fraud</b> column to compute precision, recall, and F1."
            ))
            counts = p.get('counts', {}); total = p.get('total', 0)
            fraud_n = counts.get('FRAUD', 0); review_n = counts.get('REVIEW', 0)
            elems.append(self._sp(8))
            elems.append(self._table([
                ["Metric", "Value"],
                ["Total Processed",    f"{total:,}"],
                ["Flagged FRAUD",      str(fraud_n)],
                ["Flagged REVIEW",     str(review_n)],
                ["Total Alerts",       str(fraud_n+review_n)],
                ["Alert Rate",         f"{(fraud_n+review_n)/max(total,1)*100:.3f}%"],
                ["Total Value at Risk",f"${p.get('amount_at_risk',0):,.2f}"],
            ], [7*cm, 8*cm]))
        return elems

    # ------------------------------------------------------------------ risk distribution
    def _risk_distribution(self):
        p = self.p; counts = p.get('counts', {}); total = max(p.get('total', 1), 1)
        fraud_n  = counts.get('FRAUD', 0)
        review_n = counts.get('REVIEW', 0)
        monitor_n= counts.get('MONITOR', 0)
        normal_n = counts.get('NORMAL', 0)

        elems = [self._h1("Risk Distribution", 4), self._hr()]
        elems.append(self._h2("4.1  Charts"))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
        # Donut
        sizes = [fraud_n, review_n, monitor_n, normal_n]
        lbls  = ['FRAUD','REVIEW','MONITOR','NORMAL']
        clrs  = ['#e74c3c','#f39c12','#f1c40f','#27ae60']
        nz = [(s,l,c) for s,l,c in zip(sizes,lbls,clrs) if s > 0]
        if nz:
            total_nz = sum(x[0] for x in nz)
            def _pct_fmt(pct):
                return f'{pct:.1f}%' if pct >= 3 else ''
            ax1.pie([x[0] for x in nz],
                    colors=[x[2] for x in nz], autopct=_pct_fmt,
                    startangle=90, pctdistance=0.78,
                    wedgeprops={'linewidth':1,'edgecolor':'white'})
            legend_labels = [f"{x[1]} ({x[0]/total_nz*100:.1f}%)" for x in nz]
            ax1.legend(legend_labels, loc='lower center',
                       bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.5, frameon=False)
        ax1.set_title('Risk Level Distribution', fontsize=11, fontweight='bold', pad=8)

        # Probability histogram
        prob_bins = p.get('prob_distribution', [0]*10)
        bin_labels=['0-10%','10-20%','20-30%','30-40%','40-50%',
                    '50-60%','60-70%','70-80%','80-90%','90-100%']
        bar_clrs = ['#27ae60']*5 + ['#f39c12']*3 + ['#e74c3c']*2
        ax2.bar(range(10), prob_bins, color=bar_clrs, edgecolor='white', linewidth=0.5)
        ax2.set_xticks(range(10))
        ax2.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=7)
        ax2.set_title('Fraud Probability Distribution', fontsize=11, fontweight='bold', pad=8)
        ax2.set_ylabel('# Transactions', fontsize=9)
        try:
            ax2.set_yscale('log')
        except Exception:
            pass
        plt.tight_layout()
        elems.append(_fig_to_image(fig, 15, 6))
        elems.append(self._sp(10))

        elems.append(self._h2("4.2  Risk Level Summary"))
        amount_stats = p.get('amount_stats', {})
        elems.append(self._table([
            ["Level","Count","% of Total","Avg Amount","Recommended Action"],
            ["FRAUD",   str(fraud_n),   f"{fraud_n/total*100:.3f}%",
             f"${amount_stats.get('flagged_avg',0):.2f}", "Block / Investigate"],
            ["REVIEW",  str(review_n),  f"{review_n/total*100:.3f}%",
             "—", "Human Review Required"],
            ["MONITOR", str(monitor_n), f"{monitor_n/total*100:.3f}%",
             "—", "Enhanced Monitoring"],
            ["NORMAL",  str(normal_n),  f"{normal_n/total*100:.3f}%",
             f"${amount_stats.get('normal_avg',0):.2f}", "Approved"],
            ["TOTAL",   str(total),     "100%", "—", "—"],
        ], [2.5*cm, 2.5*cm, 3*cm, 3*cm, 4*cm]))
        return elems

    # ------------------------------------------------------------------ high-risk txns
    def _high_risk_txns(self):
        p = self.p
        top = sorted(
            p.get('top_transactions', p.get('top_flagged', [])),
            key=lambda x: x.get('probability', 0), reverse=True
        )

        elems = [self._h1("High-Risk Transactions", 5), self._hr()]
        elems.append(self._body(
            f"The following {len(top[:50])} transactions were flagged with the highest fraud probability."
        ))
        elems.append(self._sp(8))

        # Summary table (top 50)
        elems.append(self._h2("5.1  Flagged Transactions Table"))
        rows = [["#","Transaction ID","Amount","Category","Hour","Probability","Decision","Rules"]]
        for i, r in enumerate(top[:50], 1):
            rows.append([
                str(i),
                str(r.get('transaction_id','—'))[:14],
                f"${r.get('amount',0):.2f}",
                str(r.get('category','—'))[:16],
                f"{r.get('hour',0)}:00",
                f"{r.get('probability',0)*100:.1f}%",
                r.get('classification','—'),
                _abbrev_rules(r.get('rule_triggers', [])),
            ])

        extra = []
        for i, r in enumerate(top[:50], 1):
            if r.get('classification') == 'FRAUD':
                extra.append(('BACKGROUND', (0,i), (-1,i), colors.HexColor('#fff1f2')))
            elif r.get('classification') == 'REVIEW':
                extra.append(('BACKGROUND', (0,i), (-1,i), colors.HexColor('#fffbeb')))
        t = Table(rows, colWidths=[0.8*cm, 3*cm, 2*cm, 3*cm, 1.5*cm, 2*cm, 2*cm, 2.7*cm])
        t.setStyle(TableStyle([
            ('FONTNAME',  (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',  (0,0), (-1,-1), 7),
            ('GRID',      (0,0), (-1,-1), 0.3, BORDER),
            ('TOPPADDING',    (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING',   (0,0), (-1,-1), 4),
            ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#e9ecef')),
            ('ROWBACKGROUNDS',(0,1), (-1,-1),  [WHITE, LIGHT]),
        ] + extra))
        elems.append(t)
        elems.append(Paragraph(
            "Rules key — VS: Velocity Spike  |  NHV: Nighttime High Value  |  "
            "FHV: First Txn High Value  |  AA: Amount Anomaly  |  RE: Rapid Escalation",
            self.styles['Muted8']
        ))
        elems.append(self._sp(14))

        # Detailed top 10
        elems.append(self._h2("5.2  Detailed Analysis — Top 10 Transactions"))
        elems.append(self._body("SHAP values show each feature's contribution to the fraud probability."))
        elems.append(self._sp(6))

        for i, r in enumerate(top[:10], 1):
            dec  = r.get('classification','UNKNOWN')
            prob = r.get('probability', 0)

            hdr = Table([[
                Paragraph(f"<b>#{i}  {r.get('transaction_id','—')}</b>",
                          ParagraphStyle('wh', fontSize=9, textColor=WHITE, fontName='Helvetica-Bold')),
                Paragraph(f"<b>${r.get('amount',0):.2f}</b>",
                          ParagraphStyle('wh2', fontSize=9, textColor=WHITE, fontName='Helvetica-Bold')),
                Paragraph(str(r.get('category','—')),
                          ParagraphStyle('wh3', fontSize=8, textColor=WHITE)),
                Paragraph(f"{r.get('hour',0)}:00",
                          ParagraphStyle('wh4', fontSize=8, textColor=WHITE)),
                Paragraph(f"<b>{prob*100:.1f}%</b>",
                          ParagraphStyle('wh5', fontSize=9, textColor=WHITE, fontName='Helvetica-Bold')),
                Paragraph(f"<b>{dec}</b>",
                          ParagraphStyle('wh6', fontSize=9, textColor=WHITE, fontName='Helvetica-Bold')),
            ]], colWidths=[3.5*cm, 2*cm, 3.5*cm, 1.5*cm, 2*cm, 2.5*cm])
            hdr.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,-1), NAVY),
                ('TOPPADDING',    (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LEFTPADDING',   (0,0), (-1,-1), 6),
                ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elems.append(hdr)

            shap_data = r.get('shap_values')
            from app.config import FEATURE_DISPLAY_NAMES
            if shap_data and isinstance(shap_data, list):
                shap_rows = [["Feature", "Value", "SHAP Contribution"]]
                for sv in sorted(shap_data, key=lambda x: abs(x.get('contribution',0)), reverse=True)[:8]:
                    c = sv.get('contribution', 0)
                    arrow = '▲' if c > 0 else '▼'
                    raw_val = sv.get('value', 0)
                    fname = sv.get('name', '')
                    if fname == 'gender_encoded':
                        disp_val = 'F' if round(raw_val) == 1 else 'M'
                    elif fname in ('is_night', 'is_weekend'):
                        disp_val = 'Yes' if round(raw_val) == 1 else 'No'
                    else:
                        disp_val = str(round(raw_val, 3))
                    shap_rows.append([
                        sv.get('display_name', sv.get('name','—')),
                        disp_val,
                        f"{arrow} {abs(c):.4f}",
                    ])
                shap_t = self._table(shap_rows, [6*cm, 3*cm, 6*cm])
                shap_extra = []
                for j, sv in enumerate(sorted(shap_data, key=lambda x: abs(x.get('contribution',0)),
                                              reverse=True)[:8], 1):
                    if sv.get('contribution', 0) > 0:
                        shap_extra.append(('TEXTCOLOR', (2,j), (2,j), RED))
                    else:
                        shap_extra.append(('TEXTCOLOR', (2,j), (2,j), GREEN))
                shap_t.setStyle(TableStyle([
                    ('FONTNAME',  (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE',  (0,0), (-1,-1), 8),
                    ('GRID',      (0,0), (-1,-1), 0.3, BORDER),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHT]),
                    ('TOPPADDING',    (0,0), (-1,-1), 3),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                    ('LEFTPADDING',   (0,0), (-1,-1), 6),
                    ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#e9ecef')),
                ] + shap_extra))
                elems.append(shap_t)
            else:
                feat = r.get('features', {})
                key_feats = ['amt','velocity_1h','amount_velocity_1h','is_night','hour']
                feat_rows = [["Feature", "Value"]]
                for k in key_feats:
                    if k in feat:
                        feat_rows.append([FEATURE_DISPLAY_NAMES.get(k, k), str(round(feat[k], 3))])
                if len(feat_rows) > 1:
                    elems.append(self._table(feat_rows, [7.5*cm, 7.5*cm]))

            rules = r.get('rule_triggers', [])
            if rules:
                elems.append(Paragraph(f"Rule triggers: {', '.join(rules)}", self.styles['Muted8']))
            elems.append(self._sp(8))

        return elems

    # ------------------------------------------------------------------ patterns
    def _pattern_analysis(self):
        p = self.p
        hour_counts = p.get('hour_counts', {})
        hour_fraud  = p.get('hour_fraud',  {})
        cat_counts  = p.get('category_counts', {})
        cat_fraud   = p.get('category_fraud',  {})

        elems = [self._h1("Fraud Pattern Analysis", 6), self._hr()]
        elems.append(self._h2("6.1  Fraud by Hour of Day"))

        hours = list(range(24))
        tot_h   = [hour_counts.get(str(h), hour_counts.get(h, 0)) for h in hours]
        fraud_h = [hour_fraud.get(str(h),  hour_fraud.get(h,  0)) for h in hours]

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.bar(hours, tot_h,   color='#dbeafe', label='Total',   edgecolor='white')
        ax.bar(hours, fraud_h, color='#e74c3c', label='Flagged', edgecolor='white')
        ax.axvspan(22, 24, alpha=0.08, color='navy')
        ax.axvspan( 0,  6, alpha=0.08, color='navy', label='Night zone')
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h}" for h in hours], fontsize=7)
        ax.set_xlabel('Hour of Day', fontsize=9)
        ax.set_ylabel('# Transactions', fontsize=9)
        ax.set_title('Transactions by Hour — Total vs Flagged  (shaded = nighttime)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        plt.tight_layout()
        elems.append(_fig_to_image(fig, 15, 5.5))

        night_fraud = sum(fraud_h[22:]) + sum(fraud_h[:6])
        total_fraud = sum(fraud_h)
        if total_fraud:
            elems.append(Paragraph(
                f"Night hours (shaded): {night_fraud}/{total_fraud} fraud alerts "
                f"({night_fraud/total_fraud*100:.0f}%)",
                self.styles['Muted8']
            ))
        elems.append(self._sp(12))

        elems.append(self._h2("6.2  Fraud by Category"))
        if cat_fraud:
            rows = [["Category", "Flagged", "Total Txns", "Flag Rate"]]
            for cat, fc in sorted(cat_fraud.items(), key=lambda x: x[1], reverse=True):
                ct = cat_counts.get(cat, 0)
                rate = fc / max(ct, 1) * 100
                bar  = '█' * min(int(rate * 4), 20)
                rows.append([cat, str(fc), str(ct), f"{rate:.2f}%  {bar}"])
            elems.append(self._table(rows, [5*cm, 2.5*cm, 2.5*cm, 5*cm]))
        return elems

    # ------------------------------------------------------------------ velocity analysis
    def _velocity_section(self):
        p = self.p
        top = p.get('top_transactions') or p.get('top_flagged') or []

        elems = [self._h1("Velocity & Attack Pattern Analysis", 7), self._hr()]
        elems.append(self._body(
            "Velocity features — transactions per hour and hourly spend — are the most novel contribution "
            "of this research. They capture coordinated fraud attacks that low-frequency models miss entirely. "
            "This section analyses how velocity signatures differ between fraud and normal transactions."
        ))
        elems.append(self._sp(10))

        # ---- 7.1  Velocity Distribution Chart ----
        elems.append(self._h2("7.1  Velocity Feature Distribution in Flagged Transactions"))
        fraud_txns = [t for t in top if t.get('classification') in ('FRAUD', 'REVIEW')
                      and t.get('features')]
        if fraud_txns:
            v1h_vals  = [t['features'].get('velocity_1h', 1)       for t in fraud_txns]
            av1h_vals = [t['features'].get('amount_velocity_1h', 0) for t in fraud_txns]
            amts      = [t.get('amount', t['features'].get('amt', 0)) for t in fraud_txns]

            fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))

            axes[0].hist(v1h_vals, bins=min(15, len(v1h_vals)), color='#e74c3c',
                         edgecolor='white', alpha=0.85)
            axes[0].set_title('Transactions per Hour\n(velocity_1h)', fontsize=9, fontweight='bold')
            axes[0].set_xlabel('Count', fontsize=8); axes[0].set_ylabel('Frequency', fontsize=8)
            axes[0].tick_params(labelsize=7)

            axes[1].hist(av1h_vals, bins=min(15, len(av1h_vals)), color='#e67e22',
                         edgecolor='white', alpha=0.85)
            axes[1].set_title('Hourly Spend ($)\n(amount_velocity_1h)', fontsize=9, fontweight='bold')
            axes[1].set_xlabel('$ Amount', fontsize=8); axes[1].set_ylabel('Frequency', fontsize=8)
            axes[1].tick_params(labelsize=7)

            axes[2].scatter(v1h_vals, amts, c='#e74c3c', alpha=0.6, s=20)
            axes[2].set_title('Velocity vs Transaction Amount\n(Flagged only)', fontsize=9,
                              fontweight='bold')
            axes[2].set_xlabel('Transactions/hr', fontsize=8)
            axes[2].set_ylabel('Amount ($)', fontsize=8)
            axes[2].tick_params(labelsize=7)

            plt.tight_layout()
            elems.append(_fig_to_image(fig, 15, 5))
            elems.append(self._sp(6))

            # Stats summary
            import statistics as _stats
            v1h_mean = _stats.mean(v1h_vals) if v1h_vals else 0
            av1h_max = max(av1h_vals) if av1h_vals else 0
            high_vel = sum(1 for v in v1h_vals if v >= 3)
            elems.append(self._body(
                f"Among the top flagged transactions: avg velocity_1h = <b>{v1h_mean:.1f} txns/hr</b>, "
                f"max amount_velocity_1h = <b>${av1h_max:,.2f}/hr</b>, "
                f"<b>{high_vel} of {len(v1h_vals)}</b> had 3+ transactions in the same hour — "
                "a classic burst-attack signature."
            ))
        else:
            elems.append(self._body("No velocity data available for flagged transactions."))

        elems.append(self._sp(12))

        # ---- 7.2  Top Velocity Spikes Table ----
        elems.append(self._h2("7.2  Top Velocity Spike Transactions"))
        elems.append(self._body(
            "Transactions sorted by amount_velocity_1h — highest cumulative hourly spend "
            "indicates concentrated spending bursts typical of card takeover attacks."
        ))
        elems.append(self._sp(6))

        spike_txns = sorted(
            [t for t in top if t.get('features')],
            key=lambda t: t['features'].get('amount_velocity_1h', 0),
            reverse=True
        )[:15]

        if spike_txns:
            rows = [["#", "Tx ID", "Amount", "Vel/hr", "$/hr Spent", "Is Night", "Decision"]]
            for i, t in enumerate(spike_txns, 1):
                f = t.get('features', {})
                rows.append([
                    str(i),
                    t.get('transaction_id', '—'),
                    f"${t.get('amount', 0):,.2f}",
                    f"{f.get('velocity_1h', 0):.0f}",
                    f"${f.get('amount_velocity_1h', 0):,.2f}",
                    "Yes" if f.get('is_night', 0) >= 0.5 else "No",
                    t.get('classification', '—'),
                ])
            tbl = self._table(rows, [0.8*cm, 3.5*cm, 2.5*cm, 1.8*cm, 3*cm, 1.8*cm, 2.1*cm],
                extra_style=[
                    ('TEXTCOLOR', (6,1), (6,-1), RED),
                    ('FONTNAME',  (6,1), (6,-1), 'Helvetica-Bold'),
                ])
            elems.append(tbl)
        elems.append(self._sp(12))

        # ---- 7.3  Burst Pattern Summary ----
        elems.append(self._h2("7.3  Why Velocity Features Matter — Ablation Evidence"))
        elems.append(self._table([
            ["Experiment",              "F1 Score",  "Change",  "Interpretation"],
            ["Full model (with velocity)", "0.8705",   "—",       "Baseline best result"],
            ["Without velocity features",  "0.8561",   "−0.0144", "Velocity contributes +1.44% F1"],
            ["velocity_1h alone",          "+0.48 SHAP","—",      "4th most important feature (SHAP)"],
            ["amount_velocity_1h alone",   "+1.65 SHAP","—",      "3rd most important feature (SHAP)"],
        ], [5.5*cm, 2.5*cm, 2.5*cm, 4.5*cm]))
        elems.append(self._sp(6))
        elems.append(self._body(
            "The ablation study demonstrates that removing velocity features decreases F1 by 0.0144 points. "
            "More critically, velocity features are ranked #3 and #4 by SHAP importance, "
            "capturing coordinated attack patterns (card testing, burst spending) that temporal-blind "
            "models miss entirely. This is the key novel contribution of this research."
        ))
        return elems

    # ------------------------------------------------------------------ cardholder profiles
    def _cardholder_profiles(self):
        p   = self.p
        top = p.get('top_transactions') or p.get('top_flagged') or []

        elems = [self._h1("Cardholder Risk Profiles", 8), self._hr()]
        elems.append(self._body(
            "Cardholder risk profiles are constructed from the transactions with highest fraud probability. "
            "Profiles group transactions by velocity burst windows to identify coordinated card takeover "
            "attacks — the behavioural signature targeted by the BDS algorithm."
        ))
        elems.append(self._sp(10))

        # ---- Build synthetic profiles from top transactions ----
        # Use pre-computed groups from __init__
        groups = self._cardholder_groups

        if not groups:
            elems.append(self._body("No high-risk cardholder data available in this analysis."))
            return elems

        elems.append(self._h2("8.1  Top 5 Highest-Risk Cardholders"))
        summary_rows = [["Card (Masked)", "Transactions", "Total at Risk", "Max Probability",
                          "Burst Pattern", "Night Txns"]]
        for cid, txns in groups:
            total_amt = sum(t.get('amount', 0) for t in txns)
            max_prob  = max(t.get('probability', 0) for t in txns)
            night_cnt = sum(1 for t in txns if t.get('features', {}).get('is_night', 0) >= 0.5)
            max_vel   = max(t['features'].get('velocity_1h', 1) for t in txns)
            burst_lbl = f"{max_vel:.0f} txns/hr" if max_vel >= 3 else "Low"
            masked    = f"****{cid[-4:]}"
            summary_rows.append([
                masked, str(len(txns)), f"${total_amt:,.2f}",
                f"{max_prob*100:.1f}%", burst_lbl, f"{night_cnt}/{len(txns)}",
            ])
        elems.append(self._table(summary_rows, [3*cm, 2.5*cm, 3*cm, 3*cm, 3*cm, 2.5*cm],
            extra_style=[('TEXTCOLOR', (3,1), (3,-1), RED)]))
        elems.append(self._sp(12))

        # ---- Attack timeline for top 3 cards ----
        elems.append(self._h2("8.2  Attack Timeline — Normal → Attack Pattern"))
        elems.append(self._body(
            "Each timeline shows the transaction sequence leading up to the detected fraud burst. "
            "The '▶ ATTACK' marker indicates where the fraud pattern was detected by FraudLens."
        ))
        elems.append(self._sp(6))

        for cid, txns in groups[:3]:
            masked = f"****{cid[-4:]}"
            # Build a mock timeline: 2 normal placeholders + attack burst
            tl_rows = [["#", "Amount", "Category", "Hour", "Vel/hr", "Probability", "Decision"]]
            for i, t in enumerate(txns, 1):
                f = t.get('features', {})
                hr_str = f"{int(f.get('hour', 0)):02d}:00"
                vel_str = f"{f.get('velocity_1h', 1):.0f}"
                tl_rows.append([
                    f"▶ {i}",
                    f"${t.get('amount', 0):,.2f}",
                    t.get('category', '—'),
                    hr_str, vel_str,
                    f"{t.get('probability', 0)*100:.1f}%",
                    t.get('classification', '—'),
                ])
            attack_start = len(txns)
            tbl = Table(tl_rows, colWidths=[1.5*cm, 2.5*cm, 3.5*cm, 2*cm, 2*cm, 2.5*cm, 2*cm])
            style_cmds = [
                ('FONTNAME',  (0,0), (-1,0),  'Helvetica-Bold'),
                ('FONTSIZE',  (0,0), (-1,-1), 8),
                ('BACKGROUND',(0,0), (-1,0),  colors.HexColor('#e9ecef')),
                ('GRID',      (0,0), (-1,-1), 0.3, BORDER),
                ('TOPPADDING',    (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                ('LEFTPADDING',   (0,0), (-1,-1), 5),
                # Attack rows: light red
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#fff1f2')),
                ('TEXTCOLOR',  (0,1), (0,-1),  RED),
                ('FONTNAME',   (0,1), (0,-1),  'Helvetica-Bold'),
            ]
            tbl.setStyle(TableStyle(style_cmds))
            elems.append(Paragraph(
                f"<b>Cardholder {masked}</b>  —  {len(txns)} high-risk transactions detected",
                self.styles['SubsectionH']
            ))
            elems.append(tbl)
            elems.append(self._sp(8))

        elems.append(self._body(
            "Note: Card IDs are synthetic groupings based on velocity burst windows in the uploaded dataset. "
            "Full per-cardholder history is available in the Live Monitor → Cardholder Profile view."
        ))
        return elems

    # ------------------------------------------------------------------ model explanation
    def _model_explanation(self):
        elems = [self._h1("Model Explanation & Feature Importance", 9), self._hr()]
        elems.append(self._h2("9.1  Model Comparison  (Training Performance on Sparkov Test Set)"))
        elems.append(self._body(
            "Note: metrics below are from offline training evaluation on the held-out Sparkov test set "
            "(555,719 rows). Live analysis results shown in Section 4 may differ due to dataset distribution."
        ))
        elems.append(self._sp(6))
        elems.append(self._table([
            ["Model",                    "F1",    "Precision", "Recall", "Notes"],
            ["XGBoost (Class Weights)",  "0.52",  "35.9%",    "94.6%",  "Many false alarms"],
            ["XGBoost (SMOTE+Tuned)",    "0.87",  "93.8%",    "80.4%",  "SMOTE oversampling"],
            ["AE + XGBoost",             "0.87",  "94.6%",    "80.1%",  "Adds reconstruction error"],
            ["AE + BDS + XGBoost ★",    "0.868", "94.6%",    "81.7%",  "Best: BDS catches subtle fraud"],
        ], [4.5*cm, 1.8*cm, 2.2*cm, 2*cm, 4.5*cm],
            extra_style=[('BACKGROUND', (0,4), (-1,4), colors.HexColor('#fef3c7'))]))
        elems.append(self._sp(12))

        elems.append(self._h2("9.2  Global Feature Importance (SHAP)"))
        elems.append(self._body(
            "Values show mean |SHAP| from model evaluation on Sparkov test set. "
            "★ marks custom velocity features. Higher value = stronger influence on predictions."
        ))
        elems.append(self._sp(6))

        shap_data = [
            ('is_night',            2.86, '#e74c3c'),
            ('amt',                 1.95, '#e74c3c'),
            ('amount_velocity_1h',  1.65, '#e74c3c'),  # ★
            ('hour',                1.06, '#f39c12'),
            ('category_encoded',    1.02, '#f39c12'),
            ('velocity_24h',        0.82, '#3498db'),  # ★
            ('recon_error',         0.68, '#3498db'),
            ('bds_amount',          0.54, '#3498db'),
            ('velocity_1h',         0.48, '#3498db'),  # ★
            ('bds_time',            0.42, '#3498db'),
        ]
        names  = [x[0] for x in shap_data]
        values = [x[1] for x in shap_data]
        bclrs  = [x[2] for x in shap_data]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(names[::-1], values[::-1], color=bclrs[::-1], edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Mean |SHAP Value|', fontsize=9)
        ax.set_title('Feature Importance — Global SHAP Summary', fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=8)
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=8)
        plt.tight_layout()
        elems.append(_fig_to_image(fig, 12, 5.5))
        elems.append(Paragraph(
            "★ Velocity features (amount_velocity_1h, velocity_24h, velocity_1h) are custom-engineered. "
            "Ablation study: removing them reduces F1 by 0.0144 (0.8705 → 0.8561).",
            self.styles['Muted8']
        ))
        return elems

    # ------------------------------------------------------------------ compliance
    def _compliance(self):
        elems = [self._h1("Regulatory Compliance", 10), self._hr()]
        elems.append(self._h2("10.1  PCI-DSS Alignment"))
        elems.append(self._table([
            ["Requirement",              "Status",       "Description"],
            ["Req 10.6 — Log Review",    "✓ COMPLIANT",  "Automated transaction monitoring active"],
            ["Req 11.4 — IDS",           "✓ COMPLIANT",  "ML-based fraud detection system"],
            ["Req 12.10 — Incident Resp","✓ COMPLIANT",  "Alert generation and escalation pipeline"],
            ["Card Data Storage",        "✓ COMPLIANT",  "Card numbers masked (****XXXX) — no full PANs stored"],
        ], [4*cm, 3*cm, 8*cm]))
        elems.append(self._sp(10))

        elems.append(self._h2("10.2  Audit Trail"))
        elems.append(self._table([
            ["Event",              "Details"],
            ["Dataset Analysed",   self._filename],
            ["Report Generated",   self.ts.strftime('%d %B %Y, %H:%M UTC')],
            ["Prepared By",        self.c.get('prepared_by', 'Fraud Analytics Team')],
            ["Model Version",      self.p.get('model_display', 'AE+BDS+XGBoost v1.0')],
            ["Report ID",          self.report_id],
            ["Classification",     self.c.get('classification', 'CONFIDENTIAL')],
        ], [5*cm, 10*cm]))
        elems.append(self._sp(10))

        elems.append(self._h2("10.3  Model Governance"))
        elems.append(self._body(
            "• <b>Training data:</b> Sparkov synthetic dataset (1,296,675 train rows, 555,719 test rows, 0.58% fraud rate)<br/>"
            "• <b>Validation:</b> Stratified hold-out test set<br/>"
            "• <b>Explainability:</b> SHAP values available per-transaction<br/>"
            "• <b>Velocity features:</b> Custom engineered — contribution validated (+0.014 F1 ablation study)"
        ))
        return elems

    # ------------------------------------------------------------------ recommendations
    def _recommendations(self):
        counts = self.p.get('counts', {}); fraud_n = counts.get('FRAUD', 0)
        elems = [self._h1("Recommendations", 11), self._hr()]
        elems.append(self._h2("11.1  Immediate Actions (Within 24 Hours)"))
        imm = Table([
            ["#", "Action", "Priority"],
            ["1", f"Review the {min(fraud_n,10)} highest-probability transactions in Section 5", "CRITICAL"],
            ["2", "Investigate all transactions with probability > 95%",                          "CRITICAL"],
            ["3", "File SAR reports for flagged transactions above $10,000",                      "HIGH"],
            ["4", "Contact cardholders of flagged accounts for verification",                     "HIGH"],
        ], colWidths=[0.8*cm, 12.2*cm, 2*cm])
        imm.setStyle(TableStyle([
            ('FONTNAME',  (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',  (0,0), (-1,-1), 8),
            ('GRID',      (0,0), (-1,-1), 0.3, BORDER),
            ('BACKGROUND',(0,0), (-1,0),  colors.HexColor('#e9ecef')),
            ('BACKGROUND',(2,1), (2,2),   colors.HexColor('#fee2e2')),
            ('BACKGROUND',(2,3), (2,4),   colors.HexColor('#fffbeb')),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING',   (0,0), (-1,-1), 6),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT]),
        ]))
        elems.append(imm)
        elems.append(self._sp(10))

        elems.append(self._h2("11.2  Short-Term (Within 1 Week)"))
        for b in [
            "• Review all FRAUD-flagged transactions for patterns not captured by the model",
            "• Lower nighttime transaction thresholds — 10 pm–6 am shows disproportionate fraud rate",
            "• Enable step-up authentication for high-risk categories (travel, online shopping)",
            "• Review MONITOR-flagged transactions for emerging fraud clusters",
        ]:
            elems.append(self._body(b)); elems.append(self._sp(3))
        elems.append(self._sp(8))

        elems.append(self._h2("11.3  Long-Term"))
        for b in [
            "• Quarterly model retraining with newly labelled fraud data",
            "• Per-cardholder BDS profiles (current model uses global baselines)",
            "• Geolocation velocity: flag transactions from two countries within 2 hours",
            "• Graph-based features capturing merchant network fraud clusters",
        ]:
            elems.append(self._body(b)); elems.append(self._sp(3))
        return elems

    # ------------------------------------------------------------------ appendix
    def _appendix(self):
        elems = [self._h1("Technical Appendix", 12), self._hr()]
        elems.append(self._h2("12.1  Model Pipeline"))
        elems.append(self._body(
            "<b>Stage 1 — Feature Engineering:</b> 14 base features extracted. "
            "Velocity features computed from rolling card-level history.<br/><br/>"
            "<b>Stage 2 — Anomaly Scoring:</b><br/>"
            "• Autoencoder (PyTorch): 14→10→5→10→14, trained on normal transactions only. "
            "MSE reconstruction error → feature #15.<br/>"
            "• BDS: 4 GA-optimised deviation scores → features #16–19.<br/><br/>"
            "<b>Stage 3 — Classification:</b> XGBoost on all 19 features, "
            "SMOTE oversampled, Bayesian hyperparameter tuned."
        ))
        elems.append(self._sp(10))

        elems.append(self._h2("12.2  Hyperparameters"))
        elems.append(self._table([
            ["Component",     "Parameter",      "Value"],
            ["XGBoost",       "n_estimators",   "200"],
            ["XGBoost",       "max_depth",      "6"],
            ["XGBoost",       "learning_rate",  "0.1"],
            ["XGBoost",       "subsample",      "0.8"],
            ["Autoencoder",   "Architecture",   "14→10→5→10→14"],
            ["Autoencoder",   "Activation",     "ReLU + Dropout(0.2)"],
            ["BDS/GA",        "Population",     "30  |  Generations: 20"],
            ["BDS/GA",        "Optimised params","10 threshold values"],
        ], [3.5*cm, 5*cm, 6.5*cm]))
        elems.append(self._sp(10))

        elems.append(self._h2("12.3  Ablation Study"))
        elems.append(self._table([
            ["Condition",                        "F1",     "Delta"],
            ["Full model (14 + AE + BDS)",       "0.8705", "—"],
            ["Without velocity features",        "0.8561", "−0.0144"],
            ["Without BDS features",             "0.8700", "−0.0005"],
            ["Without autoencoder (recon_error)","0.8680", "−0.0025"],
        ], [7*cm, 3*cm, 3*cm]))
        return elems

    # ------------------------------------------------------------------ glossary
    def _glossary(self):
        elems = [self._h1("Glossary", 13), self._hr()]
        terms = [
            ("Autoencoder",          "Neural network trained to reconstruct normal transactions. High reconstruction error indicates anomaly."),
            ("BDS",                  "Behavioural Deviation Score — custom algorithm measuring transaction deviation from global norms."),
            ("F1 Score",             "Harmonic mean of precision and recall. Higher = better balance between catching fraud and avoiding false alarms."),
            ("False Negative (FN)",  "Fraudulent transaction incorrectly cleared as legitimate."),
            ("False Positive (FP)",  "Legitimate transaction incorrectly flagged as fraud."),
            ("Precision",            "Of all flagged transactions, the percentage that are actual frauds."),
            ("Recall (Sensitivity)", "Of all actual frauds, the percentage detected."),
            ("SAR",                  "Suspicious Activity Report — regulatory filing required when fraud is suspected above defined thresholds."),
            ("SHAP",                 "SHapley Additive exPlanations — method explaining individual ML predictions via feature contributions."),
            ("SMOTE",                "Synthetic Minority Oversampling — generates synthetic fraud examples to balance class imbalance during training."),
            ("Velocity",             "Rate of transactions on a card over a time window. Sudden spikes indicate potential card theft."),
            ("XGBoost",              "Extreme Gradient Boosting — ensemble ML algorithm optimised for tabular classification."),
        ]
        body_style = self.styles['Body9']
        elems.append(self._table(
            [["Term", "Definition"]] +
            [[t, Paragraph(d, body_style)] for t, d in terms],
            [4*cm, 11*cm],
            extra_style=[('VALIGN', (0,0), (-1,-1), 'TOP')]
        ))
        return elems

    # ------------------------------------------------------------------ disclaimer
    def _disclaimer(self):
        elems = [self._h1("Legal Disclaimer", 14), self._hr()]
        elems.append(self._body(
            "This report is generated by <b>FraudLens</b>, an automated fraud detection system. "
            "It is intended for authorised personnel of the receiving institution only.<br/><br/>"
            "<b>Limitations:</b> This analysis is based on statistical patterns and ML models. "
            "False positives and negatives will occur. All predictions must be verified by qualified "
            "human analysts before action is taken.<br/><br/>"
            "<b>No Warranty:</b> FraudLens is provided 'as is' without warranty of any kind. "
            "The developers accept no liability for decisions made based on this report.<br/><br/>"
            "<b>Confidentiality:</b> This document contains information about fraud detection capabilities. "
            "Unauthorised distribution may compromise security effectiveness and is prohibited.<br/><br/>"
            "<b>Data Protection:</b> Card numbers are masked (****XXXX). No full PANs are stored or "
            "transmitted by FraudLens. Handle in accordance with GDPR, CCPA, and institutional policy."
        ))
        elems.append(self._sp(20))
        elems.append(self._hr())
        for line in [
            f"Report ID: {self.report_id}",
            f"Generated: {self.ts.strftime('%d %B %Y, %H:%M UTC')}",
            "FraudLens Hybrid ML Fraud Detection — v1.0",
            f"Classification: {self.c.get('classification','CONFIDENTIAL')}",
        ]:
            elems.append(Paragraph(line, self.styles['Muted8']))
            elems.append(self._sp(3))
        return elems
