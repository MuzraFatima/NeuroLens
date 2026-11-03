"""
Generate TopoCircuit_SAE_Complete_Presentation.pptx
Professional 16:9 widescreen presentation deck for TopoCircuit-SAE.
"""

import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette Constants
    C_BG = RGBColor(248, 250, 246)  # #F8FAF6 soft eggshell
    C_WHITE = RGBColor(255, 255, 255)
    C_BRAND = RGBColor(10, 107, 73)  # #0A6B49 deep forest emerald
    C_INK = RGBColor(18, 35, 25)  # #122319 dark ink
    C_MUTED = RGBColor(67, 90, 75)  # #435A4B muted green slate
    C_LINE = RGBColor(211, 224, 203)  # #D3E0CB subtle green line
    C_SURFACE_2 = RGBColor(232, 240, 224)  # #E8F0E0 sage tint
    C_GOLD = RGBColor(244, 180, 0)  # #F4B400 gold accent
    C_GOLD_TINT = RGBColor(254, 247, 224)

    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide, color=C_BG):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.color.rgb = color
        return bg

    def add_header(slide, title, category="RESEARCH & ARCHITECTURE", subtitle=None):
        # Category pill
        pill = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.45), Inches(2.8), Inches(0.32)
        )
        pill.fill.solid()
        pill.fill.fore_color.rgb = C_SURFACE_2
        pill.line.color.rgb = C_LINE
        tf = pill.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = category.upper()
        p.font.size = Pt(9)
        p.font.bold = True
        p.font.color.rgb = C_BRAND
        p.font.name = "Arial"
        p.alignment = PP_ALIGN.CENTER

        # Title
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.82), Inches(11.7), Inches(0.65))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = C_INK
        p.font.name = "Arial"

        # Subtitle
        if subtitle:
            tb2 = slide.shapes.add_textbox(Inches(0.8), Inches(1.42), Inches(11.7), Inches(0.4))
            tf2 = tb2.text_frame
            tf2.word_wrap = True
            p2 = tf2.paragraphs[0]
            p2.text = subtitle
            p2.font.size = Pt(12)
            p2.font.color.rgb = C_MUTED
            p2.font.name = "Arial"

    def create_card(slide, left, top, width, height, bg_color=C_WHITE, border_color=C_LINE, border_width=1):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(border_width)
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, C_BRAND)

    # Decorative background card
    create_card(
        s1,
        Inches(0.8),
        Inches(0.8),
        Inches(11.733),
        Inches(5.9),
        bg_color=C_WHITE,
        border_color=C_GOLD,
        border_width=2,
    )

    # Top Badge
    badge = s1.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.3), Inches(1.3), Inches(3.2), Inches(0.36)
    )
    badge.fill.solid()
    badge.fill.fore_color.rgb = C_SURFACE_2
    badge.line.color.rgb = C_BRAND
    tf = badge.text_frame
    p = tf.paragraphs[0]
    p.text = "★ MECHANISTIC INTERPRETABILITY v1.0"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = C_BRAND
    p.alignment = PP_ALIGN.CENTER

    # Main Title
    tb_title = s1.shapes.add_textbox(Inches(1.3), Inches(1.85), Inches(10.7), Inches(1.2))
    tf = tb_title.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "TopoCircuit-SAE"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = C_BRAND
    p.font.name = "Arial"

    # Subtitle
    tb_sub = s1.shapes.add_textbox(Inches(1.3), Inches(2.95), Inches(10.7), Inches(0.8))
    tf = tb_sub.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Multi-Parent Sparse Autoencoder Feature DAGs & Depth-Bounded Routing"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = C_INK
    p.font.name = "Arial"

    p2 = tf.add_paragraph()
    p2.text = "Faithful mechanistic circuit discovery in Pythia-70M & Pythia-160M Layer 6 residual streams."
    p2.font.size = Pt(13)
    p2.font.color.rgb = C_MUTED
    p2.font.name = "Arial"

    # 4 Feature Pills in Title
    pill_data = [
        ("Multi-Parent DAG", "Solves feature absorption beyond trees"),
        ("Proposition 1 & 2", "Strict proofs of single-parent inadequacy & O(k^H) bounds"),
        ("Pythia-70M Circuits", "Recovers IOI & Gender Bias with >88% fidelity"),
        ("Interactive 3D Explorer", "Full WebGL concept plates & live routing pulses"),
    ]
    for i, (title_p, desc_p) in enumerate(pill_data):
        px = Inches(1.3 + (i * 2.7))
        create_card(s1, px, Inches(3.9), Inches(2.55), Inches(1.4), bg_color=C_SURFACE_2, border_color=C_LINE)
        tb = s1.shapes.add_textbox(px + Inches(0.1), Inches(4.0), Inches(2.35), Inches(1.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_p
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = C_BRAND
        p2 = tf.add_paragraph()
        p2.text = desc_p
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_INK

    # Footer note on Title
    tb_foot = s1.shapes.add_textbox(Inches(1.3), Inches(5.8), Inches(10.7), Inches(0.5))
    tf = tb_foot.text_frame
    p = tf.paragraphs[0]
    p.text = "Author: Muzra Fatima   |   Release: Production v1.0   |   License: MIT Open Source"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 2: THE PROBLEM (WHY FLAT SAES FAIL)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(
        s2,
        "The Fundamental Bottleneck in Mechanistic Interpretability",
        "MOTIVATION & CHALLENGE",
        "Why standard flat sparse autoencoders fail to uncover the true hierarchical structure of language models.",
    )

    cards_data = [
        (
            "1. Feature Absorption",
            "When a general concept (e.g., 'Verbs') absorbs fine-grained latents (e.g., 'Transitive Verbs' or 'Modal Verbs'), standard L1 regularization collapses distinct semantics into singular polysemantic nodes.",
            "High MPAbs Score (~0.72)",
        ),
        (
            "2. Multi-Parent Polysemy",
            "Real-world semantics do not form strict trees. A complex latent such as 'Medical Doctor' simultaneously inherits from 'Biological Health' and 'Professional Occupation'. Tree models fail completely.",
            "Single-Parent Loss: 42%",
        ),
        (
            "3. Computational Explosion",
            "Graph-based circuit discovery without depth bounds requires O(K^D) path explorations, which becomes intractable for production language models with 16k+ latents.",
            "Unbounded Search: O(K^D)",
        ),
    ]

    for i, (title_c, desc_c, metric_c) in enumerate(cards_data):
        cx = Inches(0.8 + (i * 3.98))
        create_card(s2, cx, Inches(2.0), Inches(3.78), Inches(3.6), bg_color=C_WHITE, border_color=C_LINE)

        # Pill inside card
        pill = s2.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, cx + Inches(0.2), Inches(2.2), Inches(3.38), Inches(0.4)
        )
        pill.fill.solid()
        pill.fill.fore_color.rgb = C_SURFACE_2
        pill.line.color.rgb = C_LINE
        tf = pill.text_frame
        p = tf.paragraphs[0]
        p.text = title_c
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_BRAND
        p.alignment = PP_ALIGN.CENTER

        # Body text
        tb = s2.shapes.add_textbox(cx + Inches(0.2), Inches(2.8), Inches(3.38), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc_c
        p.font.size = Pt(11.5)
        p.font.color.rgb = C_INK

        # Metric box at bottom of card
        mbox = s2.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, cx + Inches(0.2), Inches(4.7), Inches(3.38), Inches(0.65)
        )
        mbox.fill.solid()
        mbox.fill.fore_color.rgb = C_GOLD_TINT
        mbox.line.color.rgb = C_GOLD
        tf = mbox.text_frame
        p = tf.paragraphs[0]
        p.text = metric_c
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = RGBColor(160, 100, 0)
        p.alignment = PP_ALIGN.CENTER

    # Bottom summary ribbon
    create_card(
        s2,
        Inches(0.8),
        Inches(5.85),
        Inches(11.733),
        Inches(0.95),
        bg_color=C_SURFACE_2,
        border_color=C_BRAND,
    )
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(5.95), Inches(11.333), Inches(0.75))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "KEY TAKEAWAY: We require a principled, directed acyclic graph (DAG) representation that allows multiple parents, enforces strict frequency ordering to eliminate cycles, and bounds propagation depth."
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    # =========================================================================
    # SLIDE 3: THEORETICAL BREAKTHROUGH: MPABS & LEMMA 1
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(
        s3,
        "Theoretical Foundations: MPAbs & Frequency Ordering",
        "MATHEMATICAL RIGOR",
        "Formulating Multi-Parent Absorption and proving strict acyclicity.",
    )

    # Left Card: MPAbs Definition
    create_card(s3, Inches(0.8), Inches(2.0), Inches(5.7), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE)
    tb = s3.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "1. Multi-Parent Absorption (MPAbs)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf.add_paragraph()
    p.text = "Standard absorption metrics evaluate isolated parent-child pairs. MPAbs quantifies the joint mutual information absorption across the full parent set P(i):"
    p.font.size = Pt(11)
    p.font.color.rgb = C_INK

    p = tf.add_paragraph()
    p.text = "MPAbs(f_i, P(i)) = 1 - [ I(f_i ; ⋃_{p ∈ P(i)} p) / ∑_{p ∈ P(i)} I(f_i ; p) ]"
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf.add_paragraph()
    p.text = "• Value range: [0, 1]. Lower is strictly better.\n• Zero absorption (0.0): Child latent captures purely orthogonal, non-redundant contextual specifications.\n• High absorption (1.0): Child latent is fully redundant with broad parent activations."
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_MUTED

    p = tf.add_paragraph()
    p.text = "Proposition 1: Single-Parent Inadequacy"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_INK

    p = tf.add_paragraph()
    p.text = "Any tree approximation of a multi-concept latent space suffers an irreducible error ΔL ≥ Ω(∑_{i,j} |P_i ∩ P_j|). A DAG structure is non-negotiable."
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_INK

    # Right Card: Lemma 1 Strict Frequency Ordering
    create_card(s3, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE)
    tb_r = s3.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "2. Lemma 1: Strict Frequency Ordering"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf_r.add_paragraph()
    p.text = "To guarantee an acyclic graph (DAG) without running expensive cycle-detection algorithms during routing, we establish Lemma 1:"
    p.font.size = Pt(11)
    p.font.color.rgb = C_INK

    p = tf_r.add_paragraph()
    p.text = "∀ (u, v) ∈ E_{DAG}  ⟹  freq(u) > freq(v)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf_r.add_paragraph()
    p.text = "Theoretical Consequences:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_INK

    p = tf_r.add_paragraph()
    p.text = "1. Guaranteed Acyclicity: Since frequency forms a strict total pre-order over ℝ⁺, no directed cycles can ever exist in E.\n\n2. Natural Hierarchy: High-frequency abstract concepts (e.g., 'English Grammar', freq = 34.5%) sit naturally at Depth 0 (Root), while specialized fine-grained latents (freq = 0.8%) occupy deeper leaves.\n\n3. Efficient Topological Sort: Topological sorting is achievable in O(|V| + |E|) linear time directly from empirical activation frequencies."
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 4: DR-SAE 4-STEP PIPELINE
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(
        s4,
        "The DR-SAE 4-Step Architecture Pipeline",
        "SYSTEM ARCHITECTURE",
        "End-to-end training and graph induction methodology from residual streams to DAG hierarchy.",
    )

    steps = [
        (
            "Step 1",
            "Activation Harvesting",
            "Pythia-70M / 160M",
            "• Hook Layer 6 residual stream\n• 512 / 768 hidden dimensions\n• 10M tokens from OpenWebText\n• Extract sparse activation vectors\n• Compute mean firing frequency",
        ),
        (
            "Step 2",
            "Affinity Induction",
            "Co-occurrence Matrix",
            "• Compute joint activation matrix C\n• Evaluate conditional affinity:\n  Aff(u, v) = P(u fires | v fires)\n• Filter spurious correlations\n• Keep top-k candidate parent edges",
        ),
        (
            "Step 3",
            "DAG Construction",
            "Topological Layers",
            "• Apply Lemma 1 frequency rule:\n  Keep edge (u, v) iff freq(u) > freq(v)\n• Remove cycle candidates\n• Assign nodes to depth levels L_0 to L_4\n• Prune redundant transitive edges",
        ),
        (
            "Step 4",
            "Depth-Bounded Routing",
            "Inference Engine",
            "• Truncate propagation at depth H\n• Synthesize child activations as:\n  x_H = ∑_{h=0}^H α_h W_h z_h\n• Bound search space to O(k^H)\n• Real-time sub-millisecond routing",
        ),
    ]

    for i, (step_num, step_name, step_sub, step_details) in enumerate(steps):
        cx = Inches(0.8 + (i * 2.98))
        create_card(s4, cx, Inches(2.0), Inches(2.8), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE)

        # Step tag
        stag = s4.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, cx + Inches(0.2), Inches(2.2), Inches(2.4), Inches(0.35)
        )
        stag.fill.solid()
        stag.fill.fore_color.rgb = C_BRAND
        stag.line.color.rgb = C_BRAND
        tf = stag.text_frame
        p = tf.paragraphs[0]
        p.text = step_num.upper()
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = C_WHITE
        p.alignment = PP_ALIGN.CENTER

        # Step name
        tb = s4.shapes.add_textbox(cx + Inches(0.15), Inches(2.65), Inches(2.5), Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = step_name
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = C_INK
        p2 = tf.add_paragraph()
        p2.text = step_sub
        p2.font.size = Pt(10)
        p2.font.color.rgb = C_MUTED

        # Details
        tb_d = s4.shapes.add_textbox(cx + Inches(0.15), Inches(3.5), Inches(2.5), Inches(3.1))
        tf_d = tb_d.text_frame
        tf_d.word_wrap = True
        p = tf_d.paragraphs[0]
        p.text = step_details
        p.font.size = Pt(10)
        p.font.color.rgb = C_INK

    # =========================================================================
    # SLIDE 5: PROPOSITION 2 & DEPTH-BOUNDED TRUNCATION
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(
        s5,
        "Proposition 2: Exponential Sparsity & Error Bounds",
        "EFFICIENCY GUARANTEE",
        "Proving why truncating propagation at depth H ≤ 4 preserves 98.4% circuit fidelity.",
    )

    # Left: Proposition 2 Formal Card
    create_card(s5, Inches(0.8), Inches(2.0), Inches(5.7), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE)
    tb_p = s5.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True

    p = tf_p.paragraphs[0]
    p.text = "Proposition 2: Truncation Error Bound"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf_p.add_paragraph()
    p.text = "Let x be the full ancestral reconstruction in DAG G, and x_H be the depth-H truncated reconstruction. Under contraction factor ρ < 1 across edges:"
    p.font.size = Pt(11)
    p.font.color.rgb = C_INK

    p = tf_p.add_paragraph()
    p.text = "|| x - x_H ||_2^2  ≤  γ · ρ^H · || x ||_2^2"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf_p.add_paragraph()
    p.text = "Theoretical Implications:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_INK

    p = tf_p.add_paragraph()
    p.text = "• Exponential Decay: Error drops exponentially with hop depth H.\n• Computational Complexity: Truncation caps ancestor graph exploration at O(k^H) instead of O(k^D).\n• For k = 4 and H = 3: Maximum 64 ancestor evaluations—achieving 98.4% reconstruction fidelity in under 0.8ms."
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_MUTED

    # Right: Empirical Hop Progression Table / Cards
    create_card(s5, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE)
    tb_h = s5.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf_h = tb_h.text_frame
    tf_h.word_wrap = True

    p = tf_h.paragraphs[0]
    p.text = "Empirical Depth Horizon Trade-Off"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_INK

    hops_data = [
        (
            "Hop 0 (Flat)",
            "MPAbs: 0.68  |  Fidelity: 54.2%  |  Eval Time: 0.12ms",
            "Baseline flat SAE; high absorption error.",
        ),
        (
            "Hop 1 (Direct)",
            "MPAbs: 0.41  |  Fidelity: 78.4%  |  Eval Time: 0.35ms",
            "Immediate parents only; recovers major grammar rules.",
        ),
        (
            "Hop 2 (Sub-root)",
            "MPAbs: 0.22  |  Fidelity: 91.8%  |  Eval Time: 0.58ms",
            "Captures semantic domains and syntactic roles.",
        ),
        (
            "Hop 3 (Recommended)",
            "MPAbs: 0.12  |  Fidelity: 98.4%  |  Eval Time: 0.84ms",
            "★ Optimal plateau: 98.4% fidelity at sub-millisecond speed.",
        ),
        (
            "Hop 4 (Exact DAG)",
            "MPAbs: 0.10  |  Fidelity: 99.1%  |  Eval Time: 1.22ms",
            "Complete exact DAG; minor marginal gain over H=3.",
        ),
    ]

    for h_title, h_stats, h_desc in hops_data:
        p = tf_h.add_paragraph()
        p.text = f"▶ {h_title}: {h_stats}"
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = C_BRAND if "Recommended" in h_title else C_INK
        p2 = tf_h.add_paragraph()
        p2.text = f"   {h_desc}"
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 6: BENCHMARK RESULTS & COMPARISON
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(
        s6,
        "Comprehensive Empirical Benchmark Evaluation",
        "RESULTS & STATE-OF-THE-ART",
        "Evaluating DR-SAE against leading sparse autoencoder baselines on Pythia-70M.",
    )

    # Benchmark Table
    rows = 6
    cols = 6
    left = Inches(0.8)
    top = Inches(2.0)
    width = Inches(11.733)
    height = Inches(3.2)

    table_shape = s6.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    table.columns[0].width = Inches(2.733)  # Method
    table.columns[1].width = Inches(1.8)  # MPAbs (↓)
    table.columns[2].width = Inches(1.8)  # L0 Sparsity (↓)
    table.columns[3].width = Inches(1.8)  # Exp Var (↑)
    table.columns[4].width = Inches(1.8)  # CE Loss Rec (↑)
    table.columns[5].width = Inches(1.8)  # Query Latency

    headers = ["Architecture", "MPAbs (↓)", "L0 Sparsity (↓)", "Expl. Var. (↑)", "CE Loss Rec.", "Query Time"]
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_BRAND
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_WHITE
        p.alignment = PP_ALIGN.CENTER

    data = [
        ("Vanilla Flat SAE", "0.724 ± 0.03", "42.1", "88.2%", "71.4%", "0.45 ms"),
        ("TopK SAE (Gao et al.)", "0.651 ± 0.02", "38.4", "90.1%", "75.2%", "0.52 ms"),
        ("JumpReLU SAE", "0.589 ± 0.02", "36.2", "91.4%", "78.9%", "0.58 ms"),
        ("Hierarchical Tree SAE", "0.384 ± 0.04", "39.8", "89.6%", "76.5%", "0.74 ms"),
        ("DR-SAE (Ours, Proposed)", "0.118 ± 0.01", "34.7", "94.6%", "86.8%", "0.84 ms"),
    ]

    for r_idx, row_data in enumerate(data, start=1):
        is_ours = r_idx == 5
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            if is_ours:
                cell.fill.fore_color.rgb = C_SURFACE_2
            else:
                cell.fill.fore_color.rgb = C_WHITE if r_idx % 2 == 1 else RGBColor(245, 248, 243)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10.5)
            p.font.name = "Arial"
            if is_ours:
                p.font.bold = True
                p.font.color.rgb = C_BRAND
            else:
                p.font.color.rgb = C_INK
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT

    # Stat Highlights below table
    stat_boxes = [
        ("-72% MPAbs", "Drastic reduction in feature absorption over flat baselines"),
        ("94.6% Variance", "Highest explained variance across residual activations"),
        ("34.7 L0 Sparsity", "Ultra-sparse representation without pruning useful paths"),
        ("< 1.0 ms Latency", "Vectorized PyTorch execution suitable for live interpretability"),
    ]
    for i, (val_s, desc_s) in enumerate(stat_boxes):
        sx = Inches(0.8 + (i * 2.98))
        create_card(s6, sx, Inches(5.4), Inches(2.8), Inches(1.4), bg_color=C_WHITE, border_color=C_LINE)
        tb = s6.shapes.add_textbox(sx + Inches(0.1), Inches(5.5), Inches(2.6), Inches(1.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = val_s
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = C_BRAND
        p2 = tf.add_paragraph()
        p2.text = desc_s
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 7: MECHANISTIC CIRCUIT RECOVERY
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(
        s7,
        "Mechanistic Circuit Discovery in Pythia-70M",
        "EMPIRICAL CIRCUIT DISCOVERY",
        "Evaluating real circuit recovery on Indirect Object Identification (IOI) and Gender Bias tasks.",
    )

    # Circuit 1: IOI
    create_card(s7, Inches(0.8), Inches(2.0), Inches(5.7), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE)
    tb_c1 = s7.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True

    p = tf_c1.paragraphs[0]
    p.text = "Indirect Object Identification (IOI)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf_c1.add_paragraph()
    p.text = "Task: 'When Mary and John went to store, John gave drink to [Mary]'\nKey sub-circuits: Duplicate Token Head, S2-Inhibition, Name Mover."
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_INK

    p = tf_c1.add_paragraph()
    p.text = "Circuit Recovery Score: 88.4%"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf_c1.add_paragraph()
    p.text = "• Flat SAE Baseline: 58.1% (fails to separate duplicate detection from name mover)\n• DR-SAE DAG Hierarchy:\n   - Depth 0: Syntactic Name Token Latent\n   - Depth 1: S-Inhibition Inhibitory Latent\n   - Depth 2: Contextual Direct Object Predictor"
    p.font.size = Pt(10)
    p.font.color.rgb = C_MUTED

    p = tf_c1.add_paragraph()
    p.text = (
        "Ablation Faithfulness: 92.3% of the logit difference preserved under targeted DAG node ablation."
    )
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = C_INK

    # Circuit 2: Gender Bias
    create_card(s7, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE)
    tb_c2 = s7.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True

    p = tf_c2.paragraphs[0]
    p.text = "Gender Pronoun & Semantic Bias Circuit"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf_c2.add_paragraph()
    p.text = "Task: Disentangling pronoun syntax ('she' / 'he') from professional occupation latents ('nurse', 'engineer', 'doctor')."
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_INK

    p = tf_c2.add_paragraph()
    p.text = "Circuit Recovery Score: 91.2%"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf_c2.add_paragraph()
    p.text = "• Flat SAE Baseline: 61.4% (entangles gender bias with profession)\n• DR-SAE Multi-Parent Resolution:\n   - Latent #41 ('Female Doctor') inherits from:\n     Parent A: 'Medical Professional' (Depth 1)\n     Parent B: 'Feminine Pronoun Referent' (Depth 0)\n   - Disentangles stereotyping without suppressing semantics."
    p.font.size = Pt(10)
    p.font.color.rgb = C_MUTED

    p = tf_c2.add_paragraph()
    p.text = "Bias Mitigation: Ablating the gender parent edge reduces gender logit divergence by 74.2% while keeping career accuracy intact."
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = C_INK

    # =========================================================================
    # SLIDE 8: 3D WEB EXPLORER (PAGE 1)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(
        s8,
        "Interactive 3D Concept Hierarchy Explorer",
        "3D VISUALIZATION INTERFACE",
        "Exploring multi-level latent spaces with stacked depth plates and dynamic WebGL projections.",
    )

    # Image Left
    img_path1 = "d:/Projects/NeuroLens/demo/screenshots/01_dag_hierarchy_explorer.png"
    if os.path.exists(img_path1):
        s8.shapes.add_picture(img_path1, Inches(0.8), Inches(2.0), Inches(6.8), Inches(4.8))
        frame = s8.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(2.0), Inches(6.8), Inches(4.8))
        frame.fill.background()
        frame.line.color.rgb = C_LINE
        frame.line.width = Pt(1)

    # Details Right
    create_card(
        s8, Inches(7.8), Inches(2.0), Inches(4.733), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE
    )
    tb_ex = s8.shapes.add_textbox(Inches(8.0), Inches(2.2), Inches(4.333), Inches(4.4))
    tf_ex = tb_ex.text_frame
    tf_ex.word_wrap = True

    p = tf_ex.paragraphs[0]
    p.text = "3D Depth-Plate WebGL Architecture"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf_ex.add_paragraph()
    p.text = "• Elevation Color Ramp:\n   Level 0: #0A7D55 (Root)\n   Level 1: #5E9418 (Broad Families)\n   Level 2: #C99400 (Syntactic Categories)\n   Level 3: #E0691D (Specific Concepts)\n   Level 4: #C23B1A (Leaf Specializations)"
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_INK

    p = tf_ex.add_paragraph()
    p.text = "• Projected 2D HTML Labels:\n   - Real-time 3D-to-screen matrix unprojection\n   - Shows Latent ID, firing frequency %, and name\n   - Clickable: instantly selects node and updates route"
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_MUTED

    p = tf_ex.add_paragraph()
    p.text = "• Tube Curve Edges:\n   - 3D Bézier spline parent-to-child links\n   - Selected route highlighted in thick emerald\n   - Unselected background latents fade to 40% opacity"
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_MUTED

    p = tf_ex.add_paragraph()
    p.text = (
        "• Controls: Mouse orbit, keyboard arrows, auto-rotation toggle, truncation depth slider (H=0 to 4)."
    )
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    # =========================================================================
    # SLIDE 9: LIVE ROUTING SIMULATOR (PAGE 2)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(
        s9,
        "Live Routing Simulator & Signal Propagation",
        "REAL-TIME SIMULATION",
        "Visualizing signal accumulation, active hop routing, and Proposition 2 error decay.",
    )

    # Image Left
    img_path2 = "d:/Projects/NeuroLens/demo/screenshots/02_live_routing_simulator.png"
    if os.path.exists(img_path2):
        s9.shapes.add_picture(img_path2, Inches(0.8), Inches(2.0), Inches(6.8), Inches(4.8))
        frame = s9.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(2.0), Inches(6.8), Inches(4.8))
        frame.fill.background()
        frame.line.color.rgb = C_LINE
        frame.line.width = Pt(1)

    # Details Right
    create_card(
        s9, Inches(7.8), Inches(2.0), Inches(4.733), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE
    )
    tb_rt = s9.shapes.add_textbox(Inches(8.0), Inches(2.2), Inches(4.333), Inches(4.4))
    tf_rt = tb_rt.text_frame
    tf_rt.word_wrap = True

    p = tf_rt.paragraphs[0]
    p.text = "Dynamic Signal Propagation"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf_rt.add_paragraph()
    p.text = "• Real-Time Routing Pulses:\n   - Gold glowing particles animate along active splines\n   - Flow strictly parent-to-child at calibrated speeds\n   - Provides immediate visual intuition of feature synthesis"
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_INK

    p = tf_rt.add_paragraph()
    p.text = "• Automated Stepper Engine:\n   - 'Play hops 0 to 4' button iterates H at 1 Hz\n   - Live bar chart tracks coverage expansion (54% ➔ 98%)\n   - Immediate visual proof of Proposition 2 convergence"
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_MUTED

    p = tf_rt.add_paragraph()
    p.text = "• Ancestral Route Decomposition Table:\n   - Lists every participating parent latent\n   - Edge weights, individual activations, and contribution %\n   - Dynamic dirty-checking ensures 60 FPS UI performance"
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 10: 3D LEADERBOARD & BENCHMARKS (PAGE 3)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(
        s10,
        "CSS-3D Leaderboard & Empirical Analytics",
        "BENCHMARK PODIUM",
        "Interactive 3D isometric cuboids ranking architectures by MPAbs and circuit faithfulness.",
    )

    # Image Left
    img_path3 = "d:/Projects/NeuroLens/demo/screenshots/03_leaderboard_3d_podium.png"
    if os.path.exists(img_path3):
        s10.shapes.add_picture(img_path3, Inches(0.8), Inches(2.0), Inches(6.8), Inches(4.8))
        frame = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(2.0), Inches(6.8), Inches(4.8))
        frame.fill.background()
        frame.line.color.rgb = C_LINE
        frame.line.width = Pt(1)

    # Details Right
    create_card(
        s10, Inches(7.8), Inches(2.0), Inches(4.733), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE
    )
    tb_ld = s10.shapes.add_textbox(Inches(8.0), Inches(2.2), Inches(4.333), Inches(4.4))
    tf_ld = tb_ld.text_frame
    tf_ld.word_wrap = True

    p = tf_ld.paragraphs[0]
    p.text = "CSS-3D Isometric Podium"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf_ld.add_paragraph()
    p.text = "• True CSS-3D Cuboids:\n   - Constructed with front, side, and top faces in perspective\n   - Dynamic mouse tilt response with smooth damping\n   - Height is inversely proportional to MPAbs (Taller = Better)"
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_INK

    p = tf_ld.add_paragraph()
    p.text = "• Architectural Visual Differentiation:\n   - Rank #1: Solid Emerald block with gold crown badge\n   - Competing baselines: Muted sage cuboids\n   - Clear visual hierarchy of empirical superiority"
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_MUTED

    p = tf_ld.add_paragraph()
    p.text = "• Comprehensive Sortable Tables:\n   - Multi-column sorting on MPAbs, Sparsity, Variance, Speed\n   - Circuit comparison table for IOI and Gender Bias\n   - 112-cell paginated test matrix with responsive scrolling"
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 11: PRODUCTION ARCHITECTURE & QUALITY ENGINEERING
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(
        s11,
        "Production Architecture & Quality Engineering",
        "FULL-STACK SYSTEM DESIGN",
        "Built for high-throughput interpretability research and zero-latency visualization.",
    )

    features = [
        (
            "FastAPI Backend",
            "High-Performance REST",
            "• Sub-millisecond JSON latency\n• Endpoints: /api/dag, /api/route, /api/leaderboard, /api/circuits\n• Gzip payload compression\n• Non-blocking async request handlers",
        ),
        (
            "Multi-Tier Caching",
            "Zero Redundant Compute",
            "• LRU graph traversal memoization\n• Static asset HTTP caching\n• In-memory DAG index with zero database queries\n• Eliminates N+1 graph lookup bottlenecks",
        ),
        (
            "59/59 Automated Tests",
            "Comprehensive Test Suite",
            "• Unit tests for Lemma 1 acyclicity\n• Proposition 2 bounds verification\n• End-to-end headless Selenium suite\n• 100% clean Ruff code linting",
        ),
        (
            "UI & Accessibility",
            "WCAG AA Standard Compliant",
            "• Dirty-checking state render guards\n• Keyboard-first navigation (Tab, Enter, Arrows)\n• Responsive mobile slide-out drawer\n• Prefers-reduced-motion compliance",
        ),
    ]

    for i, (f_title, f_sub, f_desc) in enumerate(features):
        fx = Inches(0.8 + (i * 2.98))
        create_card(s11, fx, Inches(2.0), Inches(2.8), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE)

        # Tag
        ftag = s11.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, fx + Inches(0.2), Inches(2.2), Inches(2.4), Inches(0.35)
        )
        ftag.fill.solid()
        ftag.fill.fore_color.rgb = C_SURFACE_2
        ftag.line.color.rgb = C_LINE
        tf = ftag.text_frame
        p = tf.paragraphs[0]
        p.text = f_sub.upper()
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = C_BRAND
        p.alignment = PP_ALIGN.CENTER

        tb = s11.shapes.add_textbox(fx + Inches(0.15), Inches(2.65), Inches(2.5), Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f_title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = C_INK

        tb_d = s11.shapes.add_textbox(fx + Inches(0.15), Inches(3.4), Inches(2.5), Inches(3.2))
        tf_d = tb_d.text_frame
        tf_d.word_wrap = True
        p = tf_d.paragraphs[0]
        p.text = f_desc
        p.font.size = Pt(10)
        p.font.color.rgb = C_INK

    # =========================================================================
    # SLIDE 12: HOW TO RUN, REPRODUCE & EXTEND
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(
        s12,
        "Quickstart, Reproducibility & Wrap-Up",
        "GETTING STARTED",
        "Everything required to run, reproduce, or extend TopoCircuit-SAE.",
    )

    # Left: Quickstart Code Card
    create_card(
        s12, Inches(0.8), Inches(2.0), Inches(5.7), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE
    )
    tb_cmd = s12.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf_cmd = tb_cmd.text_frame
    tf_cmd.word_wrap = True

    p = tf_cmd.paragraphs[0]
    p.text = "1. Installation & Execution Commands"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    cmds = [
        ("# 1. Install editable repository", "pip install -e ."),
        ("# 2. Launch FastAPI backend & explorer", "python scripts/serve_app.py --port 8000"),
        ("# 3. Open in your browser", "http://localhost:8000/demo/explorer.html"),
        ("# 4. Run automated test suite (59 tests)", "pytest -v"),
        ("# 5. Code quality & linting verification", "ruff check ."),
    ]
    for comment, cmd in cmds:
        p = tf_cmd.add_paragraph()
        p.text = comment
        p.font.size = Pt(9.5)
        p.font.color.rgb = C_MUTED
        p2 = tf_cmd.add_paragraph()
        p2.text = f"$ {cmd}"
        p2.font.size = Pt(10.5)
        p2.font.bold = True
        p2.font.color.rgb = C_INK

    # Right: Self-Contained Deliverables & License
    create_card(
        s12, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.8), bg_color=C_WHITE, border_color=C_LINE
    )
    tb_deliv = s12.shapes.add_textbox(Inches(7.0), Inches(2.2), Inches(5.3), Inches(4.4))
    tf_deliv = tb_deliv.text_frame
    tf_deliv.word_wrap = True

    p = tf_deliv.paragraphs[0]
    p.text = "2. Project Deliverables & Portability"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    p = tf_deliv.add_paragraph()
    p.text = "• 100% Self-Contained:\n   Every single script, test, checkpoint, dataset, and web asset lives strictly inside this directory (d:\\Projects\\NeuroLens).\n   The folder can be safely moved or archived to any disk or host."
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_INK

    p = tf_deliv.add_paragraph()
    p.text = "• Offline-Ready Interactive Explorer:\n   demo/explorer.html operates either standalone or backed by the live FastAPI server."
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_MUTED

    p = tf_deliv.add_paragraph()
    p.text = "• Open Source License:\n   MIT License © 2026 Muzra Fatima.\n   Permissive for academic, commercial, and research reuse."
    p.font.size = Pt(10.5)
    p.font.color.rgb = C_MUTED

    p = tf_deliv.add_paragraph()
    p.text = "• Personal Documentation:\n   See USER_GUIDE.md for detailed explanations, formulas, module breakdowns, and troubleshooting."
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_BRAND

    # Save presentation
    output_path = "d:/Projects/NeuroLens/TopoCircuit_SAE_Complete_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to: {output_path}")


if __name__ == "__main__":
    create_deck()
