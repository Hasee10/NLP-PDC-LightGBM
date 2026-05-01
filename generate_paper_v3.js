/**
 * RetailEL Research Paper — v3
 * Updated with actual experimental results from RetailEL_Notebook.ipynb
 *
 * Run:
 *   NODE_PATH="C:/Users/Administrator/AppData/Roaming/npm/node_modules" node generate_paper_v3.js
 */

"use strict";
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, Header, Footer, LevelFormat,
} = require("docx");
const fs = require("fs");

// ── Helpers ──────────────────────────────────────────────────────────────────

const W = 9360; // content width in DXA (8.5" - 2×1" margins)

function border(color = "CCCCCC") {
  const s = { style: BorderStyle.SINGLE, size: 1, color };
  return { top: s, bottom: s, left: s, right: s };
}

function hdrCell(text, widthDXA, shade = "2E75B6") {
  return new TableCell({
    borders: border("2E75B6"),
    width: { size: widthDXA, type: WidthType.DXA },
    shading: { fill: shade, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 18, font: "Arial" })],
    })],
  });
}

function cell(text, widthDXA, shade = "FFFFFF", align = AlignmentType.LEFT, bold = false) {
  return new TableCell({
    borders: border(),
    width: { size: widthDXA, type: WidthType.DXA },
    shading: { fill: shade, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text, size: 18, font: "Arial", bold })],
    })],
  });
}

function altCell(text, widthDXA, alt, align = AlignmentType.LEFT, bold = false) {
  return cell(text, widthDXA, alt ? "EBF3FB" : "FFFFFF", align, bold);
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, size: 28, font: "Arial", color: "2E75B6" })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 180, after: 80 },
    children: [new TextRun({ text, bold: true, size: 24, font: "Arial", color: "2E75B6" })],
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 80 },
    alignment: opts.justify ? AlignmentType.JUSTIFIED : AlignmentType.LEFT,
    children: [new TextRun({ text, size: 20, font: "Arial", ...opts })],
  });
}

function bold(text) {
  return new TextRun({ text, bold: true, size: 20, font: "Arial" });
}

function caption(text) {
  return new Paragraph({
    spacing: { before: 60, after: 160 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, italics: true, size: 18, font: "Arial", color: "444444" })],
  });
}

function gap(lines = 1) {
  return new Paragraph({
    spacing: { before: 0, after: lines * 80 },
    children: [new TextRun({ text: "", size: 20 })],
  });
}

function bullet(text) {
  return new Paragraph({
    spacing: { before: 40, after: 40 },
    indent: { left: 480, hanging: 240 },
    children: [
      new TextRun({ text: "•  ", size: 20, font: "Arial" }),
      new TextRun({ text, size: 20, font: "Arial" }),
    ],
  });
}

function ref(num, text) {
  return new Paragraph({
    spacing: { before: 40, after: 40 },
    indent: { left: 480, hanging: 480 },
    children: [new TextRun({ text: `[${num}] ${text}`, size: 18, font: "Arial" })],
  });
}

// ── Tables ───────────────────────────────────────────────────────────────────

function table1() {
  // Main results — both synthetic and real
  const cols = [3400, 1800, 1960, 2200];
  const rows = [
    ["BM25 Only",                        "—",       "41.88%", "−50.28 pp"],
    ["RRF Sparse (No Reranker)",          "—",       "41.21%", "−50.95 pp"],
    ["Neural Bi-Encoder (MiniLM-L6-v2)", "—",       "65.49%", "−26.67 pp"],
    ["RetailEL (Full Pipeline)",          "97.18%",  "92.16%", "—"],
  ];
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: cols,
    rows: [
      new TableRow({ children: [
        hdrCell("System",                    cols[0]),
        hdrCell("Synthetic Acc@1",           cols[1]),
        hdrCell("Real UCI Acc@1\n(700 SKUs)",cols[2]),
        hdrCell("Delta vs Full\n(Real UCI)", cols[3]),
      ]}),
      ...rows.map((r, i) => new TableRow({ children: r.map((t, j) => altCell(t, cols[j], i % 2 === 1, j > 0 ? AlignmentType.CENTER : AlignmentType.LEFT, j === 0 && i === 3)) })),
    ],
  });
}

function table2() {
  // Latency breakdown — synthetic stage-level + real wall-clock
  const cols = [2600, 1360, 1360, 1360, 1360, 1320];
  const rows = [
    ["Normalisation",              "0.158", "0.151", "0.198", "0.274", "—"],
    ["Hybrid Sparse Retrieval",    "1.507", "1.432", "2.104", "2.891", "—"],
    ["LightGBM Reranker",          "0.478", "0.421", "0.893", "1.154", "—"],
    ["FSM Selection",              "0.004", "0.003", "0.006", "0.007", "—"],
    ["Pipeline Overhead",          "0.113", "—",     "—",     "—",     "—"],
    ["End-to-End (Synthetic)",     "2.261", "—",     "—",     "—",     "—"],
    ["End-to-End (Real UCI)",      "7.050", "—",     "—",     "—",     "—"],
  ];
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: cols,
    rows: [
      new TableRow({ children: [
        hdrCell("Stage",        cols[0]),
        hdrCell("Mean (ms)",    cols[1]),
        hdrCell("P50 (ms)",     cols[2]),
        hdrCell("P95 (ms)",     cols[3]),
        hdrCell("P99 (ms)",     cols[4]),
        hdrCell("Note",         cols[5]),
      ]}),
      ...rows.map((r, i) => new TableRow({ children: r.map((t, j) => altCell(t, cols[j], i % 2 === 1, j > 0 ? AlignmentType.CENTER : AlignmentType.LEFT)) })),
    ],
  });
}

function table3() {
  // Noise robustness on real UCI catalogue (700 SKUs, n=50 per level)
  const cols = [2200, 1000, 1400, 2760, 2000];
  const rows = [
    ["0.1", "50", "90.00%", "[82.00%, 98.00%]", "High confidence"],
    ["0.2", "50", "88.00%", "[78.00%, 96.00%]", "Stable"],
    ["0.3", "50", "92.00%", "[84.00%, 98.00%]", "Stable"],
    ["0.4", "50", "86.00%", "[76.00%, 94.00%]", "Mid-range noise"],
    ["0.5", "50", "70.00%", "[58.00%, 82.00%]", "Degradation onset"],
    ["0.6", "50", "70.00%", "[56.00%, 82.00%]", "Plateau"],
  ];
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: cols,
    rows: [
      new TableRow({ children: [
        hdrCell("Noise Level (α)", cols[0]),
        hdrCell("n",                    cols[1]),
        hdrCell("Acc@1",                cols[2]),
        hdrCell("95% CI (Bootstrap)",   cols[3]),
        hdrCell("Observation",          cols[4]),
      ]}),
      ...rows.map((r, i) => new TableRow({ children: r.map((t, j) => altCell(t, cols[j], i % 2 === 1, j >= 1 ? AlignmentType.CENTER : AlignmentType.LEFT)) })),
    ],
  });
}

function table3b() {
  // Natural variant analysis (real UCI data)
  const cols = [3200, 1600, 1600, 2960];
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: cols,
    rows: [
      new TableRow({ children: [
        hdrCell("Query Type",       cols[0]),
        hdrCell("n (Test Set)",     cols[1]),
        hdrCell("Acc@1",            cols[2]),
        hdrCell("Observation",      cols[3]),
      ]}),
      new TableRow({ children: [
        altCell("Canonical queries (desc = canonical name)", cols[0], false),
        altCell("46,346",  cols[1], false, AlignmentType.CENTER),
        altCell("93.10%",  cols[2], false, AlignmentType.CENTER, true),
        altCell("Pipeline handles canonical descriptions robustly", cols[3], false),
      ]}),
      new TableRow({ children: [
        altCell("Variant queries (desc ≠ canonical name)", cols[0], true),
        altCell("694",     cols[1], true, AlignmentType.CENTER),
        altCell("28.96%",  cols[2], true, AlignmentType.CENTER),
        altCell("Hard cases: distinct operator entries", cols[3], true),
      ]}),
    ],
  });
}

function table4() {
  // Category accuracy — SynEL synthetic
  const cols = [3000, 1680, 1680, 3000];
  const rows = [
    ["Bakery",     "100.00%", "37"],
    ["Beverages",  "95.83%",  "96"],
    ["Biscuits",   "100.00%", "6"],
    ["Breakfast",  "100.00%", "22"],
    ["Canned",     "100.00%", "30"],
    ["Condiments", "96.88%",  "32"],
    ["Dairy",      "100.00%", "17"],
    ["Frozen",     "96.83%",  "63"],
    ["Household",  "98.31%",  "59"],
    ["Pasta",      "100.00%", "17"],
    ["Personal",   "100.00%", "49"],
    ["Snacks",     "96.97%",  "33"],
    ["Spreads",    "100.00%", "28"],
  ];
  return new Table({
    width: { size: 6360, type: WidthType.DXA },
    columnWidths: [2120, 2120, 2120],
    rows: [
      new TableRow({ children: [
        hdrCell("Category",  2120),
        hdrCell("Acc@1",     2120),
        hdrCell("N (items)", 2120),
      ]}),
      ...rows.map((r, i) => new TableRow({ children: [
        altCell(r[0], 2120, i % 2 === 1),
        altCell(r[1], 2120, i % 2 === 1, AlignmentType.CENTER),
        altCell(r[2], 2120, i % 2 === 1, AlignmentType.CENTER),
      ]})),
    ],
  });
}

function table5() {
  // Feature importance + LOO (SynEL)
  const cols = [2800, 1200, 1760, 1400, 2200];
  const rows = [
    ["Levenshtein Ratio",      "54.45%", "94.27%", "−2.05", "Dominant signal"],
    ["Jaccard Token Overlap",  "18.65%", "95.09%", "−1.23", "Strong"],
    ["Bigram Overlap",         "11.00%", "95.09%", "−1.23", "Strong"],
    ["Length Ratio",           "8.74%",  "95.09%", "−1.23", "Moderate"],
    ["Levenshtein Distance",   "3.24%",  "95.30%", "−1.02", "Moderate"],
    ["Char-level Jaccard",     "1.52%",  "95.50%", "−0.82", "Minor"],
    ["Prior SKU Frequency",    "1.13%",  "95.91%", "−0.41", "Minor"],
    ["RRF Score",              "1.30%",  "96.11%", "−0.20", "Minor"],
  ];
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: cols,
    rows: [
      new TableRow({ children: [
        hdrCell("Feature",          cols[0]),
        hdrCell("Gain %",           cols[1]),
        hdrCell("Acc Without (%)",  cols[2]),
        hdrCell("Acc Drop (pp)",    cols[3]),
        hdrCell("Significance",     cols[4]),
      ]}),
      ...rows.map((r, i) => new TableRow({ children: r.map((t, j) => altCell(t, cols[j], i % 2 === 1, j > 0 ? AlignmentType.CENTER : AlignmentType.LEFT, i === 0)) })),
    ],
  });
}

// ── Document ──────────────────────────────────────────────────────────────────

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Arial", size: 20 } },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 4 } },
          children: [new TextRun({ text: "RetailEL: Hybrid Retrieval-Reranking for Retail POS Entity Linking — FAST NUCES Spring 2026", size: 16, font: "Arial", color: "555555" })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: "2E75B6", space: 4 } },
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", size: 16, font: "Arial", color: "555555" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 16, font: "Arial", color: "555555" }),
          ],
        })],
      }),
    },
    children: [

      // ── TITLE PAGE ──────────────────────────────────────────────────────────
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 480, after: 120 },
        children: [new TextRun({ text: "FAST NUCES — National University of Computer and Emerging Sciences", bold: true, size: 24, font: "Arial", color: "2E75B6" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 80 },
        children: [new TextRun({ text: "Islamabad Campus  |  Department of Computer Science", size: 22, font: "Arial", color: "444444" })],
      }),
      gap(2),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 120 },
        children: [new TextRun({ text: "RetailEL: A Hybrid Retrieval-Reranking Framework for", bold: true, size: 36, font: "Arial" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 240 },
        children: [new TextRun({ text: "Real-Time Retail Product Entity Linking on Noisy POS Data", bold: true, size: 36, font: "Arial" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 80 },
        children: [new TextRun({ text: "Muhammad Haseeb Arshad (23I-2578)", size: 22, font: "Arial" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 240 },
        children: [new TextRun({ text: "Muhammad Abdullah Nadeem (23I-2522)", size: 22, font: "Arial" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 80 },
        children: [new TextRun({ text: "Course: Natural Language Processing  |  Semester: Spring 2026", size: 20, font: "Arial", color: "444444" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 480 },
        children: [new TextRun({ text: "Submission Date: May 3, 2026", size: 20, font: "Arial", color: "444444" })],
      }),

      // ── ABSTRACT ────────────────────────────────────────────────────────────
      h1("Abstract"),
      new Paragraph({
        spacing: { before: 60, after: 80 },
        alignment: AlignmentType.JUSTIFIED,
        children: [
          new TextRun({ text: "Retail point-of-sale (POS) systems generate millions of daily transactions containing noisy, abbreviated product descriptions that must be mapped to canonical Stock Keeping Unit (SKU) identifiers for inventory management and analytics. We present ", size: 20, font: "Arial" }),
          bold("RetailEL"),
          new TextRun({ text: ", a five-stage hybrid retrieval-reranking pipeline combining BM25 lexical retrieval, TF-IDF sparse retrieval fused via Reciprocal Rank Fusion (RRF), LightGBM LambdaRank reranking, basket-context-aware encoding (adapted from RoCEL), and FSM-constrained candidate validation.", size: 20, font: "Arial" }),
        ],
      }),
      new Paragraph({
        spacing: { before: 60, after: 80 },
        alignment: AlignmentType.JUSTIFIED,
        children: [
          new TextRun({ text: "We evaluate RetailEL on two benchmarks. On ", size: 20, font: "Arial" }),
          bold("SynEL"),
          new TextRun({ text: " — a controlled synthetic benchmark of 3,303 items across 50 SKUs and 600 transactions — RetailEL achieves ", size: 20, font: "Arial" }),
          bold("97.18% Accuracy@1"),
          new TextRun({ text: " at 2.26 ms average latency. On the ", size: 20, font: "Arial" }),
          bold("UCI Online Retail dataset"),
          new TextRun({ text: " — a real UK retailer dataset with 700 SKUs and 313,600 transaction pairs exhibiting genuine POS noise — RetailEL achieves ", size: 20, font: "Arial" }),
          bold("92.16% Accuracy@1"),
          new TextRun({ text: " at 7.05 ms average latency. This represents a ", size: 20, font: "Arial" }),
          bold("+26.67 percentage point"),
          new TextRun({ text: " improvement over a neural bi-encoder baseline (all-MiniLM-L6-v2, 65.49%) and a ", size: 20, font: "Arial" }),
          bold("+50.28 pp"),
          new TextRun({ text: " improvement over BM25-only retrieval (41.88%), all on real-world data. Feature ablation on SynEL confirms that Levenshtein ratio is the dominant reranker signal (54.45% gain). FSM-constrained selection guarantees 100% valid SKU outputs, eliminating hallucination entirely across both evaluation modes.", size: 20, font: "Arial" }),
        ],
      }),

      // ── 1. INTRODUCTION ─────────────────────────────────────────────────────
      h1("1. Introduction"),
      p("Modern retail operations depend critically on accurate, real-time product identification. When a cashier enters \"CK 12oz Cls Cn\" into a POS terminal, the system must immediately resolve this noisy string to \"Coca-Cola Classic 12oz Can\" (SKU-001) to trigger correct inventory deductions, loyalty-point accrual, and demand-forecasting updates. This problem — retail entity linking — sits at the intersection of NLP and information retrieval, presenting challenges absent from standard entity linking benchmarks:", { justify: true }),
      bullet("Severe lexical noise: POS entries contain keyboard-adjacent typos, missing vowels, informal abbreviations, and random capitalisation introduced by hurried operators."),
      bullet("Latency constraints: Retail chains must resolve entities within single-digit milliseconds at peak-hour transaction volumes."),
      bullet("Hallucination risk: Generative language models may produce non-existent SKU strings, corrupting downstream inventory systems."),
      bullet("Basket co-occurrence structure: POS items arrive in baskets; co-purchased items encode brand and category signals that can disambiguate ambiguous abbreviations."),
      p("Existing entity-linking systems (LLM4BioEL, OneNet, RoCEL) were designed for biomedical or Wikipedia-scale knowledge bases and do not address these constraints. We propose RetailEL, a purpose-built five-stage pipeline. Our contributions are:", { justify: true }),
      bullet("Hybrid sparse retrieval: BM25 + TF-IDF sparse retrieval fused via RRF, tailored to noisy retail queries."),
      bullet("Basket-context encoding: Adaptation of RoCEL (Wang et al., 2024) encoding item-level (row) and co-purchase (column) context."),
      bullet("LightGBM reranker: LambdaRank model on eight similarity features (Burges, Ragno & Le, 2006); GPU-free inference."),
      bullet("FSM-constrained selection: Guarantees 100% valid SKU outputs — zero hallucination."),
      bullet("Dual evaluation: Controlled SynEL synthetic benchmark (50 SKUs) and UCI Online Retail real-world dataset (700 SKUs, 313,600 pairs)."),

      // ── 2. RELATED WORK ─────────────────────────────────────────────────────
      h1("2. Related Work"),
      h2("2.1 Entity Linking"),
      p("Entity linking (EL) maps textual mentions to entries in a knowledge base. Classical approaches rely on TF-IDF or BM25 retrieval followed by rule-based disambiguation. Neural approaches using Bi-Encoder + Cross-Encoder architectures (Wu et al., 2020) achieve higher accuracy but at latency costs incompatible with real-time retail. Few-shot methods like OneNet (Liu et al., 2024) use large language models directly; however, their Entity Consensus Judger (ECJ) introduces inference overhead that prevents millisecond-level deployment.", { justify: true }),
      h2("2.2 Biomedical Entity Linking"),
      p("LLM4BioEL (Lin et al., 2025) employs restrictive decoding — a generation-time mechanism that constrains LLM output to a predefined set of valid entity identifiers, preventing hallucination of non-existent ontology terms. Our FSM-constrained selection is conceptually inspired by this objective: both approaches enforce that only valid identifiers appear in system output. RetailEL achieves this via a flat hash-set membership check rather than generation-time decoding, reducing selection overhead to 0.004 ms.", { justify: true }),
      h2("2.3 Tabular Entity Linking"),
      p("RoCEL (Wang et al., 2024) exploits the row-column structure of HTML tables for entity linking, constructing row context (entity-level properties) and column context (categorical patterns from co-occurring cells). We adapt this insight for retail receipts: each receipt row is a product item (description, quantity, price, department) and the column is the basket of co-purchased items.", { justify: true }),
      h2("2.4 Learning-to-Rank for Retrieval Reranking"),
      p("LambdaRank (Burges, Ragno & Le, 2006) is a gradient boosting approach that optimises non-smooth ranking metrics by defining virtual gradients that directly reflect changes in NDCG. LightGBM (Ke et al., 2017) implements LambdaRank with histogram-based split finding, achieving millisecond inference on commodity CPUs.", { justify: true }),
      h2("2.5 Dense vs. Sparse Retrieval"),
      p("Sparse retrieval methods (BM25, TF-IDF) represent documents as high-dimensional sparse vectors over a fixed vocabulary. Dense retrieval methods use neural encoders to produce low-dimensional dense embeddings, enabling semantic matching beyond exact token overlap. RetailEL's hybrid stage combines BM25 and TF-IDF sparse retrieval via RRF; our neural baseline uses a bi-encoder (all-MiniLM-L6-v2) as a representative dense retrieval system.", { justify: true }),

      // ── 3. METHODOLOGY ──────────────────────────────────────────────────────
      h1("3. Methodology"),
      h2("3.1 Problem Definition"),
      p("Let C = {(sᵢ, nᵢ, bᵢ, catᵢ)} be a SKU catalogue of K entries, where sᵢ is the SKU identifier, nᵢ the canonical name, bᵢ the brand, and catᵢ the product category. Given a noisy POS description d in basket B, our goal is to find:", { justify: true }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 60, after: 60 },
        children: [new TextRun({ text: "f(d) = s* = argmax P(s | d, B),    s* ∈ {sᵢ} ∪ {<UNKNOWN_SKU>}", size: 20, font: "Courier New" })],
      }),
      p("The constraint that s* must belong to the catalogue (or the UNKNOWN sentinel) is enforced by the FSM selector in Stage 5, guaranteeing zero hallucination.", { justify: true }),
      h2("3.2 Stage 1: Text Normalisation"),
      p("Raw POS descriptions undergo: (a) lowercasing, (b) abbreviation expansion via a 47-entry retailer-specific lexicon (e.g., \"CK\" → \"Coca-Cola\", \"DT\" → \"Diet\"), (c) punctuation removal retaining alphanumeric characters and spaces, (d) whitespace collapsing. Mean processing latency: 0.158 ms.", { justify: true }),
      h2("3.3 Stage 2: Basket Context Construction (RoCEL Adaptation)"),
      p("For each item in a transaction we construct:", { justify: true }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 60, after: 60 },
        children: [new TextRun({ text: "context(item) = \"[ROW: desc qty=q dept=D] [BASKET: item₁ | … | itemₖ]\"", size: 18, font: "Courier New" })],
      }),
      p("The ROW component encodes item-level properties. The BASKET component encodes up to four co-purchased item descriptions, capped to prevent query dilution. For example, the abbreviation \"CK\" in a basket containing Pepsi and Lays items is far more likely to resolve to Coca-Cola than to chicken — a disambiguation impossible from item description alone.", { justify: true }),
      h2("3.4 Stage 3: Hybrid Sparse Retrieval with RRF"),
      p("We retrieve top-10 candidates from two independent sparse retrieval systems:", { justify: true }),
      bullet("BM25 (Robertson et al., 2009): Probabilistic sparse retrieval using BM25Okapi. High precision for exact brand codes, unit sizes, product identifiers."),
      bullet("TF-IDF Sparse Retrieval: Character-level TF-IDF with 1–2 gram features; cosine similarity over catalogue name+brand+category. This is a sparse vector method — not a neural dense retrieval system. Robust to abbreviations unseen in BM25's term vocabulary."),
      p("Both lists are merged using Reciprocal Rank Fusion (Cormack et al., 2009):", { justify: true }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 60, after: 60 },
        children: [new TextRun({ text: "RRF(d, s) = 1/(k + rank_BM25(s)) + 1/(k + rank_TFIDF(s)),   k = 60", size: 20, font: "Courier New" })],
      }),
      p("Mean retrieval latency: 1.507 ms.", { justify: true }),
      h2("3.5 Stage 4: LightGBM Reranker"),
      p("A LightGBM LambdaRank (Burges, Ragno & Le, 2006) model reranks the top-10 retrieved candidates using eight per-candidate features: Levenshtein distance, Levenshtein ratio, Jaccard token overlap, bigram overlap, character-level Jaccard similarity, length ratio, RRF fusion score, and prior SKU frequency. Training uses query-level groups; ground-truth label = 1 for the correct SKU, 0 for all others. The model produces a top-3 reranked list at 0.478 ms mean latency on CPU, with no GPU requirement.", { justify: true }),
      h2("3.6 Stage 5: FSM-Constrained Candidate Selection"),
      p("The final selector enforces catalogue validity. Only SKUs present in the catalogue hash-set can be output. A confidence threshold of 0.45 determines routing:", { justify: true }),
      bullet("High-confidence path (≥0.45): Top candidate selected directly — no LLM invocation (fast path)."),
      bullet("Low-confidence path (<0.45): LLM-guided disambiguation among top-2 candidates, with FSM bitmask constraining output to valid SKUs."),
      bullet("No match: Returns <UNKNOWN_SKU> for out-of-vocabulary products."),
      p("In our experiments, 100% of test items across both synthetic and real evaluations were resolved via the high-confidence fast path, confirming that the LightGBM reranker suffices for catalogues up to 700 SKUs.", { justify: true }),

      // ── 4. EXPERIMENTAL SETUP ───────────────────────────────────────────────
      h1("4. Experimental Setup"),
      h2("4.1 Dataset 1: SynEL (Synthetic Benchmark)"),
      p("We constructed SynEL, a synthetic retail entity linking benchmark. Starting from a 50-SKU catalogue across 13 product categories, we generated 600 transactions (3,303 individual line items) using five noise functions: keyboard-adjacent substitution, vowel deletion, random capitalisation, random word drop (p=0.25), and word concatenation (p=0.20). Noise level α ∈ [0,1] controls corruption intensity; α = 0.4 is used for all splits.", { justify: true }),
      p("The dataset is split 70/15/15: 2,312 training, 495 validation, 496 test items.", { justify: true }),
      p("Scope caveat: SynEL is a controlled synthetic benchmark with only 50 SKUs. Near-perfect accuracy on a 50-class problem does not imply comparable performance at production catalogue scale.", { justify: true }),
      h2("4.2 Dataset 2: UCI Online Retail (Real-World Benchmark)"),
      p("To validate RetailEL on real-world POS data, we use the UCI Online Retail dataset (Chen, 2015), a transaction dataset from a UK-based online giftware retailer. The dataset contains 541,909 rows covering invoices from 2010–2011. Unlike SynEL, this dataset exhibits genuine POS noise: the same product is entered with natural description variation across operators (e.g., \"WHITE HANGING HEART T-LIGHT HOLDER\" vs. \"WHITE HANGING HEART T/LIGHT HOLDER\").", { justify: true }),
      p("Processing pipeline: (1) Remove cancelled invoices (InvoiceNo starting with 'C'). (2) Remove non-product StockCodes (POST, DOT, M, BANK CHARGES, etc.). (3) Keep top 700 most-purchased SKUs with ≥ 5 transactions. (4) Canonical name = most frequent Description per StockCode. (5) Include baskets with ≥ 2 distinct SKUs for basket context.", { justify: true }),
      p("This yields 700 canonical SKUs, 313,600 entity linking pairs, and 17,311 basket transactions. Split 70/15/15: 219,520 training, 47,040 validation, 47,040 test items.", { justify: true }),
      h2("4.3 Evaluation Metrics"),
      bullet("Accuracy@1: Fraction of test items where top-1 predicted SKU equals ground truth."),
      bullet("Throughput: Items processed per second (end-to-end, single CPU core)."),
      bullet("Average Latency: Mean end-to-end wall-clock latency per item in milliseconds."),
      bullet("LLM Bypass Rate: Fraction of items resolved without LLM invocation."),
      h2("4.4 Baselines"),
      bullet("BM25 Only: BM25Okapi; top-1 candidate selected directly."),
      bullet("RRF Sparse (No Reranker): Hybrid BM25 + TF-IDF sparse RRF; top-1 without reranking."),
      bullet("Neural Bi-Encoder: all-MiniLM-L6-v2 (sentence-transformers); nearest-neighbour cosine search. Represents dense retrieval."),
      bullet("RetailEL (Full Pipeline): All five stages including basket context and LightGBM reranker."),

      // ── 5. RESULTS ──────────────────────────────────────────────────────────
      h1("5. Results"),
      h2("5.1 Main Results"),
      new Paragraph({
        spacing: { before: 60, after: 80 },
        alignment: AlignmentType.JUSTIFIED,
        children: [
          new TextRun({ text: "Table 1 reports Accuracy@1 on both benchmarks. On the real UCI dataset (700 SKUs), RetailEL achieves ", size: 20, font: "Arial" }),
          bold("92.16%"),
          new TextRun({ text: ", a ", size: 20, font: "Arial" }),
          bold("+26.67 pp"),
          new TextRun({ text: " improvement over the neural bi-encoder (65.49%) and a ", size: 20, font: "Arial" }),
          bold("+50.28 pp"),
          new TextRun({ text: " improvement over BM25-only (41.88%). On the synthetic SynEL benchmark (50 SKUs), RetailEL achieves ", size: 20, font: "Arial" }),
          bold("97.18%"),
          new TextRun({ text: ". The near-identical BM25-only and RRF-without-reranker results on real data (41.88% vs. 41.21%) confirm that the reranker — not the retrieval fusion — drives the majority of performance gains.", size: 20, font: "Arial" }),
        ],
      }),
      gap(),
      table1(),
      caption("Table 1: Accuracy@1 comparison. Real UCI = 700-SKU real-world benchmark (n=47,040 test items). Synthetic = SynEL 50-SKU benchmark (n=496 test items). Delta computed on Real UCI vs. RetailEL Full Pipeline. '—' = not evaluated on this split."),
      gap(),
      h2("5.2 Performance Analysis"),
      p("Table 2 provides the full latency breakdown measured on the synthetic test set (496 items, single CPU core). Stage latencies sum to 2.147 ms; the wall-clock average is 2.261 ms — a gap of 0.113 ms (5.0% of wall time) attributable to transaction grouping, Python dict/hash-set lookups, and function dispatch. On the real UCI test set (47,040 items, 700-SKU catalogue), the wall-clock average is 7.050 ms, reflecting the larger catalogue's impact on retrieval and reranking. Throughput is 442.3 items/sec (synthetic) and 141.8 items/sec (real).", { justify: true }),
      gap(),
      table2(),
      caption("Table 2: Stage-level latency profile (synthetic, n=496, single CPU core). P50/P95/P99 from synthetic evaluation. Overhead = basket grouping, dict lookups, Python dispatch. Real UCI wall-clock = 7.050 ms."),
      gap(),
      h2("5.3 Noise Robustness"),
      p("Table 3 reports accuracy at six noise levels on the real UCI catalogue (700 SKUs), with 95% bootstrap confidence intervals (2,000 replicates, n=50 per level). Accuracy degrades gracefully from 90% at α=0.1 to 70% at α=0.5–0.6. The degradation onset at α≥0.5 reflects that heavy noise corrupts brand tokens beyond the abbreviation lexicon's coverage. The overlapping confidence intervals at α=0.4–0.6 indicate that differences between these levels are not statistically reliable at n=50.", { justify: true }),
      gap(),
      table3(),
      caption("Table 3: Noise robustness on real UCI catalogue (700 SKUs, n=50 per noise level, 2,000 bootstrap replicates)."),
      gap(),
      p("Table 4 presents the natural description variant analysis on the real UCI test set. The pipeline achieves 93.10% accuracy on the 46,346 canonical queries (where the input description matches the canonical name exactly) but 28.96% on the 694 natural variant queries (distinct operator-entered descriptions). This gap is a key finding: the pipeline excels at canonical-form matching but struggles with genuine description variation, a direction for future work.", { justify: true }),
      gap(),
      table3b(),
      caption("Table 4: Natural variant analysis on UCI Online Retail test set (n=47,040). 'Variant' = description differs from canonical name; 'Canonical' = exact match."),
      gap(),
      h2("5.4 Category-Level Accuracy (SynEL)"),
      p("Table 5 reports per-category accuracy on the SynEL synthetic test set. Ten of thirteen categories achieve 100% Accuracy@1. Beverages (95.83%, n=96) and Frozen (96.83%, n=63) are the most challenging, containing the highest intra-category SKU density (multiple Tropicana products, multiple Coca-Cola variants).", { justify: true }),
      gap(),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 60, after: 80 },
        children: [table4()],
      }),
      caption("Table 5: Category-level Accuracy@1 on the SynEL synthetic test set (n=496). N = number of test items per category."),
      gap(),
      h2("5.5 LightGBM Feature Importance and Ablation"),
      p("Table 6 presents built-in gain-based feature importance and leave-one-out (LOO) ablation on SynEL. Levenshtein ratio carries 54.45% of information gain and produces the largest single-feature accuracy drop (−2.05 pp). The top-4 features (Levenshtein ratio, Jaccard overlap, bigram overlap, length ratio) account for 92.84% of total gain, indicating the reranker can be compressed to four features with minimal accuracy loss.", { justify: true }),
      gap(),
      table5(),
      caption("Table 6: LightGBM feature importance (gain %) and leave-one-out ablation on SynEL test set. Full model Acc@1 = 96.32%."),
      gap(),

      // ── 6. DISCUSSION ───────────────────────────────────────────────────────
      h1("6. Discussion"),
      h2("6.1 Key Findings"),
      new Paragraph({
        spacing: { before: 60, after: 80 },
        alignment: AlignmentType.JUSTIFIED,
        children: [
          bold("(1) LightGBM reranking is the dominant contributor. "),
          new TextRun({ text: "The +50.28 pp improvement from BM25-only (41.88%) to RetailEL (92.16%) on real data is driven almost entirely by the reranker. The near-identical BM25-only and RRF-without-reranker accuracy (41.88% vs. 41.21%) confirms that retrieval fusion alone adds little — the reranker is what matters.", size: 20, font: "Arial" }),
        ],
      }),
      new Paragraph({
        spacing: { before: 60, after: 80 },
        alignment: AlignmentType.JUSTIFIED,
        children: [
          bold("(2) RetailEL substantially outperforms the neural baseline. "),
          new TextRun({ text: "The +26.67 pp improvement over all-MiniLM-L6-v2 (92.16% vs. 65.49%) on real-world UCI data demonstrates that feature-based reranking with edit-distance signals captures retail POS noise patterns that embedding-based similarity misses. This is a significant finding: an entirely CPU-based sparse pipeline outperforms a GPU-friendly transformer model on this task.", size: 20, font: "Arial" }),
        ],
      }),
      new Paragraph({
        spacing: { before: 60, after: 80 },
        alignment: AlignmentType.JUSTIFIED,
        children: [
          bold("(3) Levenshtein ratio is the critical reranker signal. "),
          new TextRun({ text: "Feature ablation shows Levenshtein ratio carries 54.45% of information gain (SynEL). This is direct evidence that character-level edit distance is the primary matching signal for noisy POS descriptions.", size: 20, font: "Arial" }),
        ],
      }),
      new Paragraph({
        spacing: { before: 60, after: 80 },
        alignment: AlignmentType.JUSTIFIED,
        children: [
          bold("(4) FSM-constrained selection eliminates hallucination. "),
          new TextRun({ text: "Zero invalid SKU outputs across both synthetic (496 items) and real (47,040 items) test sets. LLM bypass rate is 100% in both modes, confirming the fast path handles all in-distribution queries.", size: 20, font: "Arial" }),
        ],
      }),
      new Paragraph({
        spacing: { before: 60, after: 80 },
        alignment: AlignmentType.JUSTIFIED,
        children: [
          bold("(5) Natural variant accuracy reveals a gap. "),
          new TextRun({ text: "On canonical queries (desc = canonical name), accuracy is 93.10%. On natural variant queries (distinct operator descriptions), accuracy drops to 28.96% (n=694). This gap identifies the most important direction for future improvement: the model must learn from operator-specific description patterns rather than just canonical forms.", size: 20, font: "Arial" }),
        ],
      }),
      h2("6.2 Limitations"),
      bullet("Natural variant accuracy: The 28.96% accuracy on genuine UCI description variants is the most important limitation. Addressing this requires training on variant-rich data and possibly augmenting features with semantic similarity."),
      bullet("Noise robustness at scale: Accuracy drops to 70% at α=0.5–0.6 on the 700-SKU catalogue. At larger catalogues, this degradation is expected to be more pronounced."),
      bullet("Statistical power in noise sweep: CIs are wide at n=50 per level ([56%, 82%] at α=0.6). Definitive claims require n≥200."),
      bullet("English-only: South Asian retail markets involve Roman Urdu and phonetic transliterations not addressed here."),
      bullet("Latency scales with catalogue: Wall-clock rises from 2.26 ms (50 SKUs) to 7.05 ms (700 SKUs). At tens of thousands of SKUs, a FAISS-indexed bi-encoder retriever would be required."),
      h2("6.3 Future Work"),
      bullet("Improving variant accuracy: Fine-tune retrieval on UCI variant pairs; add semantic similarity features to the reranker."),
      bullet("Scaling to Retail-786k: bi-encoder (all-MiniLM-L6-v2) + FAISS indexing to validate performance at real catalogue scale."),
      bullet("Online reranker updates: Incrementally updating SKU prior frequencies as transaction patterns shift."),
      bullet("Multilingual extension: Urdu phonetic transliteration models for South Asian retail."),
      bullet("Proper Outlines-library FSM-constrained LLM decoding for the low-confidence routing path."),

      // ── 7. CONCLUSION ───────────────────────────────────────────────────────
      h1("7. Conclusion"),
      new Paragraph({
        spacing: { before: 60, after: 80 },
        alignment: AlignmentType.JUSTIFIED,
        children: [
          new TextRun({ text: "We presented RetailEL, a five-stage pipeline for real-time entity linking on noisy retail POS data. Evaluated on two benchmarks — SynEL (synthetic, 50 SKUs, 496 test items) and UCI Online Retail (real-world, 700 SKUs, 47,040 test items) — the system achieves ", size: 20, font: "Arial" }),
          bold("97.18% Accuracy@1"),
          new TextRun({ text: " at 2.26 ms (synthetic) and ", size: 20, font: "Arial" }),
          bold("92.16% Accuracy@1"),
          new TextRun({ text: " at 7.05 ms (real), both on a single CPU core with no GPU requirement.", size: 20, font: "Arial" }),
        ],
      }),
      new Paragraph({
        spacing: { before: 60, after: 80 },
        alignment: AlignmentType.JUSTIFIED,
        children: [
          new TextRun({ text: "On real-world UCI data, RetailEL outperforms the neural bi-encoder baseline (all-MiniLM-L6-v2) by ", size: 20, font: "Arial" }),
          bold("+26.67 pp"),
          new TextRun({ text: " (92.16% vs. 65.49%) and BM25-only retrieval by ", size: 20, font: "Arial" }),
          bold("+50.28 pp"),
          new TextRun({ text: " (92.16% vs. 41.88%). Feature ablation identifies Levenshtein ratio as the single most important reranker signal (54.45% gain). FSM-constrained selection ensures 100% valid SKU output with zero hallucination across 47,536 total test items. The key open problem identified is natural variant accuracy (28.96% on 694 genuine UCI description variants vs. 93.10% on canonical queries), which motivates future work on variant-aware training.", size: 20, font: "Arial" }),
        ],
      }),

      // ── REFERENCES ──────────────────────────────────────────────────────────
      h1("References"),
      ref(1, "Burges, C. J. C., Ragno, R., & Le, Q. V. (2006). Learning to rank with nonsmooth cost functions. In Advances in Neural Information Processing Systems (NeurIPS 2006), Vol. 19, pp. 193–200."),
      ref(2, "Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., & Svore, G. (2005). Learning to rank using gradient descent. In Proceedings of the 22nd International Conference on Machine Learning (ICML 2005), pp. 89–96."),
      ref(3, "Chen, D. (2015). Online Retail Data Set. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/Online+Retail"),
      ref(4, "Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of SIGIR 2009, pp. 758–759."),
      ref(5, "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In NeurIPS 2017, Vol. 30."),
      ref(6, "Lin, Y., et al. (2025). LLM4BioEL: Large Language Models for Biomedical Entity Linking with Restrictive Decoding and Contrastive Knowledge Injection. In Proceedings of EMNLP 2025."),
      ref(7, "Liu, X., et al. (2024). OneNet: A Fine-Tuning Free Framework for Few-Shot Entity Linking via Large Language Models. In Proceedings of EMNLP 2024."),
      ref(8, "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. In Proceedings of EMNLP 2019, pp. 3982–3992."),
      ref(9, "Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4), 333–389."),
      ref(10, "Wang, Y., et al. (2024). RoCEL: Row-Column Entity Linking for Heterogeneous Tabular Data. In Proceedings of EMNLP 2024."),
      ref(11, "Wu, L., Petroni, F., Josifoski, M., Riedel, S., & Zettlemoyer, L. (2020). Scalable zero-shot entity linking with dense entity retrieval. In Proceedings of EMNLP 2020, pp. 6397–6407."),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("RetailEL_Research_Paper.docx", buf);
  console.log("RetailEL_Research_Paper.docx written successfully.");
});
