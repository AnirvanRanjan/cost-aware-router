// Builds slides/cost_aware_router_deck.pptx (npm install pptxgenjs): the team's combined presentation (T56).
// Content comes from notes/lane_{a,b,c,d}_presentation_slides.md and the locked numbers in
// notes/t27_t28_locked_numbers.md; every slide's speaker notes name the source of its numbers.
//   node slides/build_deck.js
const path = require("path");
const pptxgen = require("pptxgenjs");
// Optional: writes THEME's colours and fonts into the .pptx theme (only matters when editing in PowerPoint).
let applyTheme = async () => {};
try { ({ applyTheme } = require(process.env.PPTX_SKILL_DIR
  ? path.join(process.env.PPTX_SKILL_DIR, "scripts/apply_theme.js")
  : "/mnt/skills/public/pptx/scripts/apply_theme.js")); } catch (e) { /* not available: colours are set per shape anyway */ }

const OUT = path.join(__dirname, "cost_aware_router_deck.pptx");
const INK = "1C1B19", MUTED = "6B665C", RULE = "D9D3C5", LOCAL = "2F6B3F", FRONT = "A2402A", PALE = "F3F1EC";
const THEME = {
  name: "Cost-Aware Router",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: { dk1: INK, lt1: "FFFFFF", dk2: MUTED, lt2: PALE, accent1: LOCAL, accent2: FRONT, accent3: "8A857A",
            accent4: "4A6E8A", accent5: "B08A3E", accent6: "5C5C5C", hlink: "4A6E8A", folHlink: "6B665C" },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                       // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "A Cost-Aware Model Router";
pres.author = "Anirvan Ranjan, Huwaid Siddiqui, Sarthak Jakhmola, Kartiken Sharma";

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE", background: { color: INK },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 1.6, fontSize: 48, bold: true, align: "left",
                                color: "FFFFFF", fontFace: "Cambria", valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11.7, h: 2.6, fontSize: 18,
                                color: "D9D3C5", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION", background: { color: INK },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.2, fontSize: 40, bold: true, align: "left",
                                color: "FFFFFF", fontFace: "Cambria", valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 11.7, h: 1.6, fontSize: 18,
                                color: "D9D3C5", valign: "top", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: "8A857A", align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  margin: [0.5, 0.6, 0.7, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 32, bold: true, align: "left",
                                color: INK, fontFace: "Cambria", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "A cost-aware model router · UPES B.Tech CSE major project",
              options: { x: 0.6, y: 6.95, w: 8, h: 0.3, fontSize: 10, color: "8A857A", margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: "8A857A", align: "right" },
});

// ---------- helpers ----------
const T = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, fontFace: "Calibri", color: INK, ...o });
function content(title, section, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function bullets(s, items, o) {
  s.addText(items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, j) => ({ text: r.text, options: { ...(r.options || {}), bullet: j === 0 ? { indent: 18 } : undefined,
      breakLine: j === runs.length - 1 && i < items.length - 1 } }));
  }).flat(), { isTextBox: true, margin: 0, fontFace: "Calibri", fontSize: 18, color: INK, paraSpaceAfter: 10, valign: "top", ...o });
}
// stat callout: big number + label
function stat(s, x, y, w, num, label, color, name) {
  T(s, num, { x, y, w, h: 0.95, fontSize: 48, bold: true, fontFace: "Cambria", color: color || INK, objectName: name + " number" });
  T(s, label, { x, y: y + 1.0, w, h: 0.75, fontSize: 15, color: MUTED, valign: "top", objectName: name + " label" });
}
function table(s, rows, o) {
  const head = rows[0].map(h => ({ text: h, options: { bold: true, color: MUTED, fontSize: 13, border: [{ type: "none" }, { type: "none" }, { pt: 1, color: INK }, { type: "none" }] } }));
  const body = rows.slice(1).map((r, i) => r.map(c => {
    const cell = typeof c === "object" ? c : { text: String(c) };
    return { text: cell.text, options: { fontSize: 16, color: INK, ...(cell.options || {}),
      border: [{ type: "none" }, { type: "none" }, { pt: 0.75, color: RULE }, { type: "none" }] } };
  }));
  s.addTable([head, ...body], { x: 0.6, y: 1.5, w: 12.1, fontFace: "Calibri", margin: [4, 6, 4, 6], autoPage: false, ...o });
}
const b = (text, color) => ({ text, options: { bold: true, ...(color ? { color } : {}) } });
function box(s, x, y, w, h, fill, name) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || PALE }, line: { type: "none" }, objectName: name });
}
function arrow(s, x1, y1, x2, y2, name) {
  s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: MUTED, width: 1.5, endArrowType: "triangle" }, objectName: name });
}
const CHART_TEXT = { catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 13, valAxisLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 14, dataLabelColor: INK,
  titleFontFace: "+mn-lt", titleColor: MUTED, titleFontSize: 13, valGridLine: { color: RULE, size: 0.5 }, catGridLine: { style: "none" },
  showLegend: false };

// =====================================================================================
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Opening" });
  s.addText("A cost-aware model router", { placeholder: "title" });
  s.addText([
    { text: "Answer on the laptop when the small model is sure; pay for the frontier only when it is not — with a guarantee fixed before testing.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Anirvan Ranjan · Huwaid Siddiqui · Sarthak Jakhmola · Kartiken Sharma", options: { color: "FFFFFF", breakLine: true } },
    { text: "B.Tech (Hons.) CSE major project · School of Computer Science, UPES Dehradun" },
  ], { placeholder: "body" });
  s.addNotes("Ani opens. One sentence on the idea, then names. ~30 s.");
}
{
  const s = content("Most questions don't need the frontier model", "Opening",
    "Numbers: T27 headroom sweep, notes/t27_t28_locked_numbers.md (Qwen 0.8126, GPT-4 0.8874 on the 4,707 held-out Track B test questions). PGR definition from RouteLLM (Ong et al.).");
  stat(s, 0.6, 1.6, 3.6, "81.3%", "Qwen3.5-4B, 4-bit, on a laptop — $0 per question", LOCAL, "qwen");
  stat(s, 4.6, 1.6, 3.6, "88.7%", "GPT-4 — paid per token", FRONT, "gpt4");
  stat(s, 8.6, 1.6, 4.1, "0.075", "the whole quality gap the frontier adds, on 4,707 held-out questions", INK, "gap");
  box(s, 0.6, 4.2, 12.1, 2.25, PALE, "question box");
  T(s, [
    { text: "Our question", options: { bold: true, color: MUTED, fontSize: 13, breakLine: true } },
    { text: "How much frontier spend can be cut while keeping ≥ 95% of the frontier's quality advantage — and can that be guaranteed before deployment, not just observed afterwards?", options: { fontSize: 20, fontFace: "Cambria", breakLine: true } },
    { text: "Quality kept = PGR = (Q − Q_small) / (Q_frontier − Q_small)", options: { fontSize: 15, color: MUTED } },
  ], { x: 0.9, y: 4.4, w: 11.5, h: 1.9, valign: "top", paraSpaceAfter: 6 });
}
{
  const s = content("The answer, up front", "Opening",
    "Table 5.1, Ch. 5; notes/t27_t28_locked_numbers.md §3.1. 41.3% [39.0, 43.5] is M1 (API spend). Test PGR 102.8% [99.1, 107.0], escalation 75.9% [74.7, 77.1]. The 2.1% is the question-only router (query_f2) under the same bound.");
  stat(s, 0.6, 1.6, 3.8, "41.3%", "less API spend, certified\n95% CI 39.0–43.5%", LOCAL, "saving");
  stat(s, 4.75, 1.6, 3.8, "102.8%", "of GPT-4's quality advantage kept on the test split (target ≥ 95%)", INK, "pgr");
  stat(s, 8.9, 1.6, 3.8, "2.1%", "what routing on the question text alone could certify", FRONT, "query only");
  bullets(s, [
    [b("How: "), { text: "Qwen answers first; a small router reads how sure Qwen is (3 answers agree? option probability?) and escalates the rest." }],
    [b("Guarantee: "), { text: "the threshold τ* = 8.2187 was fixed on a calibration split with a seed-free WSR bound, before the test split was opened." }],
    [b("Live: "), { text: "the same router runs on a MacBook with Gemini 3 Flash as the frontier — 24/24 demo questions as planned." }],
  ], { x: 0.6, y: 4.3, w: 12.1, h: 2.3 });
}
{
  const s = content("Four lanes, one pipeline", "Opening", "Who presents what. Each lane hands over to the next.");
  const lanes = [
    ["A", "Anirvan Ranjan", "Data & system", "audit, re-grading, frozen splits · cost models · threshold code · Track B labels · demo router"],
    ["B", "Huwaid Siddiqui", "Features & router", "F0/F1/F2 features · two-head MLP · calibration · the cascade routers"],
    ["C", "Sarthak Jakhmola", "Guarantee & results", "Learn-then-Test · bootstrap undercoverage · seed-free WSR · results tables"],
    ["D", "Kartiken Sharma", "On the Mac", "4-bit Qwen · serving & cache · demo set · soak & memory"],
  ];
  lanes.forEach((l, i) => {
    const x = 0.6 + i * 3.1;
    box(s, x, 1.55, 2.85, 4.2, PALE, "lane " + l[0]);
    T(s, l[0], { x: x + 0.25, y: 1.75, w: 1, h: 0.9, fontSize: 44, bold: true, fontFace: "Cambria", color: INK });
    T(s, l[1], { x: x + 0.25, y: 2.75, w: 2.4, h: 0.4, fontSize: 17, bold: true });
    T(s, l[2], { x: x + 0.25, y: 3.15, w: 2.4, h: 0.35, fontSize: 14, color: MUTED });
    T(s, l[3], { x: x + 0.25, y: 3.7, w: 2.4, h: 2.5, fontSize: 14, valign: "top" });
    if (i < 3) arrow(s, x + 2.86, 4.0, x + 3.09, 4.0, "handover " + i);
  });
}

// =====================================================================================
pres.addSection({ title: "A · Data and system" });
{
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: "A · Data and system" });
  s.addText("A · Getting the labels right", { placeholder: "title" });
  s.addText("Anirvan Ranjan — a router can only learn what its labels say", { placeholder: "body" });
}
{
  const s = content("Read the responses, not the score column", "A · Data and system",
    "notes/routerbench_audit.md (items 1–4), D-05. RouterBench: 36,497 rows, 11 models; kept 34,778 questions in 7 datasets. Mistral-7B answers 'A' on 94% of MMLU and 100% of HellaSwag.");
  stat(s, 0.6, 1.6, 3.4, "34,778", "RouterBench questions kept, 7 datasets, 11 models", INK, "kept");
  table(s, [
    ["What we found by reading answers", "What we did"],
    [{ text: "Mistral-7B answers \"A\" on 94% of MMLU and 100% of HellaSwag" }, "Dropped it (D-05): a format failure, not a skill"],
    ["GSM8K scores take 5 values; GPT-4 never scores 1.0 (mean 0.66 vs ~0.9 real)", "Re-graded every answer"],
    ["GSM8K items come from its train split", "Gold answers taken from GSM8K train"],
    ["MMLU is the full test set despite the 'val' label", "Kept, documented"],
  ], { x: 4.5, y: 1.6, w: 8.2, colW: [4.5, 3.7] });
}
{
  const s = content("Re-grading, gold letters and leakage", "A · Data and system",
    "notes/gsm8k_grader_check.md (48/50 hand check), notes/chapter_A_facts.md A4 (gold letters, 0 conflicts), notes/dedup_report.md and D-11 (1,883 pairs, frozen splits). Raw BGE cosine chained all 895 MMLU moral-scenarios questions into one group because of their shared preamble; centring on the dataset/subject mean fixed it.");
  T(s, "GSM8K grader", { x: 0.6, y: 1.55, w: 3.8, h: 0.4, fontSize: 14, bold: true, color: MUTED });
  stat(s, 0.6, 1.95, 3.8, "48 / 50", "hand-checked grades correct (gate: 95%)", LOCAL, "grader");
  T(s, "Gold answer letters", { x: 4.75, y: 1.55, w: 3.8, h: 0.4, fontSize: 14, bold: true, color: MUTED });
  stat(s, 4.75, 1.95, 3.8, "0", "conflicts — voted from every model that was right (MMLU 95.5% covered)", INK, "letters");
  T(s, "Near-duplicates", { x: 8.9, y: 1.55, w: 3.8, h: 0.4, fontSize: 14, bold: true, color: MUTED });
  stat(s, 8.9, 1.95, 3.8, "1,883", "duplicate pairs merged before splitting", FRONT, "dups");
  box(s, 0.6, 4.25, 12.1, 2.2, PALE, "split box");
  T(s, [
    { text: "Raw embedding similarity chained all 895 MMLU moral-scenarios questions into one group (they share a preamble). ", options: {} },
    { text: "Subtracting the subject's mean embedding first fixed it.", options: { bold: true, breakLine: true } },
    { text: "Grouped 60 / 20 / 20 splits, frozen: train 20,867 · calibration 6,957 · test 6,954. A duplicate across calibration and test would make the guarantee optimistic.", options: { color: MUTED } },
  ], { x: 0.9, y: 4.45, w: 11.5, h: 1.85, fontSize: 17, valign: "top", paraSpaceAfter: 8 });
}
{
  const s = content("What a question costs: two cost models", "A · Data and system",
    "router/costs.py (lane A). M2 assumptions: $1,199 MacBook, 3 years, 8 h/day, 20 W, $0.12/kWh. Qwen speed 28.5 tok/s measured on the M2 (Ch. 5 §5.2). Batch API at 50% is the bar any real-time router must beat.");
  box(s, 0.6, 1.55, 5.85, 3.0, PALE, "M1 box");
  T(s, [{ text: "M1 · API spend", options: { bold: true, fontSize: 22, fontFace: "Cambria", breakLine: true } },
        { text: "Local answer: $0.", options: { breakLine: true } },
        { text: "Frontier answer: list price × tokens in and out.", options: { breakLine: true } },
        { text: "The headline 41.3% is M1.", options: { color: LOCAL, bold: true } }],
     { x: 0.9, y: 1.75, w: 5.3, h: 2.6, fontSize: 17, valign: "top", paraSpaceAfter: 8 });
  box(s, 6.85, 1.55, 5.85, 3.0, PALE, "M2 box");
  T(s, [{ text: "M2 · the laptop costs money too", options: { bold: true, fontSize: 22, fontFace: "Cambria", breakLine: true } },
        { text: "$1,199 MacBook over 3 years at 8 h/day, plus 20 W at $0.12/kWh, charged per second of generation.", options: { breakLine: true } },
        { text: "Cascade under M2: 28.4% (all 3 local answers billed).", options: { color: LOCAL, bold: true } }],
     { x: 7.15, y: 1.75, w: 5.3, h: 2.6, fontSize: 17, valign: "top", paraSpaceAfter: 8 });
  bullets(s, [
    [b("Crossover speed: "), { text: "below it, the laptop costs more than the API. Qwen on the M2 runs at 28.5 tokens/s — well above it." }],
    [b("Batch API baseline: "), { text: "always-frontier at 50% off is the number a real-time router has to beat." }],
  ], { x: 0.6, y: 4.85, w: 12.1, h: 1.7 });
}
{
  const s = content("Track B: our own labels, from the model we actually serve", "A · Data and system",
    "artifacts/trackb/gpu_manifest.json: 7,450 GSM8K x 3 samples + 16,130 MCQ on one H100 80GB in 327.5 s. Release trackb-6265adf7ffa4a602. Same prompt builder and sampling as the Mac (T 0.7, top-p 0.8).");
  const steps = [["RouterBench", "no Qwen, no log-probabilities"], ["H100, bf16 Qwen", "7,450 GSM8K × 3 answers\n16,130 multiple-choice"], ["Release", "trackb-6265adf7…\n327.5 s of GPU time"], ["Router signals", "agreement · option probability\nlog-probabilities · length"]];
  steps.forEach((st, i) => {
    const x = 0.6 + i * 3.15;
    box(s, x, 1.8, 2.75, 2.1, i === 3 ? "E6EFE8" : PALE, "step " + i);
    T(s, st[0], { x: x + 0.2, y: 1.95, w: 2.4, h: 0.5, fontSize: 18, bold: true, fontFace: "Cambria" });
    T(s, st[1], { x: x + 0.2, y: 2.5, w: 2.4, h: 1.3, fontSize: 14, color: MUTED, valign: "top" });
    if (i < 3) arrow(s, x + 2.78, 2.85, x + 3.12, 2.85, "flow " + i);
  });
  bullets(s, [
    "Same prompts, same sampling (3 answers at temperature 0.7, top-p 0.8) and the same option-letter pass as the Mac.",
    "So the router is trained on exactly the kind of signal it reads live.",
  ], { x: 0.6, y: 4.4, w: 12.1, h: 1.8 });
}

// =====================================================================================
pres.addSection({ title: "B · Features and router" });
{
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: "B · Features and router" });
  s.addText("B · What can a router see?", { placeholder: "title" });
  s.addText("Huwaid Siddiqui — from the question text to the small model's own confidence", { placeholder: "body" });
}
{
  const s = content("Routing on the question text alone", "B · Features and router",
    "thesis/04_router.md. F0: 23 hand-made features (<0.5 ms). F1: TF-IDF word 1-2 + char 3-5 grams, fitted on train only. F2: bge-base-en-v1.5, 768-d. Two-head MLP on [F2 || F0] (791-d), shared trunk 256, GELU; loss = BCE(S) + BCE(L) + 0.5 * MSE on the gap. Isotonic calibration on cal cuts ECE by 40.0-56.7%.");
  const tiers = [["F0", "23 surface features", "length, digits, LaTeX, option count, “prove”…"], ["F1", "100,000 n-grams", "TF-IDF, fitted on train only"], ["F2", "768-d embedding", "bge-base-en-v1.5"]];
  tiers.forEach((t, i) => {
    const y = 1.6 + i * 1.25;
    box(s, 0.6, y, 5.6, 1.05, PALE, "tier " + t[0]);
    T(s, t[0], { x: 0.85, y: y + 0.15, w: 0.9, h: 0.75, fontSize: 30, bold: true, fontFace: "Cambria" });
    T(s, [{ text: t[1], options: { bold: true, breakLine: true } }, { text: t[2], options: { color: MUTED, fontSize: 14 } }], { x: 1.9, y: y + 0.12, w: 4.2, h: 0.85, fontSize: 16, valign: "middle" });
  });
  arrow(s, 6.3, 3.05, 6.9, 3.05, "into mlp");
  box(s, 7.0, 1.6, 5.7, 3.55, "FFFFFF", "mlp frame");
  s.addShape(pres.shapes.RECTANGLE, { x: 7.0, y: 1.6, w: 5.7, h: 3.55, fill: { type: "none" }, line: { color: INK, width: 1 }, objectName: "mlp border" });
  T(s, [{ text: "Two-head MLP", options: { bold: true, fontSize: 20, fontFace: "Cambria", breakLine: true } },
        { text: "shared trunk → ŝ_small and ŝ_frontier", options: { breakLine: true } },
        { text: "loss = BCE + BCE + 0.5 × error on the gap", options: { breakLine: true } },
        { text: "isotonic calibration: ECE −40% to −57%", options: { color: MUTED } }],
     { x: 7.3, y: 1.8, w: 5.2, h: 3.2, fontSize: 16, valign: "top", paraSpaceAfter: 8 });
  T(s, "Anti-leakage: the query router sees only the question text — no dataset names, no labels.", { x: 0.6, y: 5.55, w: 12.1, h: 0.8, fontSize: 17, color: MUTED });
}
{
  const s = content("Question text hits a ceiling; Qwen's own answers don't", "B · Features and router",
    "Certified M1 savings, seed-free WSR, Table 5.1: query_f2 2.1% [1.5, 2.7]; cascade 1 answer (sample-0 quality) 29.5% [27.8, 31.5]; cascade 3 answers 41.3% [39.0, 43.5]. Mixtral pair two-head MLP: 9.4% [7.8, 10.9]. The gap Qwen leaves is only 0.075, so a few misrouted questions break the 95% floor.");
  s.addChart(pres.charts.BAR, [{ name: "Certified saving", labels: ["Question text only", "Cascade, 1 answer", "Cascade, 3 answers"], values: [2.1, 29.5, 41.3] }], {
    x: 0.6, y: 1.5, w: 7.4, h: 4.9, barDir: "bar", chartColors: ["B8B2A6", "8FAF97", LOCAL], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: '0.0"%"', valAxisMaxVal: 50, valAxisMinVal: 0, valAxisLabelFormatCode: '0"%"', catAxisOrientation: "maxMin",
    showTitle: true, title: "Certified API saving, Qwen3.5-4B vs GPT-4 (test, n = 4,707)", ...CHART_TEXT, valAxisHidden: false, barGapWidthPct: 60,
  });
  bullets(s, [
    [b("Why text fails: "), { text: "two maths questions can read the same and one still drops a sign in step 3." }],
    [b("Small gap: "), { text: "Qwen leaves only 0.075 of accuracy on the table, so a few misroutes break the 95% floor." }],
    [b("Mixtral pair: "), { text: "the same question-only MLP certifies 9.4% — a bigger gap leaves more to route." }],
  ], { x: 8.4, y: 1.7, w: 4.3, h: 4.6, fontSize: 16 });
}
{
  const s = content("The cascade: let Qwen answer, then ask how sure it is", "B · Features and router",
    "router/signals.py; T61 cascade_3sample (HistGradientBoostingRegressor, 300 iterations, lr 0.05, 31 leaves) on 7 confidence signals + 4 question-type one-hots. Adding F2 embeddings to the cascade cut certified savings to 1.1-3.4% (trackb_cascade_report.md).");
  // two lanes of the flow
  const rows = [["Multiple choice", "one forward pass over the option letters", "p(chosen), margin"], ["Arithmetic", "3 answers at temperature 0.7", "agreement, # distinct, log-probs, length"]];
  rows.forEach((r, i) => {
    const y = 1.65 + i * 1.75;
    box(s, 0.6, y, 2.6, 1.35, PALE, "kind " + i);
    T(s, r[0], { x: 0.8, y: y + 0.1, w: 2.3, h: 1.15, fontSize: 18, bold: true, fontFace: "Cambria", valign: "middle" });
    arrow(s, 3.25, y + 0.67, 3.75, y + 0.67, "a" + i);
    box(s, 3.8, y, 3.6, 1.35, PALE, "qwen " + i);
    T(s, [{ text: "Qwen on the Mac", options: { bold: true, breakLine: true } }, { text: r[1], options: { color: MUTED, fontSize: 14 } }], { x: 4.0, y: y + 0.1, w: 3.3, h: 1.15, fontSize: 16, valign: "middle" });
    arrow(s, 7.45, y + 0.67, 7.95, y + 0.67, "b" + i);
    box(s, 8.0, y, 2.4, 1.35, PALE, "sig " + i);
    T(s, r[2], { x: 8.15, y: y + 0.1, w: 2.15, h: 1.15, fontSize: 14, valign: "middle" });
  });
  arrow(s, 10.45, 2.32, 10.95, 2.9, "c0"); arrow(s, 10.45, 4.07, 10.95, 3.5, "c1");
  s.addShape(pres.shapes.RECTANGLE, { x: 11.0, y: 2.45, w: 1.7, h: 1.5, fill: { color: INK }, line: { type: "none" }, objectName: "router box" });
  T(s, [{ text: "Router", options: { bold: true, breakLine: true } }, { text: "boosted trees", options: { fontSize: 12 } }], { x: 11.0, y: 2.45, w: 1.7, h: 1.5, color: "FFFFFF", align: "center", valign: "middle", fontSize: 17 });
  bullets(s, [
    [b("Missing values are native: "), { text: "maths has no option probability and MCQ has no agreement — the trees route around the gaps." }],
    [b("Counter-intuitive: "), { text: "adding the 768-d embedding to the cascade dropped certified savings to 1.1–3.4%; the trees fit topic noise." }],
  ], { x: 0.6, y: 5.25, w: 12.1, h: 1.5, fontSize: 16 });
}
{
  const s = content("The demo serves the certified router (D-15)", "B · Features and router",
    "notes/d15_demo_router.md, DECISIONS.md D-15 (agreed by all four lanes). D-14 read Qwen's answer text and escalated every easy question Qwen got right on demo runs 1-2. Test frontier calls, seed-free: D-14 85.2% / PGR 100.5%; D-15 65.8% / PGR 100.6%. Free-form questions are answered the same way but labelled uncertified.");
  table(s, [
    ["Demo router (seed-free bound, test split)", "Frontier calls", "PGR", "Easy demo questions kept local"],
    ["D-14: features from Qwen's answer text", { text: "85.2%", options: { color: FRONT } }, "100.5%", { text: "none (runs 1–2)", options: { color: FRONT } }],
    [{ text: "D-15: the research router, signals computed live", options: { bold: true } }, { text: "65.8%", options: { bold: true, color: LOCAL } }, { text: "100.6%", options: { bold: true } }, { text: "9 / 9", options: { bold: true, color: LOCAL } }],
  ], { y: 1.6, colW: [5.6, 2.0, 1.6, 2.9] });
  bullets(s, [
    "One implementation of the signals (router/signals.py) for training and serving — that is what makes the live threshold the certified one.",
    "Anything that isn't multiple choice or arithmetic is answered the same way but labelled \"not certified\" on the page.",
    "Cost: ~6 s on the M2 for three short arithmetic answers.",
  ], { x: 0.6, y: 3.6, w: 12.1, h: 2.8 });
}

// =====================================================================================
pres.addSection({ title: "C · Guarantee and results" });
{
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: "C · Guarantee and results" });
  s.addText("C · From a good curve to a guarantee", { placeholder: "title" });
  s.addText("Sarthak Jakhmola — which threshold, and how sure are we?", { placeholder: "body" });
}
{
  const s = content("Reading the answer off the test curve is not allowed", "C · Guarantee and results",
    "Table 5.1 rows for cascade_3sample: post-hoc test curve 78.9% (uncertified, CPT 39.5%); naive tau on cal 77.0% at test PGR 97.2% [91.9, 103.2] - the lower end is below 95%; seed-free WSR 41.3% [39.0, 43.5] at PGR 102.8% [99.1, 107.0].");
  s.addChart(pres.charts.BAR, [{ name: "Saving", labels: ["Best point on the test curve", "Threshold picked naively on cal", "Seed-free WSR (certified)"], values: [78.9, 77.0, 41.3] }], {
    x: 0.6, y: 1.5, w: 7.4, h: 4.9, barDir: "bar", chartColors: ["B8B2A6", "B8B2A6", LOCAL], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: '0.0"%"', valAxisMaxVal: 100, valAxisMinVal: 0, valAxisLabelFormatCode: '0"%"', catAxisOrientation: "maxMin",
    showTitle: true, title: "API saving, 3-answer cascade (same router, different threshold choice)", ...CHART_TEXT, barGapWidthPct: 60,
  });
  bullets(s, [
    [b("Test curve: "), { text: "the threshold has seen the data it is judged on." }],
    [b("Naive on cal: "), { text: "test PGR 97.2%, but its interval reaches 91.9% — below the promise." }],
    [b("Certified: "), { text: "PGR ≥ 95% with 95% confidence, τ* fixed before the test split was opened." }],
    [{ text: "The gap is the price of a guarantee.", options: { italic: true, color: MUTED } }],
  ], { x: 8.4, y: 1.7, w: 4.3, h: 4.6, fontSize: 16 });
}
{
  const s = content("Learn-then-Test, and why the bootstrap fails here", "C · Guarantee and results",
    "Ch. 5 §5.4-5.5. Paired margin e_i(tau) = s_route - [(1-rho) s_S + rho s_L], rho = 0.95. Fixed-sequence testing from the safest tau outward, stop at first failure: no Bonferroni. Bootstrap undercoverage: Monte Carlo, n_cal = 2,000, 15.2% miss rate vs 5% nominal.");
  box(s, 0.6, 1.55, 6.0, 4.9, PALE, "ltt box");
  T(s, [{ text: "One test per threshold", options: { bold: true, fontSize: 20, fontFace: "Cambria", breakLine: true } },
        { text: "Margin per question:", options: { color: MUTED, breakLine: true } },
        { text: "e(τ) = s_routed − [0.05 · s_small + 0.95 · s_frontier]", options: { fontFace: "Cambria", breakLine: true } },
        { text: "PGR ≥ 95%  ⇔  mean margin ≥ 0", options: { bold: true, breakLine: true } },
        { text: "Test thresholds in a fixed order, from the safest outward; stop at the first that fails → no Bonferroni correction.", options: {} }],
     { x: 0.9, y: 1.75, w: 5.4, h: 4.5, fontSize: 17, valign: "top", paraSpaceAfter: 10 });
  T(s, "Which bound?", { x: 7.0, y: 1.55, w: 5.7, h: 0.45, fontSize: 14, bold: true, color: MUTED });
  stat(s, 7.0, 1.95, 5.7, "15.2%", "how often the nominal 95% bootstrap bound missed (allowed: 5%) — the margin is 0 on 70–80% of questions", FRONT, "bootstrap");
  bullets(s, [
    [b("Hoeffding: "), { text: "valid, but ignores the variance — throws most savings away." }],
    [b("WSR betting martingale: "), { text: "valid at any sample size and uses the variance." }],
  ], { x: 7.0, y: 4.4, w: 5.7, h: 2.0, fontSize: 16 });
}
{
  const s = content("WSR depended on the shuffle; averaging fixed it", "C · Guarantee and results",
    "notes/t60_certification_report.md (T60). Seeds 0-19: tau* 8.11 to 16.72, test savings 40.8% to 74.9%. Seed-free: mean of terminal e-values over M = 50 orders is an e-value; p = min(1, 1/mean) by Markov. tau* = 8.2187. scripts/17_t60_seed_free_wsr.py, ported to router/threshold.py method='wsr_avg' (identical p-values).");
  // range bar: 30% .. 80% scale
  const x0 = 0.9, x1 = 12.4, lo = 30, hi = 80;
  const X = (v) => x0 + (x1 - x0) * (v - lo) / (hi - lo);
  T(s, "Certified test saving from plain WSR, same data, only the random seed changed (seeds 0–19)", { x: 0.6, y: 1.5, w: 12.1, h: 0.4, fontSize: 15, color: MUTED });
  s.addShape(pres.shapes.LINE, { x: x0, y: 3.0, w: x1 - x0, h: 0, line: { color: RULE, width: 1 }, objectName: "axis" });
  [30, 40, 50, 60, 70, 80].forEach(v => {
    s.addShape(pres.shapes.LINE, { x: X(v), y: 2.92, w: 0, h: 0.16, line: { color: MUTED, width: 1 }, objectName: "tick " + v });
    T(s, v + "%", { x: X(v) - 0.4, y: 3.3, w: 0.8, h: 0.3, fontSize: 12, color: MUTED, align: "center" });
  });
  s.addShape(pres.shapes.RECTANGLE, { x: X(40.8), y: 2.8, w: X(74.9) - X(40.8), h: 0.4, fill: { color: "E8C9C0" }, line: { type: "none" }, objectName: "seed range" });
  T(s, "40.8%", { x: X(40.8) - 0.5, y: 2.3, w: 1.0, h: 0.4, fontSize: 15, bold: true, color: FRONT, align: "center" });
  T(s, "74.9%", { x: X(74.9) - 0.5, y: 2.3, w: 1.0, h: 0.4, fontSize: 15, bold: true, color: FRONT, align: "center" });
  s.addShape(pres.shapes.RECTANGLE, { x: X(41.3) - 0.025, y: 2.6, w: 0.05, h: 0.95, fill: { color: LOCAL }, line: { type: "none" }, objectName: "seed-free mark" });
  T(s, "seed-free: 41.3%", { x: X(41.3) + 0.1, y: 3.6, w: 2.5, h: 0.4, fontSize: 15, bold: true, color: LOCAL });
  box(s, 0.6, 4.3, 12.1, 2.15, PALE, "fix box");
  T(s, [
    { text: "Each seed gives a valid certificate — but if you can choose the seed, you can choose your result.", options: { bold: true, breakLine: true } },
    { text: "For one ordering, the final wealth is an e-value. The average of 50 orderings is still an e-value, so p = 1 / average is valid (Markov). One threshold, no seed: τ* = 8.2187.", options: { color: MUTED } },
  ], { x: 0.9, y: 4.5, w: 11.5, h: 1.8, fontSize: 17, valign: "top", paraSpaceAfter: 8 });
}
{
  const s = content("Headline results (Table 5.1)", "C · Guarantee and results",
    "Ch. 5 Table 5.1 / notes/t27_t28_locked_numbers.md. All rows: rule v2, seed-free WSR, M = 50, test n = 4,707. M2 for the 3-answer cascade bills all three local answers (36.5% if only one is billed - the earlier convention).");
  table(s, [
    ["Qwen3.5-4B vs GPT-4 (test n = 4,707)", "τ*", "Frontier calls", "Test PGR", "M1 saving [95% CI]", "M2 saving"],
    ["Question text only (query_f2)", "9.8934", "98.1%", "98.6%", "2.1% [1.5, 2.7]", "2.0%"],
    ["Cascade, 1 answer", "13.6310", "67.9%", "98.4%", "29.5% [27.8, 31.5]", "24.7%"],
    [{ text: "Cascade, 3 answers (the demo router)", options: { bold: true } }, { text: "8.2187", options: { bold: true } }, { text: "75.9%", options: { bold: true } },
     { text: "102.8%", options: { bold: true } }, { text: "41.3% [39.0, 43.5]", options: { bold: true, color: LOCAL } }, { text: "28.4%", options: { bold: true } }],
  ], { y: 1.6, colW: [4.3, 1.2, 1.6, 1.4, 2.3, 1.3] });
  bullets(s, [
    "PGR above 100%: on some questions Qwen is right and GPT-4 is wrong; keeping those local beats always-frontier.",
    "M2 charges the laptop for all three local answers, even on questions that then go to the frontier.",
    "Mixtral development pair, question-only MLP: 9.4% [7.8, 10.9] at PGR 96.3%.",
  ], { x: 0.6, y: 4.1, w: 12.1, h: 2.5, fontSize: 17 });
}
{
  const s = content("Does it travel?", "C · Guarantee and results",
    "T27 headroom sweep (Table 5.2): question-only 12.0-21.1% uncertified on Mixtral, Yi-34B and Qwen. T26 LOSO on MMLU, Mixtral pair (Table 5.3): fold mean 14.6% -> 10.3%, per fold 29.7% (best) to 4.0% (worst). T44 replay: notes/replay_report_d15.md.");
  const cols = [
    ["Other model pairs", "12–21%", "question-only savings on Mixtral, Yi-34B and Qwen alike (uncertified test curve)", INK],
    ["Unseen MMLU subjects", "10.3%", "question-only saving when the subject is held out (14.6% when seen); per fold 29.7% down to 4.0%", FRONT],
    ["Live server vs offline", "200 / 200", "identical decisions on 200 test questions; router p99 5.4 ms", LOCAL],
  ];
  cols.forEach((c, i) => {
    const x = 0.6 + i * 4.1;
    T(s, c[0], { x, y: 1.6, w: 3.8, h: 0.4, fontSize: 15, bold: true, color: MUTED });
    stat(s, x, 2.05, 3.8, c[1], c[2], c[3], "travel " + i);
  });
  box(s, 0.6, 4.55, 12.1, 1.85, PALE, "claim box");
  T(s, [{ text: "What we claim: ", options: { bold: true } }, { text: "on questions like the test distribution, ≥ 95% of GPT-4's advantage with 95% confidence at ~41% less API spend. ", options: {} },
        { text: "Not covered: ", options: { bold: true, color: FRONT } }, { text: "free-form chat, other model pairs, or a refit without re-certifying.", options: {} }],
     { x: 0.9, y: 4.75, w: 11.5, h: 1.5, fontSize: 18, valign: "top" });
}

// =====================================================================================
pres.addSection({ title: "D · On the Mac" });
{
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: "D · On the Mac" });
  s.addText("D · The same router, live on a MacBook", { placeholder: "title" });
  s.addText("Kartiken Sharma — keeping the certificate true outside the notebook", { placeholder: "body" });
}
{
  const s = content("4-bit Qwen on a 16 GB M2", "D · On the Mac",
    "notes/m2_checks.md (T59): first 100 GSM8K + 100 MMLU test questions, Mac 4-bit vs H100 bf16; with n = 100 the interval is about +/-7 points. Speed 2026-11-15. Batch 4 is 1.4x faster (notes/batch_speedup.json) but loses mean_logprob, so max_batch = 1.");
  stat(s, 0.6, 1.6, 3.6, "~3 GB", "4-bit Qwen3.5-4B (MLX) of a 16 GB machine", INK, "mem");
  stat(s, 0.6, 3.6, 3.6, "29 tok/s", "generation speed; a 110-token answer in ~5 s", INK, "speed");
  table(s, [
    ["Accuracy, 100 questions each", "Mac 4-bit", "H100 bf16", "Difference"],
    ["GSM8K", "0.840", "0.860", "−2 pts"],
    ["MMLU", "0.820", "0.790", "+3 pts"],
  ], { x: 4.7, y: 1.6, w: 8.0, colW: [3.2, 1.6, 1.6, 1.6] });
  bullets(s, [
    "Within ±7 points at n = 100: no measurable quantisation gap.",
    "Batching is 1.4× faster but drops the log-probabilities the router reads → batch size stays 1.",
  ], { x: 4.7, y: 3.7, w: 8.0, h: 2.5, fontSize: 17 });
}
{
  const s = content("One server, one lock, one cache", "D · On the Mac",
    "router/serve.py + scripts/serve_demo.py. SSE events: status, route, token, done. SQLite cache keyed by SHA-256 of (model, prompt, params, sample index); prompt templates are part of the key. --offline swaps in a same-name stand-in client so keys do not change.");
  const flow = [["Browser", "/chat/stream (SSE)"], ["Qwen answers", "option pass or 3 answers"], ["Router", "score vs τ* = 8.2187"], ["Stream", "local text, or Gemini"]];
  flow.forEach((f, i) => {
    const x = 0.6 + i * 3.15;
    box(s, x, 1.7, 2.75, 1.5, i === 2 ? "E6EFE8" : PALE, "srv " + i);
    T(s, [{ text: f[0], options: { bold: true, fontFace: "Cambria", fontSize: 18, breakLine: true } }, { text: f[1], options: { color: MUTED, fontSize: 14 } }], { x: x + 0.2, y: 1.8, w: 2.4, h: 1.3, valign: "middle" });
    if (i < 3) arrow(s, x + 2.78, 2.45, x + 3.12, 2.45, "srvflow " + i);
  });
  bullets(s, [
    [b("Cache: "), { text: "SQLite keyed by model, prompt, sampling settings and sample number — a repeat question costs nothing, and editing a prompt template invalidates exactly what it affects." }],
    [b("Offline mode: "), { text: "every demo answer is cached, so the demo runs with Wi-Fi off." }],
    [b("Money: "), { text: "frontier priced at list rates even on Gemini's free tier; local answers count $0 (M1)." }],
  ], { x: 0.6, y: 3.65, w: 12.1, h: 2.9, fontSize: 17 });
}
{
  const s = content("Checked before every run", "D · On the Mac",
    "scripts/20_demo_preflight.py on the M2 (2026-11-24): scikit-learn 1.8.0, bundle cascade_t61_3sample tau* 8.2187, MLX cache cap 2 GB, 24/24 demo questions offline. Replay: notes/replay_report_d15.md (200/200, p50 3.57 ms, p99 5.43 ms).");
  stat(s, 0.6, 1.6, 3.8, "24 / 24", "demo questions route as planned: 9 easy kept local, 8 hard escalated, 7 free-form/chat", LOCAL, "preflight");
  stat(s, 4.75, 1.6, 3.8, "200 / 200", "server decisions identical to the offline evaluation", INK, "replay");
  stat(s, 8.9, 1.6, 3.8, "5.4 ms", "router decision time, 99th percentile (target 15 ms)", INK, "latency");
  bullets(s, [
    [b("Version pin: "), { text: "refitting on scikit-learn 1.9.1 gives τ* = 9.657 — a different model. The preflight refuses anything but 1.8.0." }],
    [b("Memory cap: "), { text: "MLX's GPU pool reached 11 GB of 16 GB in a one-hour soak; it is now capped at 2 GB." }],
  ], { x: 0.6, y: 4.35, w: 12.1, h: 2.2, fontSize: 17 });
}
{
  const s = content("Live demo", "D · On the Mac",
    "Run order from notes/demo_video_script.md. Start with: python scripts/serve_demo.py --offline. If anything fails live, switch to the recorded backup video. 64.2% / 63.3% savings on demo traffic are observed, not certified - quote 41.3%.");
  const steps = [
    ["1", "Easy arithmetic", "3 / 3 answers agree → answered on the Mac, $0", LOCAL],
    ["2", "Hard multiple choice", "option probabilities spread → sent to Gemini", FRONT],
    ["3", "Free-form chat", "answered, but flagged “not certified”", MUTED],
    ["4", "Savings column", "this session vs always-frontier — observed, not the certified 41.3%", INK],
  ];
  steps.forEach((st, i) => {
    const y = 1.6 + i * 1.2;
    T(s, st[0], { x: 0.6, y, w: 0.7, h: 0.9, fontSize: 40, bold: true, fontFace: "Cambria", color: st[3] });
    T(s, [{ text: st[1], options: { bold: true, fontSize: 20, breakLine: true } }, { text: st[2], options: { color: MUTED, fontSize: 16 } }], { x: 1.4, y: y + 0.02, w: 11.2, h: 0.95, valign: "middle" });
  });
}

// =====================================================================================
pres.addSection({ title: "Close" });
{
  const s = content("Limitations and what is still open", "Close",
    "Ch. 7. Gemini key is on the free tier (20/day, then 5/min): the soak test is labelled throttled; the demo runs from cache. GSM8K items are from its train split, so some models may have seen them.");
  bullets(s, [
    [b("Labels: "), { text: "one stored RouterBench answer per model; GSM8K items come from its train split." }],
    [b("Scope of the guarantee: "), { text: "four benchmark question types; free-form chat is labelled, not certified." }],
    [b("One pair: "), { text: "Qwen3.5-4B vs GPT-4 labels; another pair needs its own calibration run." }],
    [b("Free tier: "), { text: "the frontier key is rate-limited, so the one-hour soak is marked throttled and the demo runs from cache." }],
    [b("Latency: "), { text: "three local answers cost ~6 s on maths questions — the price of the signal." }],
  ], { x: 0.6, y: 1.6, w: 12.1, h: 4.9 });
}
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Close" });
  s.addText("41.3% less API spend, with a guarantee", { placeholder: "title" });
  s.addText([
    { text: "Looking at the small model's own answers — not the question — is what makes routing work when the models are close.", options: { breakLine: true } },
    { text: "The guarantee was fixed before testing, and the same router runs live on a laptop.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Thank you — questions welcome.", options: { color: "FFFFFF" } },
  ], { placeholder: "body" });
  s.addNotes("Numbers not to mix up: 41.3% certified (test split) · 37.3% replay of 200 questions · 63-64% observed on demo traffic · 78.9% post-hoc test curve (uncertified).");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
