// Build report/FloodWing_Report.docx from results/ (summary.json, tables, figures).
// Usage: node tools/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, PageBreak, Footer, Header, PageNumber,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const RES = path.join(ROOT, "results");
const FIG = path.join(RES, "figures");
const S = JSON.parse(fs.readFileSync(path.join(RES, "summary.json"), "utf8"));

// ---------- helpers ----------
const FONT = "Arial";
const BLUE = "0B5394";
const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;
const num = (v, d = 3) => Number(v).toFixed(d);

function readCsv(file) {
  const [head, ...rows] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  return rows.map((r) => {
    const v = r.split(",");
    return Object.fromEntries(cols.map((c, i) => [c, v[i]]));
  });
}

function runs(text, opts = {}) {
  // **bold** segments inside a string
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((seg) =>
    seg.startsWith("**") ? new TextRun({ text: seg.slice(2, -2), bold: true, font: FONT, ...opts })
                         : new TextRun({ text: seg, font: FONT, ...opts }));
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 }, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(t), spacing: { after: 60 } });
const numbered = (t) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, children: runs(t), spacing: { after: 60 } });

// Equation line: _x / _{...} subscripts, ^x / ^{...} superscripts, Cambria Math, centred.
function eq(src) {
  const out = [];
  let i = 0, buf = "";
  const flush = () => { if (buf) { out.push(new TextRun({ text: buf, font: "Cambria Math", size: 23 })); buf = ""; } };
  while (i < src.length) {
    const ch = src[i];
    if ((ch === "_" || ch === "^") && i + 1 < src.length) {
      flush();
      let body;
      if (src[i + 1] === "{") { const j = src.indexOf("}", i + 2); body = src.slice(i + 2, j); i = j + 1; }
      else { body = src[i + 1]; i += 2; }
      out.push(new TextRun({ text: body, font: "Cambria Math", size: 23, subScript: ch === "_", superScript: ch === "^" }));
    } else { buf += ch; i += 1; }
  }
  flush();
  return new Paragraph({ children: out, alignment: AlignmentType.CENTER, spacing: { before: 80, after: 140 } });
}

function figure(file, caption, widthIn = 6.3) {
  const p = path.join(FIG, file);
  if (!fs.existsSync(p)) return [P(`[figure ${file} not generated in this run]`)];
  const buf = fs.readFileSync(p);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); // PNG IHDR
  const width = Math.round(widthIn * 96), height = Math.round(width * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width, height },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: "555555", font: FONT })] }),
  ];
}

const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
function table(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, isHead, w) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER },
    shading: isHead ? { type: ShadingType.CLEAR, color: "auto", fill: "DCE6F1" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ children: [new TextRun({ text: String(text), bold: isHead, size: 18, font: FONT })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, true, widths[i])) }),
           ...rows.map((r) => new TableRow({ children: r.map((v, i) => cell(v, false, widths[i])) }))],
  });
}
const spacer = () => new Paragraph({ children: [], spacing: { after: 120 } });

// ---------- numbers ----------
const summary = readCsv(path.join(RES, "tables", "summary.csv"));
const byM = Object.fromEntries(summary.map((r) => [r.method, r]));
const q = byM["QAOA"], g = byM["Greedy"];
const nInst = S.evaluation.n_instances;
const wil = S.evaluation.wilcoxon_qaoa_vs_greedy;
const cnn = S.cnn;
const demo = S.demo;
const B = S.bonus;
const ws = B.warm_start;
const xy = Object.fromEntries((B.xy_mixer || []).map((r) => [r.mixer, r]));
const depth = B.depth || [];
const pen = B.penalty || [];
const noise = B.noise || [];
const opts = Object.fromEntries((B.optimizers || []).map((r) => [r.optimizer, r]));
const dataSrc = S.data.sources;
const synthetic = dataSrc.includes("synthetic");
const hitQ = Number(q.tie_vs_exact), hitG = Number(g.tie_vs_exact);
const noise1 = noise.find((r) => r.level === 1), noise4 = noise.find((r) => r.level === 4), noise0 = noise.find((r) => r.level === 0);

// ---------- content ----------
const children = [];

// Title page
children.push(
  new Paragraph({ spacing: { before: 2400, after: 240 }, alignment: AlignmentType.LEFT,
    children: [new TextRun({ text: "FloodWing", bold: true, size: 64, color: BLUE, font: FONT })] }),
  new Paragraph({ spacing: { after: 240 },
    children: [new TextRun({ text: "Hybrid CNN + agent + QAOA drone placement for flood search-and-rescue", size: 32, font: FONT })] }),
  new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: BLUE, space: 4 } }, spacing: { after: 360 }, children: [] }),
  P("Qiskit Fall Fest 2026 | Domain: UAVs & Robotics (Disaster Response angle)"),
  P("Team: [add names]"),
  P("Submission deadline: 12 November 2026, 11:59 pm"),
  P("Repository: [add GitHub link] | Video: [add link]"),
  new Paragraph({ children: [new PageBreak()] }),
);

// 1 Summary
children.push(
  H1("1. Summary"),
  P(`FloodWing decides where a small drone fleet should hover after a flood. It picks 4 of 12 grid cells to maximise survivor coverage, avoid overlapping footprints and keep drones in radio range. On ${nInst} test instances, QAOA (p = 2) reached the exact optimum on ${hitQ} and averaged a ${pct(Number(q.gap_mean), 2)} optimality gap. Greedy averaged ${pct(Number(g.gap_mean), 2)} and reached the optimum on ${hitG}.`),
  P("A CNN predicts a survivor-likelihood heatmap from flood imagery features. A rule-based agent updates that heatmap from plain-text field reports. The placement is formulated as a QUBO and solved with Qiskit QAOA, then compared with exact brute force, greedy, simulated annealing and random placements."),
  P(`**No quantum speedup is claimed.** With 12 variables an exact solver finishes in under a millisecond. The difference between QAOA and greedy is not statistically significant (Wilcoxon p = ${wil.p_value == null ? "n/a" : num(wil.p_value, 2)}). The contribution is a correct, reproducible hybrid formulation and an honest measurement of where QAOA helps and where it does not.`),
);

// 2 Problem
children.push(
  H1("2. Problem and who it serves"),
  P("After a flood, rescue coordinators have a few drones and a large area. Each drone hovers over one grid cell and sees a limited footprint. The coordinator must decide where to place them so that:"),
  bullet("the chance of finding survivors is as high as possible,"),
  bullet("drones do not waste coverage by overlapping each other, and"),
  bullet("drones stay within radio range of each other so a relay chain reaches the base station."),
  P("This is a coverage-versus-connectivity trade-off with limited resources, not a path-finding problem. It serves emergency response teams, NGOs and disaster-management agencies that must decide quickly with incomplete information, and must re-decide as field reports arrive."),
);

// 3 Architecture
children.push(
  H1("3. Solution and architecture"),
  P("FloodWing is a three-layer hybrid pipeline. The CNN and the agent prepare the problem; the quantum solver produces the final placement."),
  ...figure("18_architecture.png", "Figure 1. FloodWing architecture. QAOA (highlighted) chooses the placement; classical baselines grade it."),
  table(["Layer", "Input", "Output", "Module"], [
    ["Deep learning", "Flood chip features (64 x 64 x 4)", "Likelihood w_i per cell", "heatmap.py"],
    ["Agent", "Plain-text field report", "Boost / block / reduce updates", "agent.py"],
    ["QUBO builder", "w, blocked cells, grid geometry", "QuadraticProgram, Ising operator", "qubo.py"],
    ["Quantum solver", "Ising operator (12 qubits)", "Drone placement + probabilities", "solvers.py"],
    ["Baselines", "Same QUBO", "Exact optimum, greedy, SA, random", "solvers.py"],
  ], [1700, 2700, 2700, 1926]),
  spacer(),
);

// 4 Data + CNN
children.push(
  H1("4. Data and heatmap model"),
  H2("4.1 Data"),
  P("The pipeline uses the Sen1Floods11 v1.1 hand-labelled subset (Bonafilia et al., 2020): 512 x 512 Sentinel-2 chips (13 bands) with a water mask. A small subset (24 chips in fast mode, 60 in full mode) is downloaded anonymously from the public bucket gs://sen1floods11. Chips need at least 5% valid pixels and 2% to 90% water. If the bucket is unreachable, a procedural generator produces chips with the same schema."),
  P(`**Data source in this reference run: ${dataSrc}.** The build environment could not reach Google Cloud Storage, so the synthetic fallback was used. On Kaggle with Internet ON, the notebook downloads the real subset.`),
  P("Each chip becomes four 64 x 64 feature maps: water fraction, NDVI, an NDBI built-up proxy and flood-edge proximity, which are pooled onto the 3 x 4 planning grid."),
  eq("NDVI = (B8 − B4) / (B8 + B4),     NDBI = (B11 − B8) / (B11 + B8)"),
  eq("e = max(0, 1 − d / 6),   d = distance to the shoreline in pixels"),
  P("**Survivor target (proxy).** Sen1Floods11 has no survivor labels, so a hand-defined proxy is used per cell i, with flood fraction f, built-up score u and edge proximity e:"),
  eq("t_i = clip(0.45 f_i + 0.35 u_i + 0.20 e_i + η_i, 0, 1),     η_i ~ N(0, 0.03^2)"),
  P("The CNN therefore learns a known function plus noise. This is stated again in the limitations."),
  ...figure("01_data_overview.png", "Figure 2. One chip: RGB composite, water, NDVI, built-up proxy and edge proximity.", 6.5),
  H2("4.2 CNN"),
  P(`Three Conv-BN-ReLU blocks (16, 32, 64 channels), adaptive average pooling to 3 x 4, a 1 x 1 head and a sigmoid: ${cnn.n_params.toLocaleString("en-US")} parameters. Trained with MSE and Adam (learning rate 0.001, batch 8, ${cnn.epochs} epochs) on an 80/20 split by chip. Predictions are rescaled per chip to [0, 1] and used as the QUBO weights.`),
  table(["Validation metric", "Value", "Meaning"], [
    ["MAE", num(cnn.validation.mae), `mean-predictor baseline: ${num(cnn.mean_predictor_val_mae)}`],
    ["Pearson r", num(cnn.validation.pearson_r, 2), "linear agreement with the proxy target"],
    ["Spearman rho", num(cnn.validation.spearman_rho, 2), "agreement on the ranking of cells"],
    ["Top-4 overlap", num(cnn.validation.topk_overlap, 2), "share of the true best 4 cells ranked top 4"],
  ], [2400, 1400, 5226]),
  spacer(),
  ...figure("02_heatmap_prediction.png", "Figure 3. Proxy truth versus CNN prediction on validation chips.", 6.3),
);

// 5 Agent
children.push(
  H1("5. Agent layer"),
  P("A deterministic keyword parser turns a field report into structured updates; no API key or network access is needed. Cells are named letter = row, number = column (A1 to C4). Each cell mentioned gets the nearest action keyword in its clause. Out-of-range cells (for example D5 or Z9) are ignored and logged."),
  table(["Action", "Example keywords", "Effect"], [
    ["boost", "survivors, rooftop, stranded, boat, SOS", "w_i multiplied by 1.5"],
    ["block", "collapsed, bridge down, impassable, no-fly", "w_i = 0 and a QUBO penalty on that cell"],
    ["reduce", "evacuated, cleared, rescued, no survivors", "w_i multiplied by 0.5"],
  ], [1400, 4000, 3626]),
  spacer(),
  P("An optional local-LLM parser sits behind a configuration flag (off by default). It uses the same output schema and checker, and falls back to the rules on malformed output. Five sample reports were applied in sequence to the demo heatmap; QAOA re-solved after each one."),
  table(["Report", "Updates", "QAOA placement after", "Gap"],
    (S.agent || []).map((a, i) => [`${i + 1}. ${a.report}`, a.updates.map((u) => `${u[0]} ${u[1]}`).join(", ") || "none",
      a.placement_after.join(", "), pct(a.gap_after)]),
    [4300, 1800, 1900, 1026]),
  spacer(),
  ...figure("09_agent_before_after.png", "Figure 4. Heatmap and QAOA placement before (top) and after (bottom) each report. Hatched cells are blocked.", 6.5),
);

// 6 QUBO
children.push(
  H1("6. QUBO formulation"),
  P("Binary variable x_i = 1 means a drone hovers over cell i (i = 0 to 11). Inputs: likelihood w_i; overlap O_ij = 1 for touching cells (8-neighbours); radio link C_ij = 1 for cells within R = 2.25 cell widths; blocked set B; k = 4 drones. The objective to minimise is:"),
  eq("f(x) = −Σ_{i} w_i x_i + a Σ_{i<j} O_{ij} x_i x_j − b Σ_{i<j} C_{ij} x_i x_j,     a = 0.6,  b = 0.3"),
  P("The constraints are added as penalties:"),
  eq("QUBO(x) = f(x) + P (Σ_{i} x_i − k)^2 + P_{block} Σ_{i∈B} x_i"),
  P("Using x_i^2 = x_i, the model is written as QUBO(x) = x^T Q x + c with an upper-triangular Q:"),
  eq("Q_{ii} = −w_i + P(1 − 2k) + P_{block}[i∈B],     Q_{ij} = a O_{ij} − b C_{ij} + 2P,     c = P k^2"),
  P(`**Penalty weight.** P = 2 (max w + a + b) = ${num(S.qubo.P, 1)}. Adding a fifth drone gains at most max w + b k = 2.2, and dropping to three saves at most a k = 2.4, so P exceeds both. Every instance is checked by scoring all 4,096 bit patterns: the global minimum always had exactly four drones at the default P.`),
  P(`**Verification.** The model is built as a Qiskit QuadraticProgram, converted with QuadraticProgramToQubo, and compared with the explicit matrix on 50 random bit patterns (largest difference ${S.qubo.max_abs_qp_vs_matrix.toExponential(1)}). The Ising form has ${S.qubo.n_pauli_terms} Pauli terms and offset ${num(S.qubo.ising_offset, 2)}. NumPyMinimumEigensolver returns the same optimum as brute force.`),
  P("**Design choices.** The radio range is 2.25, not 1.5: at 1.5 the radio and overlap matrices are identical, so the overlap penalty and link reward partly cancel. At 2.25, touching drones cost +0.3 net and drones two cells apart earn −0.3, which is the intended trade-off. Blocked cells use a linear penalty rather than removing the variable, so qubit i is always cell i and QAOA parameters can be reused after a report."),
);

// 7 QAOA
children.push(
  H1("7. Quantum solver (QAOA)"),
  P("Substituting x_i = (1 − z_i)/2 turns the QUBO into an Ising Hamiltonian H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j, divided by its largest coefficient. QAOA prepares"),
  eq("|ψ(γ, β)⟩ = Π_{l=1..p} exp(−i β_l Σ_{j} X_j) exp(−i γ_l H_C) H^{⊗12} |0⟩"),
  P("and minimises E(γ, β) = ⟨ψ|H_C|ψ⟩. Implementation: qiskit_algorithms.QAOA, p = 2, Aer SamplerV2 with 8,192 shots per evaluation, COBYLA with a first step of 0.15, up to 60 evaluations in fast mode (150 and three restarts in full mode). A final 4,096-shot sample is drawn from the optimised circuit; exact probabilities come from the statevector."),
  P("**Initialisation.** The first start is a linear ramp from adiabatic evolution: γ_l = s_l dt, β_l = −(1 − s_l) dt, with s_l = (l − 1/2)/p and dt = 0.5. The negative β matches Qiskit's mixer sign. Positive-β ramps and COBYLA's default step of 1.0 both stalled in a worse valley during testing."),
  P("**Read-outs.** Three rules are reported because they differ widely: QAOA (most probable valid bitstring of the exact distribution, simulator-only), QAOA-shots (most frequent valid bitstring in 4,096 shots, hardware-realistic) and QAOA-best-sample (best valid bitstring among the shots, close to brute force over 495 valid placements)."),
  ...figure("15_landscape_p1.png", "Figure 5. p = 1 energy landscape with the COBYLA path of a Qiskit QAOA run.", 4.6),
  ...figure("17_circuit_diagram.png", "Figure 6. QAOA circuit (p = 1) for a 6-qubit toy instance: RZ cost terms, RZZ pair terms, RX mixer.", 6.5),
);

// 8 Baselines
children.push(
  H1("8. Classical baselines"),
  table(["Method", "How it works", "Cost per instance"], [
    ["Exact", "Scores all C(12,4) = 495 valid placements; cross-checked with NumPyMinimumEigensolver", "495 evaluations, < 1 ms"],
    ["Greedy", "Adds the cell with the best marginal gain, four times", "42 evaluations"],
    ["Simulated annealing", "Swap moves that keep four drones; temperature 1.0 to 0.01", "3,000 steps"],
    ["Random", "Mean over 1,000 random valid placements", "1,000 evaluations"],
  ], [2000, 4900, 2126]),
  spacer(),
);

// 9 Results
const resRows = summary.map((r) => [r.method, pct(Number(r.gap_mean), 2), pct(Number(r.gap_median), 2),
  `${r.tie_vs_exact} / ${Number(r.tie_vs_exact) + Number(r.loss_vs_exact)}`,
  r.p_valid_mean === "" ? "n/a" : num(r.p_valid_mean, 3), r.p_optimal_mean === "" ? "n/a" : num(r.p_optimal_mean, 4),
  pct(Number(r.connected_rate), 0), Number(r.evals_mean).toFixed(0)]);
children.push(
  H1("9. Results"),
  P(`Every method ran on the same ${nInst} instances (one per chip heatmap, validation chips first). The optimality gap is (f − f*)/|f*| against the exact optimum f*.`),
  table(["Method", "Mean gap", "Median gap", "Optimum hit", "P(valid)", "P(opt)", "Linked", "Evals"],
    resRows, [1900, 1000, 1050, 1050, 950, 1050, 1050, 976]),
  spacer(),
  P(`**Paired test.** Wilcoxon signed-rank on objective values, QAOA versus greedy: ${wil.note}.`),
  P("**Reading the table.**"),
  bullet(`QAOA's most probable valid state is usually the best or a near-best placement, and it beat greedy on average (${pct(Number(q.gap_mean), 2)} versus ${pct(Number(g.gap_mean), 2)}).`),
  bullet(`The QAOA state is weak: P(valid) is about ${num(q.p_valid_mean, 2)}, and P(optimal) = ${num(q.p_optimal_mean, 4)} is only about ${(Number(q.p_optimal_mean) * 4096).toFixed(0)} times a uniform guess over 4,096 patterns.`),
  bullet(`Hardware-style read-out fails: the most frequent valid shot was ${pct(Number(byM["QAOA-shots"].gap_mean), 0)} off on average, because each valid placement holds roughly 0.1% of the probability.`),
  bullet("Exact search and simulated annealing find the optimum in milliseconds, as expected at 12 variables."),
  H2("9.1 Demo scenario"),
  P(`On the demo chip (${demo.name}), the exact optimum is ${demo.Exact.cells.join(", ")} (QUBO value ${num(demo.Exact.objective)}). QAOA chose ${demo.QAOA.cells.join(", ")} (gap ${pct(demo.QAOA.gap)}); greedy chose ${demo.Greedy.cells.join(", ")} (gap ${pct(demo.Greedy.gap)}). Greedy took the two best cells first, then picked spread-out cells for radio links and could not go back for the high-value cell next to its first pick.`),
  ...figure("05_method_comparison.png", "Figure 7. Demo scenario: QAOA, exact and greedy placements. Triangles are drones, dashed circles footprints, blue lines radio links, the square the base station.", 6.6),
  ...figure("08_optimality_gap.png", "Figure 8. Optimality gap per method across instances (exact = 0%).", 5.6),
  ...figure("07_bitstring_probabilities.png", "Figure 9. Top-20 QAOA bitstrings by probability on the demo scenario.", 6.3),
);

// 10 Bonus
const depthTxt = depth.map((r) => `p=${r.p}: gap ${pct(r.gap, 2)}, P(valid) ${num(r.p_valid, 2)}, P(opt) ${num(r.p_optimal, 4)}, ${r.evals.toFixed(0)} evals`).join("; ");
const penLow = pen.find((r) => r.multiplier === 0.25);
children.push(
  H1("10. Bonus quantum experiments"),
  P(`The clearest result is the warm start: after a field report, reusing the previous optimal angles converged in ${ws.warm.evals_to_converge} evaluations to a ${pct(ws.warm.gap)} gap, against ${ws.random_init_mean.evals_to_converge.toFixed(0)} evaluations and a ${pct(ws.random_init_mean.gap)} gap from random angles.`),
  table(["Study", "Finding"], [
    ["1. Depth p", depthTxt],
    ["2. Penalty weight", `At 0.25 x P the QUBO minimum was invalid on ${penLow ? ((1 - penLow.ground_state_valid) * 100).toFixed(0) : "?"}% of instances; from 0.5 x P upward it was always valid. P(valid) stayed near 0.5 across the sweep, and the decoded gap did not change.`],
    ["3. Warm start", `warm: ${ws.warm.evals_to_converge} evaluations to converge, gap ${pct(ws.warm.gap)}; random: ${ws.random_init_mean.evals_to_converge.toFixed(1)} evaluations, gap ${pct(ws.random_init_mean.gap)}`],
    ["4. Optimisers", Object.entries(opts).map(([k, r]) => `${k}: gap ${pct(r.gap)}, ${r.evals.toFixed(0)} evals, ${r.runtime_s.toFixed(1)} s`).join("; ")],
    ["5. XY mixer", xy["XY ring + Dicke"] ? `Dicke start + ring XY mixer: P(valid) ${num(xy["XY ring + Dicke"].p_valid, 2)}, P(optimal) ${num(xy["XY ring + Dicke"].p_optimal, 4)} versus ${num(xy["X (standard)"].p_optimal, 4)} for the standard mixer; gap ${pct(xy["XY ring + Dicke"].gap)} versus ${pct(xy["X (standard)"].gap)}` : "not run"],
    ["6. Noise", noise1 ? `P(valid) ${num(noise0.p_valid, 2)} noiseless, ${num(noise1.p_valid, 2)} at 0.1% / 1% / 1% (1q / 2q / readout), ${num(noise4.p_valid, 2)} at 4x; the circuit has 264 CNOTs` : "not run"],
    ["7. Landscape", B.landscape ? `one valley near gamma ${num(B.landscape.optimum_gamma_beta[0], 2)}, beta ${num(B.landscape.optimum_gamma_beta[1], 2)}; COBYLA reached it from the ramp start` : "not run"],
    ["8. Scaling", (B.scaling || []).map((r) => `${r.n_qubits}q: ${r.time_per_eval_s.toFixed(2)} s/eval`).join(", ") + "; statevector memory doubles per qubit"],
  ], [1900, 7126]),
  spacer(),
  ...figure("12_warm_start.png", "Figure 10. Re-solve after a field report: warm start versus random initialisation.", 5.6),
  ...figure("10_depth_study.png", "Figure 11. QAOA depth study.", 6.5),
  ...figure("11_penalty_sensitivity.png", "Figure 12. Penalty-weight sensitivity.", 6.3),
  ...figure("19_xy_mixer.png", "Figure 13. Standard X mixer versus Dicke state + ring XY mixer at the same depth.", 6.3),
  ...figure("14_noise_robustness.png", "Figure 14. Noise robustness of the trained circuit.", 6.3),
);

// 11 Limitations
children.push(
  H1("11. Limitations"),
  ...[
    "12 to 16 variables are solved exactly in milliseconds, so there is no quantum advantage claim.",
    "With the standard X mixer, QAOA mostly learns the four-drone constraint: P(optimal) stays a small multiple of uniform guessing at p ≤ 3, and results depend on the penalty weight.",
    "The headline QAOA read-out uses the exact statevector distribution, which only a simulator has. The hardware-realistic shot read-out is much worse; best-of-shots is close to brute force.",
    "Survivor likelihood is a proxy formula, not real survivor data, so the CNN learns a known function.",
    "The pairwise radio reward does not guarantee a connected relay chain to the base station; connectivity is checked afterwards.",
    "The agent is a keyword parser and is brittle to unusual phrasing; a human confirmation step is needed in practice.",
    "Simulator and simple noise-model results do not predict real-hardware behaviour. Scaling beyond 16 qubits was not tested.",
    synthetic ? "This reference run used synthetic chips because Google Cloud Storage was unreachable from the build environment; the Sen1Floods11 path is implemented but was not executed here." : "Results use a small Sen1Floods11 subset.",
  ].map(bullet),
  H2("Next steps"),
  bullet("Enforce connectivity to base with a path or spanning-tree constraint."),
  bullet("Use the XY mixer as the main solver with a larger optimisation budget and depth."),
  bullet("Add drone battery limits as a second resource constraint, and try decomposition for larger grids."),
);

// 12 Reproducibility
children.push(
  H1("12. Reproducibility"),
  bullet("Kaggle: import notebook/floodwing_kaggle.ipynb, set Internet ON, Run All. FAST_MODE=True targets under 30 minutes."),
  bullet("Local: pip install -r requirements.txt, then run_all(Config(fast_mode=True)); pytest -q runs 14 unit tests."),
  bullet(`Seed 42 for Python, NumPy, PyTorch, the Aer sampler and the optimisers. Reference run: ${Math.round(S.total_runtime_s)} s on a 2-core CPU.`),
  bullet(`Pinned Qiskit stack: qiskit ${S.versions.qiskit}, qiskit-aer ${S.versions.qiskit_aer}, qiskit-algorithms ${S.versions.qiskit_algorithms}, qiskit-optimization ${S.versions.qiskit_optimization}.`),
  bullet("All headline numbers in this report are read from results/summary.json and results/tables/summary.csv by tools/build_report.js."),
  H2("Credits"),
  P("Sen1Floods11: Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. CVPR Workshops. CC BY 4.0. Qiskit, Qiskit Aer, qiskit-algorithms and qiskit-optimization (Apache-2.0). QAOA: Farhi, Goldstone, Gutmann (2014). XY mixers: Hadfield et al. (2019); Wang et al. (2020). No third-party code was reused; see CREDITS.md."),
);

// ---------- document ----------
const doc = new Document({
  creator: "FloodWing team",
  title: "FloodWing report",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: BLUE },
        paragraph: { spacing: { before: 360, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "333333" },
        paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "FloodWing | Qiskit Fall Fest 2026", size: 16, color: "888888", font: FONT })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 16, color: "888888", font: FONT })] })] }) },
    children,
  }],
});

const outDir = path.join(ROOT, "report");
fs.mkdirSync(outDir, { recursive: true });
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(outDir, "FloodWing_Report.docx"), buf);
  console.log("wrote report/FloodWing_Report.docx");
});
