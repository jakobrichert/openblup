// OpenBLUP Studio: the results area.

import { h, s, clear, icon, ICONS, segmented, select, download, toCsv } from "./dom.js";
import { fmt, fixed, int, pct, pValue, stars, duration, isNum } from "./format.js";
import {
  figure, dataTable, dotInterval, hbars, columns as columnChart, scatter, lineChart, heatmap,
  partitionBar, sequentialScale, divergingScale, seriesHex, legendItems,
} from "./charts.js";
import { EXAMPLES, fixedFormula, termString, residualString } from "./examples.js";

// ------------------------------------------------------------ model helpers

const paramName = (p) => p[0];
const paramValue = (p) => p[1];

function componentByName(fit, name) {
  return fit.fit.variance_components.find((c) => c.name === name);
}

/** The variance of a scalar component (Identity, or AR1-type with sigma2 + correlations). */
function scalarVariance(comp) {
  if (!comp || !comp.parameters.length) return null;
  const [first, ...rest] = comp.parameters;
  const n = paramName(first);
  const isVar = n === "sigma2" || n.endsWith(".sigma2");
  if (!isVar || !rest.every((p) => /(^|\.)rho$/.test(paramName(p)))) return null;
  return paramValue(first);
}

function termMeta(result, term) {
  return result.model.random_terms.find((t) => t.label === term);
}

function reliabilityFor(result, term) {
  const meta = termMeta(result, term);
  if (!meta || meta.inner_structure !== null || meta.outer_structure !== "idv") return null;
  const s2 = scalarVariance(componentByName(result, term));
  if (!isNum(s2) || s2 <= 0) return null;
  return (se) => Math.min(1, Math.max(0, 1 - (se * se) / s2));
}

function pedigreeTermOf(fitState) {
  if (!fitState.pedigreeName) return null;
  const m = fitState.model;
  if (m.pedigreeTerm) return m.pedigreeTerm;
  const t = m.random[0];
  return t ? t.b || t.a : null;
}

/** Heritability-type ratio for single-term models with scalar variances. */
function varianceRatio(fitState) {
  const r = fitState.result;
  const terms = r.model.random_terms;
  if (terms.length !== 1) return null;
  const t = terms[0];
  if (t.inner_structure !== null || t.outer_structure !== "idv") return null;
  const su = scalarVariance(componentByName(r, t.label));
  const se = scalarVariance(componentByName(r, "residual"));
  if (!isNum(su) || !isNum(se) || su + se <= 0) return null;
  const ped = pedigreeTermOf(fitState) === t.label;
  return {
    value: su / (su + se),
    label: ped ? "Heritability (h²)" : "Heritability (H², plot basis)",
    sub: ped ? `σ²a / (σ²a + σ²e) for ${t.label}` : `σ²${t.label} / (σ²${t.label} + σ²e)`,
    term: t.label,
  };
}

function levelRecords(result, meta) {
  const labels = result.observations.labels;
  const counts = new Map();
  if (!meta) return counts;
  const [a, b] = meta.columns;
  const la = labels[a];
  const lb = b ? labels[b] : null;
  if (!la) return counts;
  for (let i = 0; i < la.length; i++) {
    const key = lb ? `${la[i]}:${lb[i]}` : la[i];
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return counts;
}

function obsLabel(result, i) {
  const parts = Object.entries(result.observations.labels).map(([k, v]) => `${k} ${v[i]}`);
  return parts.length ? parts.join(" · ") : `record ${i + 1}`;
}

function effectRows(fitState, block) {
  const r = fitState.result;
  const meta = termMeta(r, block.term);
  const rel = reliabilityFor(r, block.term);
  const records = levelRecords(r, meta);
  const ped = r.pedigree && pedigreeTermOf(fitState) === block.term ? new Map(r.pedigree.ids.map((id, i) => [id, r.pedigree.inbreeding[i]])) : null;
  const rows = block.effects.map((e) => ({
    level: e.level,
    estimate: e.estimate,
    se: e.se,
    lo: e.estimate - 1.96 * e.se,
    hi: e.estimate + 1.96 * e.se,
    reliability: rel ? rel(e.se) : null,
    records: records.get(e.level) ?? 0,
    inbreeding: ped ? ped.get(e.level) ?? null : null,
  }));
  const sorted = [...rows].sort((x, y) => y.estimate - x.estimate);
  sorted.forEach((row, i) => { row.rank = i + 1; });
  return rows;
}

// ------------------------------------------------------------ building blocks

function tile(label, value, sub, { cls = "", meter } = {}) {
  return h("div", { class: `card tile ${cls}` },
    h("span", { class: "label", text: label }),
    h("span", { class: "value", text: value }),
    meter !== undefined ? h("div", { class: "meter", attrs: { role: "presentation" } }, h("div", { style: { width: `${Math.round(Math.min(1, Math.max(0, meter)) * 100)}%` } })) : null,
    sub ? (sub instanceof Node ? sub : h("span", { class: "sub", text: sub })) : null);
}

function callout(kind, items) {
  if (!items.length) return null;
  const ic = kind === "note" ? ICONS.info : ICONS.alert;
  return h("div", { class: `callout ${kind}`, attrs: { role: kind === "note" ? "note" : "alert" } },
    icon(ic),
    items.length === 1 ? h("span", { text: items[0] }) : h("ul", {}, items.map((t) => h("li", { text: t }))));
}

function card(title, subtitle, ...content) {
  return h("section", { class: "card figure" },
    h("div", { class: "figure-head" }, h("div", {}, h("h3", { text: title }), subtitle ? h("p", { text: subtitle }) : null)),
    ...content);
}

// ------------------------------------------------------------ landing

function heroArt() {
  // A decorative REML-style fit: data dots around a curve, and a tiny pedigree.
  const pts = [[30, 150], [58, 128], [80, 138], [104, 102], [126, 110], [150, 76], [172, 88], [196, 60], [220, 70], [246, 48], [268, 58], [292, 40]];
  return s("svg", { class: "hero-art", viewBox: "0 0 340 220", "aria-hidden": "true" },
    s("rect", { x: 8, y: 8, width: 324, height: 204, rx: 16, fill: "var(--surface-2)", stroke: "var(--border)" }),
    ...[60, 100, 140, 180].map((y) => s("line", { x1: 24, x2: 316, y1: y, y2: y, stroke: "var(--grid)" })),
    s("path", { d: "M24 160 C 110 140, 170 70, 316 36", fill: "none", stroke: "var(--series-1)", "stroke-width": 2.5, "stroke-linecap": "round" }),
    s("path", { d: "M24 160 C 110 140, 170 70, 316 36 L316 64 C 170 100, 110 168, 24 184 Z", fill: "var(--series-1)", opacity: 0.1 }),
    ...pts.map(([x, y]) => s("circle", { cx: x, cy: y + 6, r: 4.5, fill: "var(--series-1)", stroke: "var(--surface-2)", "stroke-width": 2 })),
    s("g", { transform: "translate(228 128)" },
      s("path", { d: "M20 12 V26 H60 V12 M40 26 V40", fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.5 }),
      s("rect", { x: 8, y: 0, width: 24, height: 14, rx: 4, fill: "var(--series-2)" }),
      s("rect", { x: 48, y: 0, width: 24, height: 14, rx: 4, fill: "var(--series-3)" }),
      s("rect", { x: 28, y: 40, width: 24, height: 14, rx: 4, fill: "var(--series-1)" })));
}

function landing(state, actions) {
  const hero = h("section", { class: "card hero" },
    h("div", {},
      h("h1", {}, "Mixed models, ", h("em", { text: "right in your browser." })),
      h("p", { text: "OpenBLUP Studio runs the complete OpenBLUP REML engine, compiled from Rust to WebAssembly, on your machine. Load a trial or pedigree, pick the variance structures and get variance components, BLUPs, diagnostics and field maps in a fraction of a second. Your data never leaves the page." }),
      h("ul", {},
        h("li", { text: "AI-REML with standard errors, boundary detection and Satterthwaite / Kenward-Roger F-tests" }),
        h("li", { text: "Pedigree animal models, AR1 × AR1 spatial residuals, factor-analytic G×E, unstructured multi-trait" }),
        h("li", { text: "Every model comes with the equivalent CLI command and Python code" })),
      h("div", { class: "row", style: { marginTop: "22px", flexWrap: "wrap" } },
        EXAMPLES.slice(0, 3).map((ex) => h("button", {
          class: "btn", type: "button",
          on: { click: () => actions.loadExample(ex.id) },
        }, icon(ex.icon), `Try: ${ex.title}`)))),
    heroArt());
  const blocks = [hero];
  if (state.error) blocks.push(callout("error", [state.error]));
  if (state.dataset) blocks.push(dataView(state));
  return blocks;
}

// ------------------------------------------------------------ tabs

function dataView(state) {
  const info = state.dataset.info;
  const colTable = dataTable([
    { key: "name", label: "Column" },
    { key: "kind", label: "Type", format: (v, r) => (v === "factor" ? "factor" : r.integer_like ? "numeric (integer)" : "numeric") },
    { key: "n_distinct", label: "Distinct", num: true, format: int },
    { key: "n_missing", label: "Missing", num: true, format: int },
    { key: "range", label: "Range / levels", format: (_, r) => (r.kind === "factor" ? `${r.levels.join(", ")}${r.n_distinct > r.levels.length ? ", …" : ""}` : `${fmt(r.min)} – ${fmt(r.max)}`) },
  ], info.columns);
  const preview = dataTable(info.columns.map((c, j) => ({ key: (row) => row[j], label: c.name, num: c.kind === "numeric", format: (v) => (v === null ? "NA" : typeof v === "number" ? fmt(v, 6) : v) })), info.preview);
  return h("div", { class: "grid-2" },
    card("Columns", `${state.dataset.name} · ${int(info.n_rows)} rows`, colTable),
    card("Preview", `First ${info.preview.length} rows`, preview));
}

function overviewTab(fitState, actions) {
  const r = fitState.result;
  const comps = r.fit.variance_components;
  const out = h("div", { class: "grid-2" });

  // Variance components
  const scalar = comps.map((c) => ({ c, v: scalarVariance(c) }));
  const partition = scalar.every((x) => isNum(x.v)) && comps.length >= 2 && comps.length <= 8;
  const paramRows = comps.flatMap((c) => c.parameters.map((p, i) => ({
    component: c.name, structure: c.structure, name: paramName(p), value: paramValue(p),
    se: c.se[i] > 0 ? c.se[i] : null, boundary: c.at_boundary[i],
  })));
  const paramTable = dataTable([
    { key: "component", label: "Component" },
    { key: "name", label: "Parameter" },
    { key: "value", label: "Estimate", num: true, format: (v) => fmt(v) },
    { key: "se", label: "SE", num: true, format: (v) => fmt(v) },
    { key: "boundary", label: "", render: (v) => (v ? h("span", { class: "flag", text: "B", attrs: { title: "Estimated at the boundary (fixed at zero)" } }) : "") },
  ], paramRows);
  const vcCard = h("section", { class: "card figure" },
    h("div", { class: "figure-head" }, h("div", {},
      h("h3", { text: "Variance components" }),
      h("p", { text: comps.map((c) => `${c.name}: ${c.structure}`).join(" · ") }))));
  if (partition) {
    const parts = scalar.map((x, i) => ({ label: x.c.name, value: x.v, color: seriesHex(i) }));
    const total = parts.reduce((a, p) => a + p.value, 0);
    vcCard.append(h("div", { class: "legend" }, legendItems(parts.map((p) => ({ label: `${p.label} ${pct(p.value / total, 1)}`, color: p.color })))));
    const host = h("div", { style: { marginBottom: "12px" } });
    vcCard.append(host);
    const draw = () => { clear(host); if (host.clientWidth) partitionBar(host, host.clientWidth, { parts }); };
    new ResizeObserver(draw).observe(host);
    vcCard.classList.add("figure");
    vcCard.redraw = draw;
  }
  vcCard.append(paramTable);
  if (paramRows.some((p) => p.boundary)) {
    const noSe = paramRows.every((p) => p.se === null);
    vcCard.append(h("p", { class: "figure-foot", text: `B: the parameter converged to the boundary of its space and is fixed there, as in ASReml.${noSe ? " The engine does not report standard errors while a variance is on the boundary." : ""}` }));
  }
  out.append(vcCard);

  // Fixed effects and Wald tests
  const fe = dataTable([
    { key: "term", label: "Term" },
    { key: "level", label: "Level" },
    { key: "estimate", label: "Estimate", num: true, format: (v) => fmt(v, 5) },
    { key: "se", label: "SE", num: true, format: (v) => fmt(v) },
  ], r.fit.fixed_effects);
  const wald = dataTable([
    { key: "term", label: "Term" },
    { key: "f_statistic", label: "F", num: true, format: (v) => fmt(v) },
    { key: "num_df", label: "df", num: true, format: int },
    { key: "den_df", label: "ddf", num: true, format: (v) => fixed(v, 1) },
    { key: "p_value", label: "p", num: true, render: (v) => h("span", {}, pValue(v), " ", h("span", { class: "stars", text: stars(v) })) },
  ], r.wald_tests, { empty: "No fixed terms to test" });
  out.append(card("Fixed effects (BLUE)", `${fitState.model.response} ~ ${fixedFormula(fitState.model)}`, fe,
    h("h3", { style: { fontSize: "13.5px", margin: "16px 0 8px" }, text: `Wald F-tests · ${r.ddf_method} df` }), wald,
    h("p", { class: "figure-foot", text: "Signif. codes: *** < 0.001, ** < 0.01, * < 0.05, . < 0.1" })));

  // Top random effects
  const block = r.fit.random_effects[0];
  if (block) {
    const rows = effectRows(fitState, block).sort((a, b) => b.estimate - a.estimate);
    const top = rows.slice(0, 12);
    out.append(figure({
      title: rows.length > top.length ? `Top ${top.length} of ${rows.length} · ${block.term}` : `${block.term} · all ${rows.length} levels`,
      subtitle: "BLUP ± 1.96 SE",
      chart: (host, width) => dotInterval(host, width, { items: top.map((d) => ({ label: d.level, value: d.estimate, lo: d.lo, hi: d.hi, tip: effectTip(d) })), xTitle: "Predicted effect" }),
      table: () => effectTable(top),
      foot: rows.length > top.length ? "The Random effects tab ranks every level." : null,
    }));
  }

  // Residuals
  out.append(residualScatter(fitState, { height: 260 }));
  return out;
}

function effectTip(d) {
  const rows = [
    { value: fmt(d.estimate), label: "estimate" },
    { value: `${fmt(d.lo)} to ${fmt(d.hi)}`, label: "95% interval" },
  ];
  if (isNum(d.reliability)) rows.push({ value: fixed(d.reliability, 2), label: "reliability" });
  rows.push({ value: int(d.records), label: d.records === 1 ? "record" : "records" });
  if (isNum(d.inbreeding)) rows.push({ value: fixed(d.inbreeding, 3), label: "inbreeding F" });
  return { title: `#${d.rank} · ${d.level}`, rows };
}

function effectTable(rows) {
  const cols = [
    { key: "rank", label: "Rank", num: true, format: int },
    { key: "level", label: "Level" },
    { key: "estimate", label: "Estimate", num: true, format: (v) => fmt(v) },
    { key: "se", label: "SE", num: true, format: (v) => fmt(v) },
    { key: "lo", label: "95% interval", num: true, format: (_, r) => `${fmt(r.lo, 3)} … ${fmt(r.hi, 3)}` },
  ];
  if (rows.some((r) => isNum(r.reliability))) cols.push({ key: "reliability", label: "Reliability", num: true, format: (v) => fixed(v, 3) });
  cols.push({ key: "records", label: "Records", num: true, format: int });
  if (rows.some((r) => isNum(r.inbreeding))) cols.push({ key: "inbreeding", label: "Inbreeding F", num: true, format: (v) => fixed(v, 4) });
  return dataTable(cols, rows);
}

function effectsTab(fitState, view, actions) {
  const r = fitState.result;
  const blocks = r.fit.random_effects;
  if (!blocks.length) return h("div", { class: "card empty", text: "This model has no random terms." });
  view.term = blocks.some((b) => b.term === view.term) ? view.term : blocks[0].term;
  view.top = view.top ?? 30;
  view.sort = view.sort ?? "top";
  view.query = view.query ?? "";
  const block = blocks.find((b) => b.term === view.term);
  const meta = termMeta(r, block.term);
  const all = effectRows(fitState, block);
  const outerLevels = meta && meta.columns.length === 2 ? [...new Set(r.observations.labels[meta.columns[0]] || [])] : null;
  if (!outerLevels || !outerLevels.includes(view.group)) view.group = "";

  const body = h("div", { class: "stack" });
  const draw = () => {
    clear(body);
    let rows = all;
    if (view.group) rows = rows.filter((d) => d.level.startsWith(`${view.group}:`));
    const q = view.query.trim().toLowerCase();
    if (q) rows = rows.filter((d) => d.level.toLowerCase().includes(q));
    rows = [...rows].sort(view.sort === "top" ? (a, b) => b.estimate - a.estimate : view.sort === "bottom" ? (a, b) => a.estimate - b.estimate : (a, b) => a.level.localeCompare(b.level, undefined, { numeric: true }));
    const shown = view.top ? rows.slice(0, view.top) : rows;
    const isPed = pedigreeTermOf(fitState) === block.term;
    body.append(figure({
      title: `${isPed ? "Estimated breeding values" : "Predicted random effects"} · ${block.term}`,
      subtitle: `${shown.length} of ${rows.length} levels${view.group ? ` in ${view.group}` : ""} · BLUP ± 1.96 SE${isPed ? ` · ${int(all.filter((d) => d.records === 0).length)} animals without records` : ""}`,
      chart: shown.length ? (host, width) => dotInterval(host, width, {
        items: shown.map((d) => ({ label: view.group ? d.level.slice(view.group.length + 1) : d.level, value: d.estimate, lo: d.lo, hi: d.hi, tip: effectTip(d) })),
        xTitle: isPed ? "Estimated breeding value" : "Predicted effect",
        rowHeight: shown.length > 80 ? 16 : 22,
      }) : null,
      table: () => effectTable(shown),
    }));
    if (isPed && r.pedigree) {
      const p = r.pedigree;
      body.append(h("div", { class: "kpis" },
        tile("Animals in pedigree", int(p.n_animals), `${int(all.filter((d) => d.records > 0).length)} with records`),
        tile("Inbred animals", int(p.n_inbred), `max F ${fixed(p.max_inbreeding, 3)}`),
        tile("Mean inbreeding", fixed(p.mean_inbreeding, 4), "Meuwissen & Luo (1992)")));
    }
  };

  const filters = h("div", { class: "filters" });
  if (blocks.length > 1) {
    filters.append(h("label", {}, "Term", select(blocks.map((b) => [b.term, b.term]), view.term, (v) => { view.term = v; view.group = ""; actions.rerender(); })));
  }
  if (outerLevels) {
    filters.append(h("label", {}, meta.columns[0], select([["", "All"], ...outerLevels.map((l) => [l, l])], view.group, (v) => { view.group = v; draw(); })));
  }
  filters.append(
    h("label", {}, "Search", h("input", { type: "search", value: view.query, placeholder: "level…", on: { input: (e) => { view.query = e.target.value; draw(); } } })),
    h("label", {}, "Show", segmented([[30, "30"], [100, "100"], [0, "All"]], view.top, (v) => { view.top = v; actions.rerender(); })),
    h("label", {}, "Order", segmented([["top", "Highest"], ["bottom", "Lowest"], ["name", "Name"]], view.sort, (v) => { view.sort = v; actions.rerender(); })),
    h("button", {
      class: "btn", type: "button", style: { marginLeft: "auto" },
      on: { click: () => download(`${block.term.replace(/[^\w-]+/g, "_")}_blups.csv`, toCsv(["term", "level", "estimate", "se", "reliability", "records", "rank"], all.map((d) => [block.term, d.level, d.estimate, d.se, d.reliability ?? "", d.records, d.rank])), "text/csv") },
    }, icon(ICONS.download), "CSV"));
  draw();
  return h("div", { class: "stack" }, h("div", { class: "card", style: { padding: "12px 14px" } }, filters), body);
}

// Inverse standard normal CDF (Acklam).
function qnorm(p) {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const tail = (q) => (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  if (p < 0.02425) return tail(Math.sqrt(-2 * Math.log(p)));
  if (p > 1 - 0.02425) return -tail(Math.sqrt(-2 * Math.log(1 - p)));
  const q = p - 0.5, r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

function residualSeries(fitState) {
  const r = fitState.result;
  const y = r.observations.y;
  const d = r.diagnostics;
  if (d) {
    return { standardized: true, points: y.map((yi, i) => ({ i, y: yi, fitted: d.fitted[i], res: d.standardized[i], leverage: d.leverage[i], cooks: d.cooks_distance[i] })) };
  }
  const raw = y.map((yi, i) => yi - r.observations.fitted[i]);
  const mean = raw.reduce((a, v) => a + v, 0) / raw.length;
  const sd = Math.sqrt(raw.reduce((a, v) => a + (v - mean) ** 2, 0) / Math.max(1, raw.length - 1)) || 1;
  return { standardized: false, points: y.map((yi, i) => ({ i, y: yi, fitted: r.observations.fitted[i], res: raw[i] / sd, raw: raw[i] })) };
}

function residualTip(fitState, p) {
  const rows = [
    { value: fmt(p.res, 3), label: "scaled residual" },
    { value: fmt(p.y), label: "observed" },
    { value: fmt(p.fitted), label: "fitted" },
  ];
  if (isNum(p.leverage)) rows.push({ value: fixed(p.leverage, 3), label: "leverage" });
  if (isNum(p.cooks)) rows.push({ value: fmt(p.cooks, 3), label: "Cook's D" });
  return { title: obsLabel(fitState.result, p.i), rows };
}

function residualScatter(fitState, { height = 320 } = {}) {
  const series = residualSeries(fitState);
  const yTitle = series.standardized ? "Standardized residual" : "Residual / SD";
  return figure({
    title: "Residuals against fitted values",
    subtitle: series.standardized ? "Standardized conditional residuals; guides at ± 2" : "Conditional residuals y − Xb − Zu scaled by their SD (structured residual)",
    chart: (host, width) => scatter(host, width, {
      points: series.points.map((p) => ({ x: p.fitted, y: p.res, tip: residualTip(fitState, p) })),
      xTitle: "Fitted value", yTitle, ref: "zero", guides: [-2, 2], height,
    }),
    table: () => dataTable([
      { key: "i", label: "Record", num: true, format: (v) => int(v + 1) },
      { key: "label", label: "Labels", format: (_, p) => obsLabel(fitState.result, p.i) },
      { key: "y", label: "Observed", num: true, format: (v) => fmt(v) },
      { key: "fitted", label: "Fitted", num: true, format: (v) => fmt(v) },
      { key: "res", label: yTitle, num: true, format: (v) => fmt(v, 3) },
    ], series.points),
  });
}

function diagnosticsTab(fitState) {
  const r = fitState.result;
  const series = residualSeries(fitState);
  const out = h("div", { class: "grid-2" });
  if (!series.standardized) {
    out.append(h("div", { class: "span-2" }, callout("note", ["Leverage, Cook's distance and standardized residuals need an IID residual. With a structured residual the plots show conditional residuals scaled by their standard deviation; the spatial pattern is on the Field map tab."])));
  }
  out.append(residualScatter(fitState));
  const sorted = [...series.points].sort((a, b) => a.res - b.res);
  const n = sorted.length;
  const qq = sorted.map((p, k) => ({ ...p, q: qnorm((k + 1 - 0.375) / (n + 0.25)) }));
  out.append(figure({
    title: "Normal Q–Q plot",
    subtitle: "Points on the line indicate normally distributed residuals",
    chart: (host, width) => scatter(host, width, {
      points: qq.map((p) => ({ x: p.q, y: p.res, tip: residualTip(fitState, p) })),
      xTitle: "Theoretical quantile", yTitle: series.standardized ? "Standardized residual" : "Residual / SD", ref: "identity", height: 320,
    }),
    table: () => dataTable([
      { key: "q", label: "Theoretical", num: true, format: (v) => fixed(v, 3) },
      { key: "res", label: "Sample", num: true, format: (v) => fixed(v, 3) },
      { key: "i", label: "Record", num: true, format: (v) => int(v + 1) },
    ], qq),
  }));
  if (series.standardized) {
    const infl = [...series.points].sort((a, b) => b.cooks - a.cooks).slice(0, 10);
    out.append(h("div", { class: "span-2" }, card("Most influential records", "Largest Cook's distance",
      dataTable([
        { key: "i", label: "Record", num: true, format: (v) => int(v + 1) },
        { key: "label", label: "Labels", format: (_, p) => obsLabel(r, p.i) },
        { key: "y", label: "Observed", num: true, format: (v) => fmt(v) },
        { key: "fitted", label: "Fitted", num: true, format: (v) => fmt(v) },
        { key: "res", label: "Std. residual", num: true, format: (v) => fixed(v, 3) },
        { key: "leverage", label: "Leverage", num: true, format: (v) => fixed(v, 3) },
        { key: "cooks", label: "Cook's D", num: true, format: (v) => fmt(v, 3) },
      ], infl))));
  }
  return out;
}

function gridMatrices(result) {
  const g = result.grid;
  const nr = g.row_levels.length, nc = g.col_levels.length;
  const make = () => Array.from({ length: nr }, () => new Array(nc).fill(null));
  const obs = make(), res = make(), fit = make(), idx = make();
  g.cell_index.forEach((cell, i) => {
    const rr = Math.floor(cell / nc), cc = cell % nc;
    obs[rr][cc] = result.observations.y[i];
    fit[rr][cc] = result.observations.fitted[i];
    res[rr][cc] = result.observations.y[i] - result.observations.fitted[i];
    idx[rr][cc] = i;
  });
  return { obs, res, fit, idx, nr, nc };
}

function matrixTable(rowsLabel, rows, cols, M, digits = 3) {
  return dataTable(
    [{ key: "label", label: rowsLabel }, ...cols.map((c, j) => ({ key: (r) => r.values[j], label: String(c), num: true, format: (v) => (isNum(v) ? fmt(v, digits) : "—") }))],
    rows.map((label, i) => ({ label, values: M[i] })));
}

function fieldTab(fitState) {
  const r = fitState.result;
  const g = r.grid;
  const { obs, res, idx, nr, nc } = gridMatrices(r);
  const resComp = componentByName(r, "residual");
  const params = Object.fromEntries(resComp.parameters.map((p, i) => [paramName(p), { v: paramValue(p), se: resComp.se[i] }]));
  const rhoRow = params[`${g.row}.rho`], rhoCol = params[`${g.col}.rho`];
  const flat = (M) => M.flat().filter(isNum);
  const cellTip = (M, label) => (i, j, v) => ({
    title: `${g.row} ${g.row_levels[i]} · ${g.col} ${g.col_levels[j]}`,
    rows: isNum(v) ? [{ value: fmt(v), label }, ...(idx[i][j] !== null ? [{ value: obsLabel(r, idx[i][j]), label: "" }] : [])] : [{ value: "no record" }],
  });
  const n = nr * nc;
  const missing = n - flat(obs).length;
  const kpis = h("div", { class: "kpis" },
    tile("Field", `${nr} × ${nc}`, `${g.row} × ${g.col} · ${missing} empty or missing plot${missing === 1 ? "" : "s"}`),
    rhoRow ? tile(`Row correlation ρ (${g.row})`, fixed(rhoRow.v, 3), rhoRow.se > 0 ? `SE ${fixed(rhoRow.se, 3)}` : "at boundary") : null,
    rhoCol ? tile(`Column correlation ρ (${g.col})`, fixed(rhoCol.v, 3), rhoCol.se > 0 ? `SE ${fixed(rhoCol.se, 3)}` : "at boundary") : null,
    scalarVariance(resComp) !== null ? tile("Spatial residual variance", fmt(scalarVariance(resComp)), resComp.structure) : null);
  const [lo, hi] = [Math.min(...flat(obs)), Math.max(...flat(obs))];
  const maxAbs = Math.max(...flat(res).map(Math.abs));
  const response = fitState.model.response;
  return h("div", { class: "stack" }, kpis,
    figure({
      title: "Spatial trend: residuals y − Xb − Zu",
      subtitle: "What remains after the design and genotype effects; the AR1 × AR1 structure models this field trend",
      chart: (host, width) => heatmap(host, width, {
        rows: g.row_levels, cols: g.col_levels, value: (i, j) => res[i][j],
        color: divergingScale(maxAbs), rowTitle: g.row, colTitle: g.col, legendTitle: "residual (blue below, red above expectation)",
        tip: cellTip(res, "residual"),
      }),
      table: () => matrixTable(g.row, g.row_levels, g.col_levels, res),
    }),
    figure({
      title: `Observed ${response}`,
      subtitle: `Raw plot values laid out on the field (${g.row} down, ${g.col} across)`,
      chart: (host, width) => heatmap(host, width, {
        rows: g.row_levels, cols: g.col_levels, value: (i, j) => obs[i][j],
        color: sequentialScale([lo, hi]), rowTitle: g.row, colTitle: g.col, legendTitle: response,
        tip: cellTip(obs, response),
      }),
      table: () => matrixTable(g.row, g.row_levels, g.col_levels, obs),
    }));
}

function gxeTab(fitState) {
  const r = fitState.result;
  const out = h("div", { class: "stack" });
  for (const cov of r.covariances) {
    const lv = cov.levels;
    const nl = lv.length;
    const grid = h("div", { class: "grid-2" });
    grid.append(figure({
      title: `Correlation between ${cov.factor} levels`,
      subtitle: `${cov.term} · ${cov.structure.toUpperCase()} model`,
      chart: (host, width) => heatmap(host, width, {
        rows: lv, cols: lv, value: (i, j) => cov.correlation[i][j],
        color: divergingScale(1), cellText: (v) => v.toFixed(2), maxCell: 64,
        legendTitle: "correlation", format: (v) => fixed(v, 2),
        tip: (i, j, v) => ({ title: `${lv[i]} × ${lv[j]}`, rows: [{ value: fixed(v, 3), label: "correlation" }, { value: fmt(cov.covariance[i][j]), label: "covariance" }] }),
      }),
      table: () => matrixTable(cov.factor, lv, lv, cov.correlation),
    }));
    const variances = lv.map((l, i) => ({ label: l, value: cov.covariance[i][i] }));
    grid.append(figure({
      title: `Variance within each ${cov.factor}`,
      subtitle: cov.loadings ? "λ² + ψ: common (factor) plus specific variance" : "Diagonal of the estimated covariance matrix",
      chart: (host, width) => hbars(host, width, {
        items: variances.map((d, i) => ({
          ...d,
          tip: {
            title: d.label,
            rows: [
              { value: fmt(d.value), label: "total variance" },
              ...(cov.loadings ? [
                { value: fmt(d.value - (cov.specific[i] ?? 0)), label: "common (λλ′)" },
                { value: fmt(cov.specific[i]), label: "specific ψ" },
              ] : []),
            ],
          },
        })),
        xTitle: "Variance",
      }),
      table: () => dataTable([
        { key: "label", label: cov.factor },
        { key: "value", label: "Variance", num: true, format: (v) => fmt(v) },
        ...(cov.loadings ? [
          ...cov.loadings[0].map((_, f) => ({ key: (row) => cov.loadings[row.i][f], label: `λ${f + 1}`, num: true, format: (v) => fmt(v) })),
          { key: (row) => cov.specific[row.i], label: "ψ", num: true, format: (v) => fmt(v) },
        ] : []),
      ], variances.map((d, i) => ({ ...d, i }))),
    }));
    if (cov.loadings) {
      const k = cov.loadings[0].length;
      for (let f = 0; f < k; f++) {
        grid.append(figure({
          title: `Loadings on factor ${f + 1}`,
          subtitle: "How strongly each level expresses the common genetic factor (sign is arbitrary)",
          chart: (host, width) => hbars(host, width, { items: lv.map((l, i) => ({ label: l, value: cov.loadings[i][f] ?? 0 })), xTitle: `λ${f + 1}` }),
          table: () => dataTable([{ key: "label", label: cov.factor }, { key: "value", label: `λ${f + 1}`, num: true, format: (v) => fmt(v) }], lv.map((l, i) => ({ label: l, value: cov.loadings[i][f] }))),
        }));
      }
    }
    // Reaction norms: genotype BLUPs across levels, top three highlighted.
    const block = r.fit.random_effects.find((b) => b.term === cov.term);
    const meta = termMeta(r, cov.term);
    if (block && meta && meta.columns.length === 2) {
      const order = cov.loadings && cov.loadings[0].length === 1 ? lv.map((l, i) => [l, cov.loadings[i][0] ?? 0]).sort((a, b) => a[1] - b[1]).map(([l]) => l) : lv;
      const pos = new Map(order.map((l, i) => [l, i]));
      const byInner = new Map();
      for (const e of block.effects) {
        const outer = lv.find((l) => e.level.startsWith(`${l}:`));
        if (outer === undefined) continue;
        const inner = e.level.slice(outer.length + 1);
        if (!byInner.has(inner)) byInner.set(inner, new Array(nl).fill(null));
        byInner.get(inner)[pos.get(outer)] = e.estimate;
      }
      const entries = [...byInner.entries()].map(([name, ys]) => ({ name, ys, mean: ys.reduce((a, v) => a + (v ?? 0), 0) / nl }));
      entries.sort((a, b) => b.mean - a.mean);
      const top = entries.slice(0, 3);
      const topNames = new Set(top.map((e) => e.name));
      const toSeries = (e, color, muted) => ({ name: e.name, color, muted, points: e.ys.map((y, i) => ({ x: i, y })) });
      const colors = top.map((_, i) => seriesHex(i));
      out.append(grid);
      out.append(figure({
        title: `${meta.columns[1]} performance across ${meta.columns[0]}`,
        subtitle: `Predicted ${meta.columns[0]}-specific effects for ${entries.length} ${meta.columns[1]} levels; the three best on average are highlighted${cov.loadings && cov.loadings[0].length === 1 ? `; ${meta.columns[0]} ordered by loading` : ""}`,
        legend: legendItems(top.map((e, i) => ({ label: e.name, color: colors[i], line: true }))),
        chart: (host, width) => lineChart(host, width, {
          series: [...entries.filter((e) => !topNames.has(e.name)).map((e) => toSeries(e, null, true)), ...top.map((e, i) => toSeries(e, colors[i], false))],
          xLabels: order, xTitle: meta.columns[0], yTitle: "Predicted effect", height: 340,
        }),
        table: () => dataTable([
          { key: "name", label: meta.columns[1] },
          { key: "mean", label: "Mean", num: true, format: (v) => fmt(v) },
          ...order.map((l, i) => ({ key: (row) => row.ys[i], label: l, num: true, format: (v) => fmt(v) })),
        ], entries),
      }));
      continue;
    }
    out.append(grid);
  }
  return out;
}

function convergenceTab(fitState) {
  const hist = fitState.result.fit.history;
  const comps = fitState.result.fit.variance_components;
  const names = comps.flatMap((c) => c.parameters.map((p) => `${c.name} ${paramName(p)}`));
  return h("div", { class: "stack" },
    figure({
      title: "REML log-likelihood by iteration",
      subtitle: `${fitState.result.fit.converged ? "Converged" : "Did not converge"} after ${fitState.result.fit.n_iterations} iterations`,
      chart: (host, width) => lineChart(host, width, {
        series: [{ name: "log L", color: seriesHex(0), points: hist.map((it) => ({ x: it.iteration, y: it.log_likelihood })) }],
        xTitle: "Iteration", yTitle: "Log-likelihood", integerX: true, endLabels: false, format: (v) => fixed(v, 4),
      }),
      table: () => dataTable([
        { key: "iteration", label: "Iteration", num: true, format: int },
        { key: "log_likelihood", label: "log L", num: true, format: (v) => fixed(v, 5) },
        { key: "change", label: "Rel. change", num: true, format: (v) => (isNum(v) ? v.toExponential(2) : "—") },
        ...names.map((n, k) => ({ key: (it) => it.variance_params[k], label: n, num: true, format: (v) => fmt(v) })),
      ], hist),
    }));
}

function cvTab(fitState) {
  const cv = fitState.result.cv;
  const pred = cv.folds.flatMap((f) => f.predicted.map((p, i) => ({ fold: f.fold, x: p, y: f.observed[i] })));
  const n = pred.length;
  const mx = pred.reduce((a, p) => a + p.x, 0) / n, my = pred.reduce((a, p) => a + p.y, 0) / n;
  const sxx = pred.reduce((a, p) => a + (p.x - mx) ** 2, 0);
  const slope = sxx > 0 ? pred.reduce((a, p) => a + (p.x - mx) * (p.y - my), 0) / sxx : 0;
  return h("div", { class: "stack" },
    h("div", { class: "kpis" },
      tile("Prediction accuracy", fixed(cv.accuracy, 3), "correlation of predicted and observed", { meter: cv.accuracy }),
      tile("Bias (slope)", fixed(cv.bias, 3), "1 = unbiased; < 1 = over-dispersed predictions"),
      tile("MSEP", fmt(cv.msep), "mean squared error of prediction"),
      tile("MAE", fmt(cv.mae), `${cv.folds.length} folds, seed ${fitState.options.seed ?? 1}`)),
    h("div", { class: "grid-2" },
      figure({
        title: "Observed against predicted (held-out records)",
        subtitle: "Each fold is predicted from a model refitted without it; line: least-squares fit",
        chart: (host, width) => scatter(host, width, {
          points: pred.map((p) => ({ x: p.x, y: p.y, tip: { title: `Fold ${p.fold + 1}`, rows: [{ value: fmt(p.y), label: "observed" }, { value: fmt(p.x), label: "predicted" }] } })),
          xTitle: "Predicted", yTitle: "Observed", ref: { slope, intercept: my - slope * mx }, height: 320,
        }),
        table: () => dataTable([
          { key: "fold", label: "Fold", num: true, format: (v) => int(v + 1) },
          { key: "x", label: "Predicted", num: true, format: (v) => fmt(v) },
          { key: "y", label: "Observed", num: true, format: (v) => fmt(v) },
        ], pred),
      }),
      figure({
        title: "Accuracy by fold",
        chart: (host, width) => columnChart(host, width, {
          items: cv.folds.map((f) => ({ label: `Fold ${f.fold + 1}`, value: f.accuracy, tip: { title: `Fold ${f.fold + 1}`, rows: [{ value: fixed(f.accuracy, 3), label: "accuracy" }, { value: fmt(f.msep), label: "MSEP" }, { value: int(f.n_validation), label: "records" }] } })),
          yTitle: "Accuracy", height: 320, format: (v) => fixed(v, 3),
        }),
        table: () => dataTable([
          { key: "fold", label: "Fold", num: true, format: (v) => int(v + 1) },
          { key: "n_validation", label: "Records", num: true, format: int },
          { key: "accuracy", label: "Accuracy", num: true, format: (v) => fixed(v, 3) },
          { key: "msep", label: "MSEP", num: true, format: (v) => fmt(v) },
        ], cv.folds),
      })));
}

// ------------------------------------------------------------ main

function tabsFor(fitState) {
  const r = fitState.result;
  const tabs = [["overview", "Overview"]];
  if (r.fit.random_effects.length) tabs.push(["effects", r.pedigree ? "Breeding values" : "Random effects"]);
  tabs.push(["diagnostics", "Residuals"]);
  if (r.grid) tabs.push(["field", "Field map"]);
  if (r.covariances.length) tabs.push(["gxe", r.covariances.some((c) => c.structure.startsWith("fa")) ? "G×E" : "Covariances"]);
  tabs.push(["convergence", "Convergence"]);
  if (r.cv) tabs.push(["cv", "Cross-validation"]);
  tabs.push(["data", "Data"]);
  return tabs;
}

function formulaText(fitState) {
  const m = fitState.model;
  const rand = m.random.filter((t) => t.a).map((t) => `(${termString(t)})`);
  const res = residualString(m.residual);
  return `${m.response} ~ ${[fixedFormula(m), ...rand].join(" + ")}${res ? `,  R = ${res}` : ""}`;
}

function exportResults(fitState) {
  const payload = {
    generated_by: "OpenBLUP Studio",
    data: fitState.dataName,
    pedigree: fitState.pedigreeName,
    model: fitState.model,
    options: fitState.options,
    results: fitState.result,
  };
  download(`${fitState.dataName.replace(/\.[^.]+$/, "")}_openblup.json`, JSON.stringify(payload, null, 2), "application/json");
}

export function renderMain(root, state, actions) {
  clear(root);
  root.classList.toggle("busy", state.busy);
  const fitState = state.fit;
  if (!fitState) {
    if (state.busy) root.append(h("div", { class: "loading-bar" }));
    root.append(...landing(state, actions));
    return;
  }
  const r = fitState.result;
  if (state.busy) root.append(h("div", { class: "loading-bar" }));
  const wrap = h("div", { class: "results stack" });
  root.append(wrap);

  wrap.append(h("section", { class: "card results-head" },
    h("h2", { text: "Results" }),
    h("code", { class: "formula", text: formulaText(fitState) }),
    h("div", { class: "actions" },
      h("button", { class: "btn", type: "button", on: { click: () => exportResults(fitState) } }, icon(ICONS.download), "Export JSON"))));

  if (state.error) wrap.append(callout("error", [state.error]));
  if (r.warnings.length) wrap.append(callout("warning", r.warnings));
  if (r.notes.length) wrap.append(callout("note", r.notes));

  const ratio = varianceRatio(fitState);
  const kpis = h("div", { class: "kpis" });
  if (ratio) kpis.append(tile(ratio.label, fixed(ratio.value, 3), ratio.sub, { meter: ratio.value, cls: "hero-tile" }));
  kpis.append(
    tile("REML log-likelihood", fixed(r.fit.log_likelihood, 2), `AIC ${fixed(r.aic, 1)} · BIC ${fixed(r.bic, 1)}`),
    tile("Iterations", int(r.fit.n_iterations), h("span", { class: `sub ${r.fit.converged ? "good" : "bad"}` }, icon(r.fit.converged ? ICONS.check : ICONS.x, 14), r.fit.converged ? "converged" : "not converged")),
    tile("Records", int(r.model.n_obs), `${r.model.n_fixed} fixed · ${r.model.n_variance_params} variance parameters`),
    tile("Fit time", duration(fitState.ms), r.engine === "sparse" ? "sparse Cholesky + Takahashi" : "general engine (dense C)"));
  if (r.cv) kpis.append(tile("CV accuracy", fixed(r.cv.accuracy, 3), `${r.cv.folds.length}-fold`, { meter: r.cv.accuracy }));
  wrap.append(kpis);

  const tabs = tabsFor(fitState);
  if (!tabs.some(([id]) => id === state.tab)) state.tab = "overview";
  wrap.append(h("div", { class: "tabs", attrs: { role: "tablist", "aria-label": "Result views" } }, tabs.map(([id, label]) => h("button", {
    type: "button", text: label,
    attrs: { role: "tab", "aria-selected": String(id === state.tab), id: `tab-${id}` },
    on: { click: () => actions.setTab(id) },
  }))));

  const panel = h("div", { attrs: { role: "tabpanel", "aria-labelledby": `tab-${state.tab}` } });
  const view = (state.view[state.tab] ??= {});
  const renderers = {
    overview: () => overviewTab(fitState, actions),
    effects: () => effectsTab(fitState, view, actions),
    diagnostics: () => diagnosticsTab(fitState),
    field: () => fieldTab(fitState),
    gxe: () => gxeTab(fitState),
    convergence: () => convergenceTab(fitState),
    cv: () => cvTab(fitState),
    data: () => dataView(state),
  };
  panel.append(renderers[state.tab]());
  wrap.append(panel);
  wrap.append(h("p", { class: "foot" }, "Computed locally by the OpenBLUP engine (Rust → WebAssembly). ", h("a", { href: "https://github.com/jakobrichert/openblup", target: "_blank", rel: "noopener", text: "Source & docs" })));
}
