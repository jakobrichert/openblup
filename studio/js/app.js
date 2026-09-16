// OpenBLUP Studio: application state and the setup sidebar.

import { h, clear, icon, ICONS, segmented, select } from "./dom.js";
import { engineVersion, inspectCsv, fitModel } from "./engine.js";
import { EXAMPLES, STRUCTURES, INNER_STRUCTURES, termString, fixedFormula, fitRequest, cliCommand, pythonSnippet } from "./examples.js";
import { renderMain } from "./results.js";
import { redrawAll, hideTip } from "./charts.js";
import { duration } from "./format.js";

const state = {
  engine: { ready: false, version: null, error: null },
  dataset: null, // {name, csv, info}
  pedigree: null, // {name, csv}
  exampleId: null,
  pendingTab: null,
  model: null,
  options: { algorithm: "ai", ddf: "satterthwaite", cv: 0, seed: 1, maxIter: 50 },
  busy: false,
  startedAt: 0,
  error: null,
  fit: null,
  tab: "overview",
  reproduce: "cli",
  view: {},
};

const $ = (sel) => document.querySelector(sel);

// ------------------------------------------------------------ helpers

const columns = () => state.dataset?.info.columns ?? [];
const column = (name) => columns().find((c) => c.name === name);
const numericColumns = () => columns().filter((c) => c.kind === "numeric");

function defaultModel(info) {
  const cols = info.columns;
  const numeric = cols.filter((c) => c.kind === "numeric");
  const response = [...numeric].reverse().find((c) => !c.integer_like) || numeric[numeric.length - 1];
  const factors = cols.filter((c) => c !== response && c.kind === "factor" && c.n_distinct > 1 && c.n_distinct < info.n_rows);
  const random = factors.length ? [{ a: factors[factors.length > 1 ? 1 : 0].name, as: "idv", b: "", bs: "" }] : [];
  const lower = (n) => cols.find((c) => c.name.toLowerCase() === n);
  const row = lower("row"), col = lower("col") || lower("column");
  return {
    response: response?.name ?? "",
    intercept: true,
    fixed: [],
    factors: [],
    random,
    residual: { kind: "iid", row: row?.name ?? "", col: col?.name ?? "", rowStruct: "ar1", colStruct: "ar1c", text: "" },
    pedigreeTerm: "",
  };
}

function withResidualDefaults(model) {
  const d = defaultModel(state.dataset.info).residual;
  model.residual = { ...d, ...model.residual };
  return model;
}

async function fetchText(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Could not load ${path} (${res.status})`);
  return res.text();
}

// ------------------------------------------------------------ actions

async function setDataset(name, csv) {
  const info = await inspectCsv(csv);
  if (!info.n_rows) throw new Error(`${name} has no data rows`);
  state.dataset = { name, csv, info };
}

async function loadExample(ex, { run = true } = {}) {
  try {
    state.error = null;
    const [csv, ped] = await Promise.all([
      fetchText(`examples/${ex.data}`),
      ex.pedigree ? fetchText(`examples/${ex.pedigree}`) : null,
    ]);
    await setDataset(ex.data, csv);
    state.pedigree = ped ? { name: ex.pedigree, csv: ped } : null;
    state.exampleId = ex.id;
    state.model = withResidualDefaults(structuredClone(ex.model));
    state.options = { algorithm: "ai", ddf: "containment", cv: 0, seed: 1, maxIter: 50, ...ex.options };
    state.fit = null;
    state.pendingTab = ex.tab;
    state.view = {};
    render();
    if (run) await runFit();
  } catch (err) {
    state.error = err.message;
    render();
  }
}

async function loadFile(file, kind) {
  if (!file) return;
  if (file.size > 50 * 1024 * 1024) {
    state.error = `${file.name} is larger than 50 MB; use the CLI for large evaluations.`;
    render();
    return;
  }
  try {
    const text = await file.text();
    state.error = null;
    if (kind === "pedigree") {
      const first = text.split(/\r?\n/, 1)[0].toLowerCase().split(",").map((s) => s.trim());
      if (!["animal", "sire", "dam"].every((c) => first.includes(c))) {
        throw new Error("A pedigree needs the columns animal, sire and dam.");
      }
      state.pedigree = { name: file.name, csv: text };
      if (state.model && !state.model.pedigreeTerm && state.model.random[0]) {
        state.model.pedigreeTerm = state.model.random[0].b || state.model.random[0].a;
      }
    } else {
      await setDataset(file.name, text);
      state.exampleId = null;
      state.model = defaultModel(state.dataset.info);
      state.fit = null;
      state.pendingTab = "data";
      state.tab = "data";
      state.view = {};
    }
  } catch (err) {
    state.error = err.message;
  }
  render();
}

function validate() {
  const m = state.model;
  if (!state.dataset) return "Load a dataset first.";
  if (!m.response) return "Choose a numeric response column.";
  if (m.random.some((t) => !t.a)) return "Every random term needs a factor.";
  if (m.random.some((t) => t.b && t.b === t.a)) return "An interaction needs two different factors.";
  if (state.pedigree && !m.random.length) return "A pedigree needs at least one random term.";
  if (m.residual.kind === "spatial" && (!m.residual.row || !m.residual.col || m.residual.row === m.residual.col)) {
    return "A spatial residual needs two different row and column factors.";
  }
  return null;
}

async function runFit() {
  if (state.busy) return;
  const problem = validate();
  if (problem) {
    state.error = problem;
    render();
    return;
  }
  const request = fitRequest({ data: state.dataset.csv, pedigree: state.pedigree?.csv, model: state.model, options: state.options });
  state.busy = true;
  state.error = null;
  state.startedAt = performance.now();
  render();
  const ticker = setInterval(renderStatus, 100);
  try {
    const { result, ms } = await fitModel(request);
    state.fit = {
      result,
      ms,
      dataName: state.dataset.name,
      pedigreeName: state.pedigree?.name ?? null,
      model: structuredClone(state.model),
      options: { ...state.options },
      columns: columns(),
      request,
    };
    const tab = state.pendingTab;
    state.pendingTab = null;
    state.view = {};
    if (tab) state.tab = tab;
    else if (state.tab === "data") state.tab = "overview";
  } catch (err) {
    state.error = err.message;
  } finally {
    clearInterval(ticker);
    state.busy = false;
    render();
  }
}

function update(fn) {
  fn(state.model);
  render();
}

// ------------------------------------------------------------ sidebar

function panelHead(step, title, aside) {
  return h("div", { class: "panel-head" },
    h("span", { class: "step", text: String(step) }),
    h("h2", { text: title }),
    aside ? h("span", { class: "aside", text: aside }) : null);
}

function dropzone(kind) {
  const current = kind === "data" ? state.dataset : state.pedigree;
  const input = h("input", {
    type: "file",
    accept: ".csv,.txt,text/csv",
    attrs: { "aria-label": kind === "data" ? "Upload data CSV" : "Upload pedigree CSV" },
    on: { change: (e) => loadFile(e.target.files[0], kind) },
  });
  const zone = h("label", { class: "dropzone", dataset: { focusKey: `drop-${kind}` } },
    icon(kind === "data" ? ICONS.upload : ICONS.tree, 18),
    h("span", { class: "grow" },
      current
        ? [h("span", { class: "file-name", text: current.name }), h("span", { class: "hint", text: kind === "data" ? " · replace" : " · pedigree" })]
        : kind === "data" ? "Upload your own CSV" : "Add a pedigree CSV (animal, sire, dam)"),
    input);
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("over");
    loadFile(e.dataTransfer.files[0], kind);
  });
  if (kind === "pedigree" && current) {
    return h("div", { class: "row" }, zone, h("button", {
      class: "link-btn danger", type: "button", text: "Remove",
      on: { click: () => { state.pedigree = null; render(); } },
    }));
  }
  return zone;
}

function dataPanel() {
  const ds = state.dataset;
  const panel = h("section", { class: "panel", attrs: { "aria-label": "Data" } },
    panelHead(1, "Data", "runs locally"),
    h("div", { class: "examples" }, EXAMPLES.map((ex) => h("button", {
      class: "example", type: "button",
      dataset: { focusKey: `example-${ex.id}` },
      attrs: { "aria-pressed": String(state.exampleId === ex.id) },
      on: { click: () => loadExample(ex) },
    }, icon(ex.icon, 18), h("strong", { text: ex.title }), h("span", { text: ex.blurb })))),
    h("div", { class: "stack", style: { gap: "8px", marginTop: "12px" } }, dropzone("data"), ds ? dropzone("pedigree") : null));
  if (ds) {
    const info = ds.info;
    panel.append(h("div", { class: "dataset" },
      h("div", { class: "dataset-title" },
        h("strong", { text: ds.name }),
        h("span", { text: `${info.n_rows.toLocaleString()} rows · ${info.columns.length} columns` })),
      h("div", { class: "chips" }, info.columns.map((c) => h("span", { class: "chip static", attrs: { title: c.kind === "factor" ? `${c.n_distinct} levels` : `${c.n_distinct} distinct values` } },
        h("span", { class: "kind", text: c.kind === "factor" ? "Aa" : "#" }),
        c.name,
        h("span", { class: "count", text: c.kind === "factor" ? String(c.n_distinct) : c.integer_like && c.n_distinct <= 200 ? `${c.n_distinct}×` : "" }),
        c.n_missing ? h("span", { class: "warn", text: `${c.n_missing} NA` }) : null)))));
  }
  return panel;
}

function termEditor(t, i) {
  const m = state.model;
  const factorOptions = columns().filter((c) => c.name !== m.response).map((c) => [c.name, c.name]);
  const outer = select([["", "Choose factor…"], ...factorOptions], t.a, (v) => update(() => { t.a = v; }), { attrs: { "aria-label": `Random term ${i + 1} factor` }, dataset: { focusKey: `term-${i}-a` } });
  const outerStruct = select(STRUCTURES, t.as || "idv", (v) => update(() => { t.as = v; }), { attrs: { "aria-label": `Random term ${i + 1} structure` }, dataset: { focusKey: `term-${i}-as` } });
  const inner = select([["", "— no interaction —"], ...factorOptions.filter(([n]) => n !== t.a)], t.b, (v) => update(() => {
    t.b = v;
    if (!v) t.bs = "";
  }), { attrs: { "aria-label": `Random term ${i + 1} interaction factor` }, dataset: { focusKey: `term-${i}-b` } });
  const innerStruct = t.b ? select(INNER_STRUCTURES, t.bs || "", (v) => update(() => { t.bs = v; }), { attrs: { "aria-label": `Random term ${i + 1} interaction structure` }, dataset: { focusKey: `term-${i}-bs` } }) : null;
  return h("div", { class: "term" },
    h("div", { class: "row" }, outer, outerStruct),
    h("div", { class: "row" }, h("span", { class: "times", text: "×" }), inner, innerStruct),
    h("div", { class: "term-foot" },
      h("code", { text: t.a ? termString(t) : "…" }),
      h("button", {
        class: "link-btn danger", type: "button", text: "Remove",
        dataset: { focusKey: `term-${i}-remove` },
        on: { click: () => update((mm) => { mm.random.splice(i, 1); }) },
      })));
}

function modelPanel() {
  const m = state.model;
  const panel = h("section", { class: "panel", attrs: { "aria-label": "Model" } }, panelHead(2, "Model"));
  const numeric = numericColumns();

  panel.append(h("div", { class: "field" },
    h("label", { text: "Response", attrs: { for: "response" } }),
    select([["", "Choose…"], ...numeric.map((c) => [c.name, c.name])], m.response, (v) => update((mm) => {
      mm.response = v;
      mm.fixed = mm.fixed.filter((f) => f !== v);
    }), { id: "response" })));

  const candidates = columns().filter((c) => c.name !== m.response);
  const toggleFixed = (name) => update((mm) => {
    if (mm.fixed.includes(name)) mm.fixed = mm.fixed.filter((f) => f !== name);
    else {
      mm.fixed = [...mm.fixed, name];
      const c = column(name);
      if (c.kind === "numeric" && c.integer_like && c.n_distinct <= 30 && !mm.factors.includes(name)) mm.factors = [...mm.factors, name];
    }
  });
  panel.append(h("div", { class: "field" },
    h("span", { class: "field-label", text: "Fixed effects" }),
    h("div", { class: "chips" },
      h("button", {
        class: "chip", type: "button", text: "Intercept",
        dataset: { focusKey: "fixed-mu" },
        attrs: { "aria-pressed": String(m.intercept) },
        on: { click: () => update((mm) => { mm.intercept = !mm.intercept; }) },
      }),
      candidates.map((c) => h("button", {
        class: "chip", type: "button",
        dataset: { focusKey: `fixed-${c.name}` },
        attrs: { "aria-pressed": String(m.fixed.includes(c.name)) },
        on: { click: () => toggleFixed(c.name) },
      }, h("span", { class: "kind", text: c.kind === "factor" ? "Aa" : "#" }), c.name))),
    h("span", { class: "hint", text: `Formula: ${m.response || "y"} ~ ${fixedFormula(m)}` })));

  const numericFixed = m.fixed.filter((f) => column(f)?.kind === "numeric");
  if (numericFixed.length) {
    panel.append(h("div", { class: "field" },
      h("span", { class: "field-label", text: "Numeric fixed effects: treat as factor?" }),
      h("div", { class: "chips" }, numericFixed.map((f) => h("button", {
        class: "chip", type: "button",
        dataset: { focusKey: `factor-${f}` },
        attrs: { "aria-pressed": String(m.factors.includes(f)) },
        on: { click: () => update((mm) => { mm.factors = mm.factors.includes(f) ? mm.factors.filter((x) => x !== f) : [...mm.factors, f]; }) },
      }, `${f} ${m.factors.includes(f) ? "(factor)" : "(covariate)"}`)))));
  }

  const terms = h("div", { class: "field" }, h("span", { class: "field-label", text: "Random terms" }));
  if (m.random.length) terms.append(h("div", {}, m.random.map(termEditor)));
  else terms.append(h("span", { class: "hint", text: "No random terms: the model is fitted with a single residual variance." }));
  terms.append(h("div", {}, h("button", {
    class: "link-btn", type: "button", text: "+ Add random term",
    dataset: { focusKey: "add-term" },
    on: { click: () => update((mm) => {
      const used = new Set([mm.response, ...mm.random.map((t) => t.a)]);
      const next = columns().find((c) => c.kind === "factor" && !used.has(c.name));
      mm.random.push({ a: next?.name ?? "", as: "idv", b: "", bs: "" });
    }) },
  })));
  panel.append(terms);

  if (state.pedigree) {
    const pedFactors = [...new Set(m.random.flatMap((t) => (t.b ? [t.b] : [t.a])).filter(Boolean))];
    panel.append(h("div", { class: "field" },
      h("label", { text: "Pedigree (A-inverse) applies to", attrs: { for: "pedterm" } }),
      select([["", "First random term"], ...pedFactors.map((f) => [f, f])], m.pedigreeTerm, (v) => update((mm) => { mm.pedigreeTerm = v; }), { id: "pedterm" }),
      h("span", { class: "hint", text: `${state.pedigree.name}: animals without records still get breeding values.` })));
  }

  const r = m.residual;
  const res = h("div", { class: "field" },
    h("span", { class: "field-label", text: "Residual" }),
    segmented([["iid", "IID"], ["spatial", "Spatial"], ["custom", "Custom"]], r.kind, (v) => update(() => { r.kind = v; }), { block: true, label: "Residual structure" }));
  if (r.kind === "spatial") {
    const opts = columns().filter((c) => c.name !== m.response).map((c) => [c.name, c.name]);
    const rs = [["ar1", "AR1"], ["ar1c", "AR1 corr."], ["idv", "IID"]];
    res.append(
      h("div", { class: "row" },
        h("span", { class: "hint", style: { width: "34px" }, text: "Rows" }),
        select([["", "Choose…"], ...opts], r.row, (v) => update(() => { r.row = v; }), { class: "grow", attrs: { "aria-label": "Row factor" }, dataset: { focusKey: "res-row" } }),
        select(rs, r.rowStruct, (v) => update(() => { r.rowStruct = v; }), { attrs: { "aria-label": "Row structure" }, dataset: { focusKey: "res-rows" } })),
      h("div", { class: "row" },
        h("span", { class: "hint", style: { width: "34px" }, text: "Cols" }),
        select([["", "Choose…"], ...opts], r.col, (v) => update(() => { r.col = v; }), { class: "grow", attrs: { "aria-label": "Column factor" }, dataset: { focusKey: "res-col" } }),
        select(rs, r.colStruct, (v) => update(() => { r.colStruct = v; }), { attrs: { "aria-label": "Column structure" }, dataset: { focusKey: "res-cols" } })),
      h("span", { class: "hint", text: "Separable AR1 × AR1 field-trend residual over the row × column grid; missing plots are allowed." }));
  } else if (r.kind === "custom") {
    res.append(h("input", {
      type: "text", value: r.text, placeholder: "ar1   or   trait:us*unit:fixed",
      attrs: { "aria-label": "Custom residual term", spellcheck: "false" },
      class: "mono",
      dataset: { focusKey: "res-text" },
      on: { change: (e) => update(() => { r.text = e.target.value; }) },
    }), h("span", { class: "hint", text: "A structure in data order, or factor:structure*factor:structure over a grid." }));
  }
  panel.append(res);
  return panel;
}

function runPanel() {
  const o = state.options;
  const set = (key) => (v) => { o[key] = v; render(); };
  const panel = h("section", { class: "panel", attrs: { "aria-label": "Fit" } },
    panelHead(3, "Fit"),
    h("div", { class: "options" },
      h("div", { class: "field" }, h("label", { text: "Algorithm", attrs: { for: "opt-alg" } }),
        select([["ai", "AI-REML"], ["em", "EM-REML"]], o.algorithm, set("algorithm"), { id: "opt-alg" })),
      h("div", { class: "field" }, h("label", { text: "Denominator df", attrs: { for: "opt-ddf" } }),
        select([["containment", "Containment"], ["satterthwaite", "Satterthwaite"], ["kenward-roger", "Kenward-Roger"]], o.ddf, set("ddf"), { id: "opt-ddf" })),
      h("div", { class: "field" }, h("label", { text: "Cross-validation", attrs: { for: "opt-cv" } }),
        select([["0", "Off"], ["3", "3-fold"], ["5", "5-fold"], ["10", "10-fold"]], String(o.cv || 0), (v) => set("cv")(Number(v)), { id: "opt-cv" })),
      h("div", { class: "field" }, h("label", { text: "Max iterations", attrs: { for: "opt-iter" } }),
        h("input", { id: "opt-iter", type: "number", min: 1, max: 1000, value: o.maxIter, on: { change: (e) => set("maxIter")(Number(e.target.value) || 50) } }))),
    h("div", { style: { marginTop: "14px" } },
      h("button", {
        class: "fit-btn", type: "button", disabled: state.busy || !state.engine.ready,
        dataset: { focusKey: "fit" },
        on: { click: runFit },
      }, state.busy ? h("span", { class: "spinner" }) : icon(ICONS.play, 16), state.busy ? "Fitting…" : "Fit model", state.busy ? null : h("kbd", { text: navigator.platform.includes("Mac") ? "⌘ ↵" : "Ctrl ↵" }))),
    h("div", { class: "status", id: "fit-status", attrs: { role: "status", "aria-live": "polite" } }));
  return panel;
}

function renderStatus() {
  const el = $("#fit-status");
  if (!el) return;
  el.className = "status";
  if (state.busy) el.textContent = `Running REML… ${duration(performance.now() - state.startedAt)}`;
  else if (state.error) { el.className = "status error"; el.textContent = state.error; }
  else if (state.fit) {
    const r = state.fit.result;
    el.textContent = `${r.fit.converged ? "Converged" : "Stopped"} after ${r.fit.n_iterations} iterations in ${duration(state.fit.ms)} · ${r.engine === "sparse" ? "sparse MME" : "general engine"}`;
  } else if (!state.engine.ready) el.textContent = state.engine.error ? `Engine unavailable: ${state.engine.error}` : "Loading the WebAssembly engine…";
  else el.textContent = "";
}

function reproducePanel() {
  const args = { dataName: state.dataset.name, pedigreeName: state.pedigree?.name, model: state.model, options: state.options };
  const code = state.reproduce === "cli" ? cliCommand(args) : pythonSnippet(args);
  const copy = h("button", { class: "copy-btn", type: "button", text: "Copy" });
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(code.replace(/ \\\n\s+/g, " "));
      copy.textContent = "Copied";
    } catch (_) {
      copy.textContent = "Select & copy";
    }
    setTimeout(() => { copy.textContent = "Copy"; }, 1500);
  });
  return h("section", { class: "panel", attrs: { "aria-label": "Reproduce" } },
    h("div", { class: "panel-head" },
      h("h2", { text: "Reproduce" }),
      h("span", { class: "aside" }, segmented([["cli", "CLI"], ["python", "Python"]], state.reproduce, (v) => { state.reproduce = v; render(); }, { label: "Code language" }))),
    h("pre", { class: "code" }, h("code", { text: code }), copy));
}

function renderSidebar() {
  const root = $("#sidebar");
  const focusKey = document.activeElement?.dataset?.focusKey || (document.activeElement?.id && root.contains(document.activeElement) ? `#${document.activeElement.id}` : null);
  const scroll = root.scrollTop;
  clear(root);
  root.append(dataPanel());
  if (state.dataset && state.model) root.append(modelPanel(), runPanel(), reproducePanel());
  else if (state.error) root.append(h("div", { class: "callout error" }, icon(ICONS.alert), h("span", { text: state.error })));
  root.scrollTop = scroll;
  if (focusKey) {
    const target = focusKey.startsWith("#") ? root.querySelector(focusKey) : root.querySelector(`[data-focus-key="${CSS.escape(focusKey)}"]`);
    target?.focus({ preventScroll: true });
  }
  renderStatus();
}

function renderBadge() {
  const badge = $("#engine-badge");
  clear(badge);
  badge.className = `badge${state.engine.ready ? " ready" : state.engine.error ? " error" : ""}`;
  badge.append(h("span", { class: "dot" }), h("span", {
    text: state.engine.ready ? `Engine v${state.engine.version} · WebAssembly` : state.engine.error ? "Engine failed to load" : "Loading engine…",
  }));
}

function render() {
  hideTip();
  renderBadge();
  renderSidebar();
  renderMain($("#main"), state, {
    setTab: (tab) => { state.tab = tab; renderMainOnly(); },
    loadExample: (id) => loadExample(EXAMPLES.find((e) => e.id === id)),
    rerender: renderMainOnly,
  });
}

function renderMainOnly() {
  hideTip();
  renderMain($("#main"), state, {
    setTab: (tab) => { state.tab = tab; renderMainOnly(); },
    loadExample: (id) => loadExample(EXAMPLES.find((e) => e.id === id)),
    rerender: renderMainOnly,
  });
}

// ------------------------------------------------------------ theme

function effectiveTheme() {
  const t = document.documentElement.dataset.theme;
  if (t) return t;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function renderThemeButton() {
  const btn = $("#theme-toggle");
  clear(btn);
  btn.append(icon(effectiveTheme() === "dark" ? ICONS.sun : ICONS.moon, 16));
  btn.setAttribute("aria-label", effectiveTheme() === "dark" ? "Switch to light theme" : "Switch to dark theme");
}

function setupTheme() {
  $("#theme-toggle").addEventListener("click", () => {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("openblup-theme", next); } catch (_) { /* storage unavailable */ }
    renderThemeButton();
    redrawAll($("#main"));
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    renderThemeButton();
    redrawAll($("#main"));
  });
  renderThemeButton();
}

// ------------------------------------------------------------ boot

async function boot() {
  setupTheme();
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && state.dataset) {
      e.preventDefault();
      runFit();
    }
  });
  window.addEventListener("scroll", hideTip, { passive: true });
  render();
  try {
    state.engine.version = await engineVersion();
    state.engine.ready = true;
  } catch (err) {
    state.engine.error = err.message;
  }
  render();

  const params = new URLSearchParams(location.search);
  const ex = EXAMPLES.find((e) => e.id === params.get("example"));
  if (ex && state.engine.ready) {
    await loadExample(ex, { run: params.get("run") !== "0" });
    const tab = params.get("tab");
    if (tab) { state.tab = tab; renderMainOnly(); }
  }
  document.documentElement.dataset.studioReady = "true";
}

boot();
