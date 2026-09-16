// Built-in example analyses (data files live in ../examples, copied next to
// the app by build.sh).

import { ICONS } from "./dom.js";

export const EXAMPLES = [
  {
    id: "spatial",
    title: "Spatial field trial",
    blurb: "60 lines × 4 reps, AR1 × AR1 residual",
    icon: ICONS.grid,
    data: "spatial_trial.csv",
    tab: "field",
    model: {
      response: "yield",
      intercept: true,
      fixed: ["rep"],
      factors: ["rep"],
      random: [{ a: "genotype", as: "idv", b: "", bs: "" }],
      residual: { kind: "spatial", row: "row", col: "col", rowStruct: "ar1", colStruct: "ar1c" },
    },
    options: { ddf: "satterthwaite" },
  },
  {
    id: "met",
    title: "Multi-environment trial",
    blurb: "30 genotypes × 4 sites, FA1 G×E",
    icon: ICONS.lines,
    data: "met_trial.csv",
    tab: "gxe",
    model: {
      response: "yield",
      intercept: true,
      fixed: ["env"],
      factors: [],
      random: [{ a: "env", as: "fa1", b: "genotype", bs: "" }],
      residual: { kind: "iid" },
    },
    options: {},
  },
  {
    id: "animal",
    title: "Animal model",
    blurb: "63-animal pedigree, h² and EBVs",
    icon: ICONS.tree,
    data: "animal_records.csv",
    pedigree: "animal_pedigree.csv",
    tab: "effects",
    model: {
      response: "weight",
      intercept: false,
      fixed: ["sex"],
      factors: [],
      random: [{ a: "animal", as: "idv", b: "", bs: "" }],
      residual: { kind: "iid" },
      pedigreeTerm: "animal",
    },
    options: { ddf: "kenward-roger", cv: 5 },
  },
  {
    id: "rcbd",
    title: "Small RCBD",
    blurb: "6 genotypes × 3 reps, one missing plot",
    icon: ICONS.sprout,
    data: "field_trial.csv",
    tab: "overview",
    model: {
      response: "yield",
      intercept: true,
      fixed: ["rep"],
      factors: ["rep"],
      random: [{ a: "genotype", as: "idv", b: "", bs: "" }],
      residual: { kind: "iid" },
    },
    options: { ddf: "satterthwaite", cv: 3 },
  },
];

export const STRUCTURES = [
  ["idv", "IID (idv)"],
  ["diag", "Diagonal"],
  ["us", "Unstructured"],
  ["fa1", "Factor analytic 1"],
  ["fa2", "Factor analytic 2"],
  ["fa3", "Factor analytic 3"],
  ["ar1", "AR1"],
  ["ar1c", "AR1 correlation"],
];

export const INNER_STRUCTURES = [
  ["", "Identity"],
  ["ar1", "AR1"],
  ["ar1c", "AR1 correlation"],
];

/** `factor[:structure]` with the default structure omitted. */
function factorTerm(name, structure, defaultStructure) {
  return structure && structure !== defaultStructure ? `${name}:${structure}` : name;
}

export function termString(t) {
  const a = factorTerm(t.a, t.as, t.b ? "" : "idv");
  if (!t.b) return a;
  // In an interaction the outer structure is always written out.
  return `${t.a}:${t.as || "idv"}*${factorTerm(t.b, t.bs, "")}`;
}

export function residualString(r) {
  if (!r || r.kind === "iid") return null;
  if (r.kind === "spatial") return `${r.row}:${r.rowStruct || "ar1"}*${r.col}:${r.colStruct || "ar1c"}`;
  return (r.text || "").trim() || null;
}

export function fixedFormula(model) {
  const terms = [...(model.intercept ? ["mu"] : []), ...model.fixed];
  return terms.length ? terms.join(" + ") : "mu";
}

/** The engine request for a model (see `FitRequest` in crates/studio). */
export function fitRequest({ data, pedigree, model, options }) {
  return {
    data,
    pedigree: pedigree ?? null,
    spec: {
      response: model.response,
      fixed: fixedFormula(model),
      random: model.random.map(termString),
      residual: residualString(model.residual),
      factors: model.factors.filter((f) => model.fixed.includes(f)),
      pedigree_term: pedigree ? model.pedigreeTerm || null : null,
      max_iter: Number(options.maxIter) || 50,
    },
    algorithm: options.algorithm || "ai",
    ddf: options.ddf || "containment",
    cv_folds: options.cv ? Number(options.cv) : null,
    cv_seed: Number(options.seed) || 0,
  };
}

const quote = (s) => (/^[\w.:-]+$/.test(s) ? s : `"${s.replace(/(["\\$`])/g, "\\$1")}"`);

export function cliCommand({ dataName, pedigreeName, model, options }) {
  const parts = ["openblup fit", `--data ${quote(dataName)}`, `--response ${quote(model.response)}`, `--fixed ${quote(fixedFormula(model))}`];
  for (const f of model.factors.filter((f) => model.fixed.includes(f))) parts.push(`--factor ${quote(f)}`);
  for (const t of model.random) if (t.a) parts.push(`--random ${quote(termString(t))}`);
  const res = residualString(model.residual);
  if (res) parts.push(`--residual ${quote(res)}`);
  if (pedigreeName) {
    parts.push(`--pedigree ${quote(pedigreeName)}`);
    if (model.pedigreeTerm) parts.push(`--pedigree-term ${quote(model.pedigreeTerm)}`);
  }
  if (options.algorithm === "em") parts.push("--algorithm em");
  if (options.ddf && options.ddf !== "containment") parts.push(`--ddf ${options.ddf}`);
  if (options.cv) parts.push(`--cv ${options.cv}`, `--cv-seed ${options.seed ?? 1}`);
  if (options.maxIter && options.maxIter !== 50) parts.push(`--max-iter ${options.maxIter}`);
  return parts.join(" \\\n    ");
}

const py = (s) => JSON.stringify(s);

export function pythonSnippet({ dataName, pedigreeName, model, options }) {
  const lines = [`from openblup import MixedModel${pedigreeName ? ", Pedigree" : ""}`, ""];
  if (pedigreeName) lines.push(`ped = Pedigree.from_csv(${py(pedigreeName)})`);
  lines.push("model = MixedModel()", `model.load_csv(${py(dataName)})`);
  for (const f of model.factors.filter((f) => model.fixed.includes(f))) lines.push(`model.as_factor(${py(f)})`);
  lines.push(`model.set_response(${py(model.response)})`, `model.add_fixed(${py(fixedFormula(model))})`);
  for (const t of model.random) {
    if (!t.a) continue;
    const usesPed = pedigreeName && [t.a, t.b].includes(model.pedigreeTerm);
    if (!t.b) {
      if (usesPed) lines.push(`model.add_random_pedigree(${py(t.a)}, ped)`);
      else if (t.as && t.as !== "idv") lines.push(`model.add_random(${py(t.a)}, structure=${py(t.as)})`);
      else lines.push(`model.add_random(${py(t.a)})`);
    } else {
      const args = [py(t.a), py(t.b), `outer_structure=${py(t.as || "idv")}`];
      if (t.bs) args.push(`inner_structure=${py(t.bs)}`);
      if (usesPed && model.pedigreeTerm === t.b) args.push("pedigree=ped");
      lines.push(`model.add_random_interaction(${args.join(", ")})`);
    }
  }
  const r = model.residual;
  if (r && r.kind === "spatial") lines.push(`model.set_residual_interaction(${py(r.row)}, ${py(r.col)}, ${py(r.rowStruct || "ar1")}, ${py(r.colStruct || "ar1c")})`);
  else if (r && r.kind === "custom" && residualString(r)) {
    const res = residualString(r);
    const [row, col] = res.split("*").map((f) => f.split(":"));
    if (!col) lines.push(`model.set_residual(${py(res)})`);
    else lines.push(`model.set_residual_interaction(${py(row[0])}, ${py(col[0])}, ${py(row[1] || "idv")}, ${py(col[1] || "idv")})`);
  }
  lines.push("result = model.fit()", "print(result.summary())");
  if (options.ddf && options.ddf !== "containment") lines.push(`print(result.wald_tests(ddf=${py(options.ddf)}))`);
  if (options.cv) lines.push(`print(model.cross_validate(n_folds=${options.cv}, seed=${options.seed ?? 1})["accuracy"])`);
  return lines.join("\n");
}
