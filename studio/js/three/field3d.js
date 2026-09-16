// Spatial field landscape: plots as columns on the field, the AR1 x AR1
// residual as a trend surface, and the yields with that trend removed.

import { sequentialScale, divergingScale } from "../charts.js";
import { fmt, fixed } from "../format.js";
import { fillGaps, upsample } from "./stage.js";

export const FIELD_MODES = [
  ["raw", "Observed"],
  ["trend", "Field trend"],
  ["adjusted", "Trend removed"],
];

export function buildField(stage, fitState) {
  const { THREE, scene, sprite, colors } = stage;
  const r = fitState.result;
  const g = r.grid;
  const nr = g.row_levels.length;
  const nc = g.col_levels.length;
  const y = r.observations.y;
  const fitted = r.observations.fitted;
  const resid = y.map((v, i) => v - fitted[i]);
  const cells = g.cell_index.map((cell) => [Math.floor(cell / nc), cell % nc]);

  // Genotype effects of the (first) single-factor random term.
  const labels = r.observations.labels;
  const term = r.model.random_terms.find((t) => t.columns.length === 1 && labels[t.columns[0]]);
  const effect = new Map();
  if (term) {
    const block = r.fit.random_effects.find((b) => b.term === term.label);
    block?.effects.forEach((e) => effect.set(e.level, e.estimate));
  }
  const geno = term ? labels[term.columns[0]] : null;
  const genoEffect = y.map((_, i) => (geno ? effect.get(geno[i]) ?? 0 : 0));

  const unit = Math.min(8.5 / nc, 8.5 / nr);
  const xOf = (j) => (j - (nc - 1) / 2) * unit;
  const zOf = (i) => (i - (nr - 1) / 2) * unit;
  const lo = Math.min(...y, ...fitted);
  const hi = Math.max(...y, ...fitted);
  const BAR = 2.6;
  const barHeight = (v) => 0.12 + (BAR * (v - lo)) / Math.max(1e-9, hi - lo);
  const maxRes = Math.max(1e-9, ...resid.map(Math.abs));
  const maxGeno = Math.max(1e-9, ...genoEffect.map(Math.abs));
  const seq = sequentialScale([lo, hi]);
  const divRes = divergingScale(maxRes);
  const divGeno = divergingScale(maxGeno);

  const group = new THREE.Group();
  scene.add(group);

  // ---- ground ----
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(nc * unit + 3, nr * unit + 3),
    new THREE.MeshStandardMaterial({ color: stage.dark ? 0x131519 : 0xeef0ec, roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.01;
  ground.receiveShadow = true;
  group.add(ground);

  // ---- trend surface (built at full height, flattened with scale.y) ----
  const E = fillGaps(Array.from({ length: nr }, (_, i) => Array.from({ length: nc }, (_, j) => {
    const k = cells.findIndex(([a, b]) => a === i && b === j);
    return k >= 0 ? resid[k] : null;
  })));
  const K = Math.max(2, Math.min(8, Math.round(120 / Math.max(nr, nc))));
  const U = upsample(E, K);
  const ur = U.length, uc = U[0].length;
  const AMP = 1.25;
  const pos = new Float32Array(ur * uc * 3);
  const col = new Float32Array(ur * uc * 3);
  const c = new THREE.Color();
  for (let i = 0; i < ur; i++) {
    for (let j = 0; j < uc; j++) {
      const k = i * uc + j;
      const e = U[i][j];
      pos.set([xOf(j / K), AMP + (AMP * e) / maxRes, zOf(i / K)], k * 3);
      c.set(divRes(Math.max(-maxRes, Math.min(maxRes, e))));
      col.set([c.r, c.g, c.b], k * 3);
    }
  }
  const idx = [];
  for (let i = 0; i < ur - 1; i++) {
    for (let j = 0; j < uc - 1; j++) {
      const p = i * uc + j;
      idx.push(p, p + uc, p + 1, p + 1, p + uc, p + uc + 1);
    }
  }
  const sGeom = new THREE.BufferGeometry();
  sGeom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  sGeom.setAttribute("color", new THREE.BufferAttribute(col, 3));
  sGeom.setIndex(idx);
  sGeom.computeVertexNormals();
  const trend = new THREE.Mesh(sGeom, new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.5, metalness: 0.0, side: THREE.DoubleSide,
  }));
  trend.castShadow = true;
  trend.receiveShadow = true;
  const FLAT = 0.004;
  trend.scale.y = FLAT;
  group.add(trend);

  // ---- plots ----
  const n = y.length;
  const bars = new THREE.InstancedMesh(
    new THREE.BoxGeometry(unit * 0.82, 1, unit * 0.82),
    new THREE.MeshStandardMaterial({ roughness: 0.48, metalness: 0.0 }),
    n,
  );
  bars.castShadow = true;
  bars.receiveShadow = true;
  group.add(bars);

  const targets = {
    raw: { h: y.map(barHeight), color: y.map((v) => seq(v)), surface: FLAT },
    trend: { h: y.map(() => 0.02), color: resid.map((v) => divRes(v)), surface: 1 },
    adjusted: { h: fitted.map(barHeight), color: genoEffect.map((v) => divGeno(v)), surface: FLAT },
  };
  const current = { h: targets.raw.h.slice(), color: targets.raw.color.map((h) => new THREE.Color(h)), surface: FLAT };
  const mtx = new THREE.Matrix4();
  const apply = () => {
    for (let k = 0; k < n; k++) {
      const [i, j] = cells[k];
      const h = current.h[k];
      mtx.makeScale(1, h, 1).setPosition(xOf(j), h / 2, zOf(i));
      bars.setMatrixAt(k, mtx);
      bars.setColorAt(k, current.color[k]);
    }
    bars.instanceMatrix.needsUpdate = true;
    bars.instanceColor.needsUpdate = true;
    trend.scale.y = current.surface;
    bars.visible = current.h.some((h) => h > 0.021);
  };
  apply();

  let mode = "raw";
  let busy = null;
  const setMode = (next) => {
    if (next === mode && !busy) return Promise.resolve();
    mode = next;
    const from = { h: current.h.slice(), color: current.color.map((cc) => cc.clone()), surface: current.surface };
    const to = targets[next];
    const toColors = to.color.map((h) => new THREE.Color(h));
    busy = stage.tween(1.1, (e) => {
      for (let k = 0; k < n; k++) {
        current.h[k] = from.h[k] + (to.h[k] - from.h[k]) * e;
        current.color[k].copy(from.color[k]).lerp(toColors[k], e);
      }
      current.surface = from.surface + (to.surface - from.surface) * e;
      apply();
    }).then(() => { busy = null; });
    return busy;
  };

  // ---- labels ----
  const edge = (nc * unit) / 2 + 0.55;
  const depth = (nr * unit) / 2 + 0.55;
  const every = (count) => Math.max(1, Math.ceil(count / 10));
  g.col_levels.forEach((l, j) => {
    if (j % every(nc)) return;
    const s = sprite(String(l), { color: colors.ink3, size: 11 });
    s.position.set(xOf(j), 0.15, depth);
    group.add(s);
  });
  g.row_levels.forEach((l, i) => {
    if (i % every(nr)) return;
    const s = sprite(String(l), { color: colors.ink3, size: 11 });
    s.position.set(-edge, 0.15, zOf(i));
    group.add(s);
  });
  const colTitle = sprite(g.col, { color: colors.ink2, size: 13, weight: 600 });
  colTitle.position.set(0, 0.15, depth + 0.6);
  const rowTitle = sprite(g.row, { color: colors.ink2, size: 13, weight: 600 });
  rowTitle.position.set(-edge - 0.6, 0.15, 0);
  group.add(colTitle, rowTitle);

  // ---- hover ----
  const response = fitState.model.response;
  stage.addPickable(bars, (hit) => {
    const k = hit.instanceId;
    if (k === undefined) return null;
    const [i, j] = cells[k];
    const rows = [
      { value: fmt(y[k]), label: `observed ${response}` },
      { value: fmt(resid[k]), label: "field trend (residual)" },
      { value: fmt(fitted[k]), label: "trend removed" },
    ];
    if (geno) rows.push({ value: fmt(genoEffect[k]), label: `${term.label} ${geno[k]} effect` });
    return { title: `${g.row} ${g.row_levels[i]} · ${g.col} ${g.col_levels[j]}`, rows };
  });
  stage.addPickable(trend, (hit) => {
    if (mode !== "trend") return null;
    const j = Math.round(hit.point.x / unit + (nc - 1) / 2);
    const i = Math.round(hit.point.z / unit + (nr - 1) / 2);
    const k = cells.findIndex(([a, b]) => a === i && b === j);
    if (k < 0) return { title: `${g.row} ${g.row_levels[i] ?? ""} · ${g.col} ${g.col_levels[j] ?? ""}`, rows: [{ value: "no record" }] };
    return {
      title: `${g.row} ${g.row_levels[i]} · ${g.col} ${g.col_levels[j]}`,
      rows: [{ value: fmt(resid[k]), label: "field trend (residual)" }],
    };
  });

  const resComp = r.fit.variance_components.find((v) => v.name === "residual");
  const rho = (factor) => resComp.parameters.find(([name]) => name === `${factor}.rho`)?.[1];
  const rhoText = [rho(g.row), rho(g.col)].every(Number.isFinite)
    ? ` (ρ ${g.row} ${fixed(rho(g.row), 2)}, ρ ${g.col} ${fixed(rho(g.col), 2)})`
    : "";

  const captions = {
    raw: {
      title: `Observed ${response}`,
      text: "Each column is a plot. Raw yields mix two things: the genotype and where on the field the plot sits.",
      legend: { ramp: seq.css, ends: [fmt(lo, 3), fmt(hi, 3)], rampLabel: `observed ${response}` },
    },
    trend: {
      title: "The field trend the model found",
      text: `The AR1 × AR1 residual${rhoText}: a smooth fertility landscape across rows and columns. Blue is below expectation, red above.`,
      legend: { ramp: divRes.css, ends: [fmt(-maxRes, 3), fmt(maxRes, 3)], rampLabel: "residual (field trend)" },
    },
    adjusted: {
      title: "Trend removed",
      text: `Heights are the yields with the field trend taken out; colour is the ${term ? term.label : "genetic"} effect. What is left is what the genotypes contribute.`,
      legend: { ramp: divGeno.css, ends: [fmt(-maxGeno, 3), fmt(maxGeno, 3)], rampLabel: term ? `${term.label} effect (BLUP)` : "effect" },
    },
  };

  return { setMode, captions, get mode() { return mode; } };
}
