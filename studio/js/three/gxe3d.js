// G×E landscape: 3D reaction norms. One floating ribbon per genotype across the
// environments (ordered by their FA loading); height and colour are the
// environment-specific effect. Crossing ribbons are crossover interactions.

import { divergingScale, seriesHex } from "../charts.js";
import { fmt, fixed } from "../format.js";

function spline(values, k) {
  const n = values.length;
  const at = (i) => values[Math.max(0, Math.min(n - 1, i))];
  const out = [];
  for (let i = 0; i < n - 1; i++) {
    for (let s = 0; s < k; s++) {
      const t = s / k, t2 = t * t, t3 = t2 * t;
      const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)];
      out.push(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3));
    }
  }
  out.push(values[n - 1]);
  return out;
}

export function buildGxe(stage, fitState, cov) {
  const { THREE, scene, sprite, colors } = stage;
  const r = fitState.result;
  const meta = r.model.random_terms.find((t) => t.label === cov.term);
  const block = r.fit.random_effects.find((b) => b.term === cov.term);
  const [envCol, genoCol] = meta.columns;
  const levels = cov.levels;
  const byLoading = !!cov.loadings && cov.loadings[0].length === 1;
  const order = byLoading
    ? levels.map((l, i) => [l, cov.loadings[i][0] ?? 0]).sort((a, b) => a[1] - b[1]).map(([l]) => l)
    : levels.slice();
  const envPos = new Map(order.map((l, i) => [l, i]));

  const table = new Map();
  for (const e of block.effects) {
    const env = levels.find((l) => e.level.startsWith(`${l}:`));
    if (env === undefined) continue;
    const g = e.level.slice(env.length + 1);
    if (!table.has(g)) table.set(g, new Array(order.length).fill(0));
    table.get(g)[envPos.get(env)] = e.estimate;
  }
  const genos = [...table.entries()]
    .map(([name, ys]) => ({ name, ys, mean: ys.reduce((a, v) => a + v, 0) / ys.length }))
    .sort((a, b) => b.mean - a.mean);
  const G = genos.length;
  const E = order.length;
  const maxAbs = Math.max(1e-9, ...genos.flatMap((g) => g.ys.map(Math.abs)));
  const div = divergingScale(maxAbs);

  const W = 8;
  const D = 7;
  const H = 2;
  const K = 16;
  const BASE = H + 0.25; // height of the zero plane
  const xOf = (e) => -W / 2 + (W * e) / Math.max(1, E - 1);
  const zOf = (g) => D / 2 - (D * g) / Math.max(1, G - 1); // best in front
  const yOf = (v) => BASE + (H * v) / maxAbs;
  const half = Math.max(0.04, Math.min(0.12, (0.36 * D) / Math.max(1, G - 1)));
  const group = new THREE.Group();
  scene.add(group);

  // ---- floor, zero plane and environment guides ----
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(W + 5, D + 5),
    new THREE.MeshStandardMaterial({ color: stage.dark ? 0x131519 : 0xeef0f4, roughness: 1 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  group.add(floor);
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(W + 0.8, D + 0.8),
    new THREE.MeshBasicMaterial({
      color: stage.dark ? 0x8ea4c8 : 0x5d6b82, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false,
    }),
  );
  plane.rotation.x = -Math.PI / 2;
  plane.position.y = BASE;
  group.add(plane);
  const guideMat = new THREE.LineBasicMaterial({ color: new THREE.Color(colors.ink3), transparent: true, opacity: 0.45 });
  order.forEach((_, e) => {
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(xOf(e), BASE, D / 2 + 0.4), new THREE.Vector3(xOf(e), BASE, -D / 2 - 0.4),
    ]);
    group.add(new THREE.Line(g, guideMat));
  });

  // ---- ribbons ----
  const c = new THREE.Color();
  const dropPos = [];
  genos.forEach((g, gi) => {
    const ys = spline(g.ys, K);
    const pts = ys.length;
    const z = zOf(gi);
    const pos = new Float32Array(pts * 2 * 3);
    const col = new Float32Array(pts * 2 * 3);
    for (let k = 0; k < pts; k++) {
      const x = xOf(k / K);
      const y = yOf(ys[k]);
      c.set(div(Math.max(-maxAbs, Math.min(maxAbs, ys[k]))));
      pos.set([x, y, z - half, x, y, z + half], k * 6);
      col.set([c.r, c.g, c.b, c.r, c.g, c.b], k * 6);
    }
    const idx = [];
    for (let k = 0; k < pts - 1; k++) {
      const a = k * 2, b = (k + 1) * 2;
      idx.push(a, a + 1, b, a + 1, b + 1, b);
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geom.setIndex(idx);
    geom.computeVertexNormals();
    const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.35, metalness: 0.05, side: THREE.DoubleSide,
      emissive: stage.dark ? 0x1b2130 : 0x000000,
    }));
    mesh.castShadow = true;
    group.add(mesh);
    // faint drop lines from each environment's value to the zero plane
    g.ys.forEach((v, e) => dropPos.push(xOf(e), yOf(v), z, xOf(e), BASE, z));
    stage.addPickable(mesh, (hit) => {
      const e = Math.max(0, Math.min(E - 1, Math.round(((hit.point.x + W / 2) / W) * (E - 1))));
      return {
        title: `${genoCol} ${g.name} · ${envCol} ${order[e]}`,
        rows: [
          { value: fmt(g.ys[e]), label: `effect in ${order[e]}` },
          { value: fmt(g.mean), label: "mean across environments" },
          { value: `#${gi + 1} of ${G}`, label: "rank by mean" },
        ],
      };
    });
  });
  const drops = new THREE.BufferGeometry();
  drops.setAttribute("position", new THREE.Float32BufferAttribute(dropPos, 3));
  group.add(new THREE.LineSegments(drops, new THREE.LineBasicMaterial({
    color: new THREE.Color(colors.ink3), transparent: true, opacity: 0.25,
  })));

  // ---- the three best genotypes on average ----
  genos.slice(0, Math.min(3, G)).forEach((g, rank) => {
    const color = new THREE.Color(seriesHex(rank));
    const ys = spline(g.ys, K);
    const curve = new THREE.CatmullRomCurve3(ys.map((v, k) => new THREE.Vector3(xOf(k / K), yOf(v) + 0.02, zOf(rank))));
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, ys.length * 2, 0.05, 10, false),
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.4, roughness: 0.3 }),
    );
    tube.castShadow = true;
    group.add(tube);
    const tag = sprite(`#${rank + 1} ${g.name}`, { color: "#ffffff", size: 11, weight: 600, bg: seriesHex(rank), align: "left", onTop: true });
    tag.position.set(xOf(E - 1) + 0.25, yOf(g.ys[E - 1]), zOf(rank));
    group.add(tag);
  });

  // ---- labels ----
  order.forEach((l, e) => {
    const loading = byLoading ? cov.loadings[levels.indexOf(l)][0] : null;
    const s = sprite(loading !== null ? `${l} · λ ${fixed(loading, 2)}` : String(l), { color: colors.ink2, size: 12, weight: 600 });
    s.position.set(xOf(e), BASE, D / 2 + 0.75);
    group.add(s);
  });
  const envTitle = sprite(byLoading ? `${envCol}, ordered by FA loading` : envCol, { color: colors.ink3, size: 11 });
  envTitle.position.set(0, BASE, D / 2 + 1.9);
  group.add(envTitle);
  const zero = sprite("0", { color: colors.ink3, size: 11, align: "right" });
  zero.position.set(-W / 2 - 0.5, BASE, D / 2 + 0.4);
  const up = sprite(`+${fmt(maxAbs, 2)}`, { color: colors.ink3, size: 11, align: "right" });
  up.position.set(-W / 2 - 0.5, BASE + H, D / 2 + 0.4);
  group.add(zero, up);
  const axis = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-W / 2 - 0.4, BASE - H, D / 2 + 0.4), new THREE.Vector3(-W / 2 - 0.4, BASE + H, D / 2 + 0.4),
  ]);
  group.add(new THREE.Line(axis, guideMat));

  return {
    radius: Math.hypot(W / 2 + 0.6, D / 2 + 1.2) * 0.78,
    center: [0, BASE * 0.85, 0.4],
    legend: {
      ramp: div.css,
      ends: [fmt(-maxAbs, 3), fmt(maxAbs, 3)],
      rampLabel: `${genoCol} effect within ${envCol}`,
      keys: genos.slice(0, 3).map((g, i) => ({ color: seriesHex(i), label: `#${i + 1} ${g.name} (best on average)` })),
    },
    caption: {
      title: "Genotype-by-environment reaction norms",
      text: `Each ribbon is one of ${G} ${genoCol} levels across the ${E} ${envCol} levels${byLoading ? " (ordered by their loading on the common factor)" : ""}, the best on average in front. Height is the predicted effect in that environment; ribbons that cross are crossover interactions.`,
    },
  };
}
