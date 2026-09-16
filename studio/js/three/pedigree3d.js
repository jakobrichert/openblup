// Pedigree in 3D: one ring per generation, sire/dam links, animals coloured
// by estimated breeding value. Hovering an animal lights up its lineage.

import { divergingScale, seriesHex } from "../charts.js";
import { fmt, fixed, int } from "../format.js";

export function buildPedigree(stage, fitState, pedTerm) {
  const { THREE, T, scene, sprite, colors } = stage;
  const r = fitState.result;
  const ped = r.pedigree;
  const ids = ped.ids;
  const N = ids.length;
  const sire = ped.sire;
  const dam = ped.dam;

  const block = r.fit.random_effects.find((b) => b.term === pedTerm);
  const ebv = new Map(block ? block.effects.map((e) => [e.level, e]) : []);
  const comp = r.fit.variance_components.find((v) => v.name === pedTerm);
  const s2a = comp?.parameters[0]?.[1];
  const records = new Map();
  (r.observations.labels[pedTerm] || []).forEach((id) => records.set(id, (records.get(id) || 0) + 1));

  // Generation = 1 + deepest parent (the pedigree is sorted, parents first).
  const gen = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    const ps = [sire[i], dam[i]].filter((p) => p !== null && p !== undefined);
    gen[i] = ps.length ? 1 + Math.max(...ps.map((p) => gen[p])) : 0;
  }
  const nGen = Math.max(...gen) + 1;
  const layers = Array.from({ length: nGen }, () => []);
  gen.forEach((g, i) => layers[g].push(i));

  // Barycentric ordering on rings to keep parent-offspring links short.
  const angle = new Array(N).fill(0);
  const children = Array.from({ length: N }, () => []);
  for (let i = 0; i < N; i++) for (const p of [sire[i], dam[i]]) if (p !== null && p !== undefined) children[p].push(i);
  const place = (layer, key) => {
    layer.sort((a, b) => key(a) - key(b));
    layer.forEach((i, k) => { angle[i] = (2 * Math.PI * k) / layer.length; });
  };
  layers.forEach((layer) => place(layer, (i) => i));
  const circularMean = (list) => {
    if (!list.length) return null;
    const x = list.reduce((s, i) => s + Math.cos(angle[i]), 0);
    const y = list.reduce((s, i) => s + Math.sin(angle[i]), 0);
    return (Math.atan2(y, x) + 2 * Math.PI) % (2 * Math.PI);
  };
  for (let sweep = 0; sweep < 4; sweep++) {
    for (let g = 1; g < nGen; g++) {
      place(layers[g], (i) => circularMean([sire[i], dam[i]].filter((p) => p !== null && p !== undefined)) ?? angle[i]);
    }
    for (let g = nGen - 2; g >= 0; g--) {
      place(layers[g], (i) => circularMean(children[i]) ?? angle[i]);
    }
  }

  const GAP = 2.3;
  const height = (nGen - 1) * GAP;
  const radius = layers.map((layer) => Math.max(1.4, (layer.length * 0.42) / (2 * Math.PI)));
  const pos = ids.map((_, i) => {
    const g = gen[i];
    const twist = g * 0.35;
    return new THREE.Vector3(
      radius[g] * Math.cos(angle[i] + twist),
      height - g * GAP + 0.6,
      radius[g] * Math.sin(angle[i] + twist),
    );
  });

  const values = ids.map((id) => ebv.get(id)?.estimate ?? 0);
  const maxAbs = Math.max(1e-9, ...values.map(Math.abs));
  const div = divergingScale(maxAbs);
  const group = new THREE.Group();
  scene.add(group);

  // ---- floor disc ----
  const floorR = Math.max(...radius) + 1.6;
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(floorR, 96),
    new THREE.MeshStandardMaterial({ color: stage.dark ? 0x131519 : 0xeef0f4, roughness: 1 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.4;
  floor.receiveShadow = true;
  group.add(floor);

  // ---- generation rings and labels ----
  layers.forEach((layer, g) => {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(radius[g], 0.008, 6, 128),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(colors.ink3), transparent: true, opacity: 0.35 }),
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.y = height - g * GAP + 0.6;
    group.add(ring);
    const label = sprite(g === 0 ? `Founders · ${layer.length}` : `Generation ${g} · ${layer.length}`, { color: colors.ink2, size: 12, weight: 600, align: "left" });
    label.position.set(radius[g] + 0.5, height - g * GAP + 0.6, 0);
    group.add(label);
  });

  // ---- links ----
  const edges = [];
  for (let i = 0; i < N; i++) {
    for (const p of [sire[i], dam[i]]) if (p !== null && p !== undefined) edges.push([p, i]);
  }
  const base = new THREE.Color(stage.dark ? "#5a5d66" : "#b9bcc4");
  const upColor = new THREE.Color(seriesHex(1));
  const downColor = new THREE.Color(seriesHex(0));
  const linePos = [];
  const lineCol = [];
  for (const [p, i] of edges) {
    linePos.push(pos[p].x, pos[p].y, pos[p].z, pos[i].x, pos[i].y, pos[i].z);
    lineCol.push(base.r, base.g, base.b, base.r, base.g, base.b);
  }
  const lineGeom = new T.LineSegmentsGeometry().setPositions(linePos).setColors(lineCol);
  const lineMat = stage.fatMaterial({ vertexColors: true, linewidth: 1.3, opacity: 0.75 });
  const links = new T.LineSegments2(lineGeom, lineMat);
  group.add(links);

  // ---- animals ----
  const withRecord = (i) => (records.get(ids[i]) || 0) > 0;
  const nodes = new THREE.InstancedMesh(
    new THREE.SphereGeometry(1, 24, 16),
    new THREE.MeshStandardMaterial({ roughness: 0.3, metalness: 0.05 }),
    N,
  );
  nodes.castShadow = true;
  const mtx = new THREE.Matrix4();
  const nodeColor = values.map((v) => new THREE.Color(div(v)));
  const muted = new THREE.Color(stage.dark ? "#3a3c42" : "#d7d9de");
  const paint = (lit) => {
    for (let i = 0; i < N; i++) nodes.setColorAt(i, lit && !lit.has(i) ? muted : nodeColor[i]);
    nodes.instanceColor.needsUpdate = true;
  };
  for (let i = 0; i < N; i++) {
    const rad = withRecord(i) ? 0.17 : 0.12;
    mtx.makeScale(rad, rad, rad).setPosition(pos[i]);
    nodes.setMatrixAt(i, mtx);
  }
  paint(null);
  group.add(nodes);

  // Top animals get a name tag.
  const top = values.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  top.forEach(([v, i], rank) => {
    const tag = sprite(`#${rank + 1} ${ids[i]} · EBV ${fixed(v, 1)}`, { color: "#ffffff", size: 11, weight: 600, bg: seriesHex(0), onTop: true });
    tag.position.copy(pos[i]).add(new THREE.Vector3(0, 0.45 + 0.4 * rank, 0));
    group.add(tag);
  });

  // ---- lineage highlight ----
  const lineage = (i) => {
    const up = new Set();
    const down = new Set();
    const stackUp = [i];
    while (stackUp.length) {
      const k = stackUp.pop();
      for (const p of [sire[k], dam[k]]) if (p !== null && p !== undefined && !up.has(p)) { up.add(p); stackUp.push(p); }
    }
    const stackDown = [i];
    while (stackDown.length) {
      const k = stackDown.pop();
      for (const ch of children[k]) if (!down.has(ch)) { down.add(ch); stackDown.push(ch); }
    }
    return { up, down };
  };
  // Recolour the links in place (one interleaved buffer: rgb start, rgb end).
  const colorBuffer = lineGeom.attributes.instanceColorStart?.data;
  const setEdgeColor = (e, c) => {
    if (!colorBuffer) return;
    colorBuffer.array.set([c.r, c.g, c.b, c.r, c.g, c.b], e * 6);
  };
  let lit = -1;
  stage.onHover((h) => {
    const i = h && h.entry.object === nodes ? h.hit.instanceId : -1;
    if (i === lit) return;
    lit = i;
    if (i < 0) {
      edges.forEach((_, e) => setEdgeColor(e, base));
      paint(null);
    } else {
      const { up, down } = lineage(i);
      const on = new Set([i, ...up, ...down]);
      edges.forEach(([p, ch], e) => {
        const c = (up.has(p) || p === i) && (up.has(ch) || ch === i) ? upColor
          : (down.has(ch) && (down.has(p) || p === i)) ? downColor : base;
        setEdgeColor(e, c);
      });
      paint(on);
    }
    if (colorBuffer) colorBuffer.needsUpdate = true;
  });

  stage.addPickable(nodes, (hit) => {
    const i = hit.instanceId;
    if (i === undefined) return null;
    const e = ebv.get(ids[i]);
    const rows = [{ value: fmt(values[i]), label: "estimated breeding value" }];
    if (e && Number.isFinite(s2a) && s2a > 0) {
      rows.push({ value: fixed(Math.max(0, 1 - (e.se * e.se) / s2a), 2), label: "reliability" });
    }
    rows.push(
      { value: int(records.get(ids[i]) || 0), label: "records" },
      { value: fixed(ped.inbreeding[i], 3), label: "inbreeding F" },
      { value: `${sire[i] !== null ? ids[sire[i]] : "unknown"} × ${dam[i] !== null ? ids[dam[i]] : "unknown"}`, label: "parents" },
    );
    return { title: `${ids[i]} · generation ${gen[i]}`, rows };
  });

  return {
    radius: Math.hypot(floorR, height / 2 + 0.8),
    center: [0, height / 2 + 0.2, 0],
    legend: {
      ramp: div.css,
      ends: [fmt(-maxAbs, 3), fmt(maxAbs, 3)],
      rampLabel: "estimated breeding value",
      keys: [
        { color: seriesHex(1), label: "ancestors (on hover)" },
        { color: seriesHex(0), label: "descendants (on hover)" },
        { dot: true, color: colors.ink3, label: "large = has records" },
      ],
    },
    caption: {
      title: `${int(N)}-animal pedigree in 3D`,
      text: `Founders on top, each generation a ring below; links run from parents to offspring and colour is the breeding value. Hover an animal to trace its ancestors and descendants: information flows along these links, which is how animals without records get EBVs.`,
    },
  };
}
