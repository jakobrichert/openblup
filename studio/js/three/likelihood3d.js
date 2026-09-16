// REML likelihood landscape: log L over two variance parameters with the
// AI-REML iterations climbing to the maximum.

import { sequentialScale, seriesHex, niceAxis } from "../charts.js";
import { fmt, fixed } from "../format.js";
import { isoSegments, fillGaps, upsample } from "./stage.js";

// Drops in log L that correspond to 1, 2, ... standard errors along one axis.
const RINGS = [0.5, 2, 4.5, 8, 12.5, 18, 24.5];
// Half the 95% quantile of chi-squared with 2 df: the joint 95% region.
const JOINT_95 = 5.991 / 2;

export function buildLikelihood(stage, surface, { labelA, labelB }) {
  const { THREE, T, scene, sprite, colors } = stage;
  const n0 = surface.a.length;
  const m0 = surface.b.length;
  const S = 3;
  const H = 2.6;
  const FLOOR = -0.5;

  const finite = surface.logl.flat().filter((v) => Number.isFinite(v));
  const best = Math.max(surface.estimate.logl ?? -Infinity, ...finite);
  const coarse = surface.logl.map((row) => row.map((v) => (Number.isFinite(v) ? best - v : null)));
  // Height scale: square root of the drop below the maximum, so the region
  // around the peak (where the confidence contours live) gets most of the
  // height; the scale is capped so a steep edge cannot flatten the peak.
  const drops = coarse.flat().filter((v) => v !== null).sort((x, y) => x - y);
  const q90 = drops[Math.floor(drops.length * 0.9)] ?? 1;
  const maxDrop = Math.max(4, Math.min(30, q90));
  // Smooth coarse grids for rendering (the hover reads the exact grid values).
  const k = Math.max(1, Math.ceil(36 / Math.max(1, Math.min(n0, m0) - 1)));
  const clampDrop = (v) => Math.max(0, Math.min(maxDrop * 1.5, v));
  const drop = (k > 1 ? upsample(fillGaps(coarse), k) : fillGaps(coarse)).map((row) => row.map(clampDrop));
  const n = drop[0].length;
  const m = drop.length;
  // (Iterates can sit a rounding error above the grid maximum: clamp at 0.)
  const heightOf = (d) => (Number.isFinite(d) ? H * (1 - Math.sqrt(Math.max(0, Math.min(maxDrop, d)) / maxDrop)) : 0);
  const [a0, a1] = [surface.a[0], surface.a[n0 - 1]];
  const [b0, b1] = [surface.b[0], surface.b[m0 - 1]];
  const aAt = (ai) => a0 + ((a1 - a0) * ai) / (n - 1);
  const bAt = (bi) => b0 + ((b1 - b0) * bi) / (m - 1);
  const xOf = (a) => -S + (2 * S * (a - a0)) / (a1 - a0);
  const zOf = (b) => S - (2 * S * (b - b0)) / (b1 - b0);
  const inBox = (a, b) => a >= a0 && a <= a1 && b >= b0 && b <= b1;
  const ramp = sequentialScale([0, H]);
  const accent = seriesHex(1);
  const group = new THREE.Group();
  scene.add(group);

  // ---- surface ----
  const positions = new Float32Array(n * m * 3);
  const vertexColors = new Float32Array(n * m * 3);
  const c = new THREE.Color();
  for (let bi = 0; bi < m; bi++) {
    for (let ai = 0; ai < n; ai++) {
      const idx = bi * n + ai;
      const y = heightOf(drop[bi][ai]);
      positions.set([xOf(aAt(ai)), y, zOf(bAt(bi))], idx * 3);
      c.set(ramp(y));
      vertexColors.set([c.r, c.g, c.b], idx * 3);
    }
  }
  const index = [];
  for (let bi = 0; bi < m - 1; bi++) {
    for (let ai = 0; ai < n - 1; ai++) {
      const p = bi * n + ai;
      index.push(p, p + 1, p + n, p + 1, p + n + 1, p + n);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(vertexColors, 3));
  geometry.setIndex(index);
  geometry.computeVertexNormals();
  const surfaceMesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.42, metalness: 0.02, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
  }));
  surfaceMesh.castShadow = true;
  surfaceMesh.receiveShadow = true;
  group.add(surfaceMesh);

  // Wire grid on the surface.
  const wire = [];
  const step = Math.max(1, Math.round((n - 1) / 12));
  for (let bi = 0; bi < m; bi += step) {
    for (let ai = 0; ai < n - 1; ai++) {
      wire.push(...[ai, ai + 1].flatMap((q) => [xOf(aAt(q)), heightOf(drop[bi][q]) + 0.004, zOf(bAt(bi))]));
    }
  }
  for (let ai = 0; ai < n; ai += step) {
    for (let bi = 0; bi < m - 1; bi++) {
      wire.push(...[bi, bi + 1].flatMap((q) => [xOf(aAt(ai)), heightOf(drop[q][ai]) + 0.004, zOf(bAt(q))]));
    }
  }
  const wireGeom = new THREE.BufferGeometry();
  wireGeom.setAttribute("position", new THREE.Float32BufferAttribute(wire, 3));
  group.add(new THREE.LineSegments(wireGeom, new THREE.LineBasicMaterial({
    color: stage.dark ? 0xffffff : 0x0b0b0b, transparent: true, opacity: stage.dark ? 0.12 : 0.1,
  })));

  // ---- floor with a contour map ----
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(2 * S + 1.2, 2 * S + 1.2),
    new THREE.MeshStandardMaterial({ color: stage.dark ? 0x15171d : 0xf3f4f7, roughness: 0.95 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = FLOOR;
  floor.receiveShadow = true;
  group.add(floor);

  const gridX = (g) => xOf(a0 + ((a1 - a0) * g) / (n - 1));
  const gridZ = (g) => zOf(b0 + ((b1 - b0) * g) / (m - 1));
  const contour = (level, y, material) => {
    const seg = isoSegments(drop.map((row) => row.map((d) => -d)), -level);
    if (!seg.length) return null;
    const pts = [];
    for (let i = 0; i < seg.length; i += 4) {
      pts.push(gridX(seg[i]), y, gridZ(seg[i + 1]), gridX(seg[i + 2]), y, gridZ(seg[i + 3]));
    }
    const g = new T.LineSegmentsGeometry().setPositions(pts);
    const line = new T.LineSegments2(g, material);
    group.add(line);
    return line;
  };
  const ringFloor = stage.fatMaterial({ color: new THREE.Color(colors.ink3), linewidth: 1.2, opacity: 0.8 });
  const ringSurface = stage.fatMaterial({ color: stage.dark ? 0xffffff : 0x0b0b0b, linewidth: 1, opacity: 0.35 });
  const jointFloor = stage.fatMaterial({ color: new THREE.Color(colors.ink), linewidth: 2.6, opacity: 0.95 });
  const jointSurface = stage.fatMaterial({ color: stage.dark ? 0xffffff : 0x0b0b0b, linewidth: 2.4, opacity: 0.9 });
  for (const level of RINGS.filter((l) => l < maxDrop)) {
    contour(level, FLOOR + 0.004, ringFloor);
    contour(level, heightOf(level) + 0.012, ringSurface);
  }
  if (JOINT_95 < maxDrop) {
    contour(JOINT_95, FLOOR + 0.006, jointFloor);
    contour(JOINT_95, heightOf(JOINT_95) + 0.016, jointSurface);
  }

  // Box frame, ticks and axis titles.
  const frame = [
    [-S, FLOOR, -S], [S, FLOOR, -S], [S, FLOOR, -S], [S, FLOOR, S], [S, FLOOR, S], [-S, FLOOR, S], [-S, FLOOR, S], [-S, FLOOR, -S],
    [-S, FLOOR, S], [-S, H, S],
  ].flat();
  const frameGeom = new THREE.BufferGeometry();
  frameGeom.setAttribute("position", new THREE.Float32BufferAttribute(frame, 3));
  group.add(new THREE.LineSegments(frameGeom, new THREE.LineBasicMaterial({ color: new THREE.Color(colors.ink3), transparent: true, opacity: 0.6 })));

  const axA = niceAxis(a0, a1, 4);
  for (const t of axA.ticks.filter((t) => t >= a0 && t <= a1)) {
    const s = sprite(axA.format(t), { color: colors.ink3, size: 11 });
    s.position.set(xOf(t), FLOOR + 0.14, S + 0.45);
    group.add(s);
  }
  const axB = niceAxis(b0, b1, 4);
  for (const t of axB.ticks.filter((t) => t >= b0 && t <= b1)) {
    const s = sprite(axB.format(t), { color: colors.ink3, size: 11 });
    s.position.set(S + 0.5, FLOOR + 0.14, zOf(t));
    group.add(s);
  }
  const titleA = sprite(labelA, { color: colors.ink2, size: 13, weight: 600 });
  titleA.position.set(0, FLOOR + 0.14, S + 1.15);
  const titleB = sprite(labelB, { color: colors.ink2, size: 13, weight: 600 });
  titleB.position.set(S + 1.35, FLOOR + 0.14, 0);
  const titleY = sprite("REML log L", { color: colors.ink2, size: 13, weight: 600 });
  titleY.position.set(-S, H + 0.35, S);
  group.add(titleA, titleB, titleY);
  for (const d of [0, maxDrop / 4, maxDrop]) {
    const s = sprite(fixed(best - d, 1), { color: colors.ink3, size: 11, align: "right" });
    s.position.set(-S - 0.12, heightOf(d), S);
    group.add(s);
  }

  // ---- optimizer path ----
  const accentColor = new THREE.Color(accent);
  // Converged iterations repeat the same point; keep one of each (a spline
  // through coincident points is undefined).
  const pathPts = [];
  for (const p of surface.path) {
    if (!Number.isFinite(p.logl) || !inBox(p.a, p.b)) continue;
    const v = new THREE.Vector3(xOf(p.a), heightOf(best - p.logl) + 0.05, zOf(p.b));
    if (pathPts.length && pathPts[pathPts.length - 1].v.distanceTo(v) < 1e-3) continue;
    pathPts.push({ ...p, v });
  }
  let tube = null;
  if (pathPts.length >= 2) {
    const curve = new THREE.CatmullRomCurve3(pathPts.map((p) => p.v), false, "centripetal");
    const tubeGeom = new THREE.TubeGeometry(curve, pathPts.length * 16, 0.045, 12, false);
    tube = new THREE.Mesh(tubeGeom, new THREE.MeshStandardMaterial({
      color: accentColor, emissive: accentColor, emissiveIntensity: 0.35, roughness: 0.35,
    }));
    tube.castShadow = true;
    tubeGeom.setDrawRange(0, 0);
    group.add(tube);
    const total = tubeGeom.index.count;
    stage.tween(2.4, (e) => tubeGeom.setDrawRange(0, Math.floor((e * total) / 6) * 6));
  }
  const dots = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.07, 16, 12),
    new THREE.MeshStandardMaterial({ color: stage.dark ? 0xffffff : 0x0b0b0b, roughness: 0.3 }),
    Math.max(1, pathPts.length),
  );
  const mtx = new THREE.Matrix4();
  pathPts.forEach((p, i) => dots.setMatrixAt(i, mtx.makeTranslation(p.v.x, p.v.y, p.v.z)));
  dots.count = pathPts.length;
  dots.castShadow = true;
  group.add(dots);
  if (pathPts.length) {
    const start = sprite(`start · iteration ${pathPts[0].iteration}`, { color: colors.ink, size: 12, weight: 600, onTop: true });
    start.position.copy(pathPts[0].v).add(new THREE.Vector3(0, 0.3, 0));
    group.add(start);
  }

  // ---- the estimate ----
  const est = surface.estimate;
  const ev = new THREE.Vector3(xOf(est.a), H + 0.07, zOf(est.b));
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.11, 24, 18),
    new THREE.MeshStandardMaterial({ color: accentColor, emissive: accentColor, emissiveIntensity: 0.6, roughness: 0.25 }),
  );
  marker.position.copy(ev);
  marker.castShadow = true;
  group.add(marker);
  const stem = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(ev.x, FLOOR, ev.z), ev]);
  group.add(new THREE.Line(stem, new THREE.LineBasicMaterial({ color: accentColor, transparent: true, opacity: 0.7 })));
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.12, 0.17, 40),
    new THREE.MeshBasicMaterial({ color: accentColor, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(ev.x, FLOOR + 0.008, ev.z);
  group.add(ring);
  const estLabel = sprite(`REML estimate · log L ${fixed(best, 2)}`, {
    color: "#ffffff", size: 12, weight: 600, bg: accent, onTop: true,
  });
  estLabel.position.set(ev.x, ev.y + 0.42, ev.z);
  group.add(estLabel);
  stage.onTick((dt) => {
    const t = performance.now() / 1000;
    const s = 1 + 0.18 * Math.sin(t * 3);
    ring.scale.set(s, s, s);
    marker.rotation.y += dt;
  });

  // ---- hover ----
  const nearest = (point) => {
    const ai = Math.max(0, Math.min(n0 - 1, Math.round(((point.x + S) / (2 * S)) * (n0 - 1))));
    const bi = Math.max(0, Math.min(m0 - 1, Math.round(((S - point.z) / (2 * S)) * (m0 - 1))));
    return { ai, bi };
  };
  stage.addPickable(surfaceMesh, (hit) => {
    const { ai, bi } = nearest(hit.point);
    const l = surface.logl[bi][ai];
    return {
      title: `${labelA} ${fmt(surface.a[ai])} · ${labelB} ${fmt(surface.b[bi])}`,
      rows: Number.isFinite(l)
        ? [{ value: fixed(l, 3), label: "log L" }, { value: fixed(best - l, 3), label: "below the maximum" }]
        : [{ value: "not defined here" }],
    };
  });
  stage.addPickable(dots, (hit) => {
    const p = pathPts[hit.instanceId];
    if (!p) return null;
    return {
      title: `Iteration ${p.iteration}`,
      rows: [
        { value: fixed(p.logl, 3), label: "log L" },
        { value: fmt(p.a), label: labelA },
        { value: fmt(p.b), label: labelB },
      ],
    };
  });
  stage.addPickable(marker, () => ({
    title: "REML estimate",
    rows: [
      { value: fixed(best, 3), label: "log L" },
      { value: fmt(est.a), label: labelA },
      { value: fmt(est.b), label: labelB },
    ],
  }));

  return {
    legend: {
      ramp: ramp.css,
      ends: [fixed(best - maxDrop, 1), fixed(best, 1)],
      rampLabel: "REML log-likelihood (square-root height scale)",
      keys: [
        { color: accent, label: `AI-REML path (${pathPts.length} iterations)` },
        { color: colors.ink, label: "95% joint confidence region" },
        { color: colors.ink3, label: "1, 2, 3 … SE contours" },
      ],
    },
    points: { best, maxDrop },
  };
}
