// Shared Three.js stage for the 3D views: renderer, lights, orbit controls,
// hover picking, text sprites, PNG snapshots and 1080p clip recording.
// Three.js is loaded on first use (see the import map in index.html).

import { isDark, cssVar, showTip, hideTip } from "../charts.js";

let threeModules = null;

export function loadThree() {
  threeModules ??= Promise.all([
    import("three"),
    import("three/addons/controls/OrbitControls.js"),
    import("three/addons/lines/Line2.js"),
    import("three/addons/lines/LineGeometry.js"),
    import("three/addons/lines/LineSegments2.js"),
    import("three/addons/lines/LineSegmentsGeometry.js"),
    import("three/addons/lines/LineMaterial.js"),
    import("three/addons/environments/RoomEnvironment.js"),
  ]).then(([THREE, oc, l2, lg, ls2, lsg, lm, re]) => ({
    THREE,
    OrbitControls: oc.OrbitControls,
    Line2: l2.Line2,
    LineGeometry: lg.LineGeometry,
    LineSegments2: ls2.LineSegments2,
    LineSegmentsGeometry: lsg.LineSegmentsGeometry,
    LineMaterial: lm.LineMaterial,
    RoomEnvironment: re.RoomEnvironment,
  })).catch((err) => {
    threeModules = null;
    throw new Error(`Could not load Three.js (${err.message}); the 3D views need network access to cdn.jsdelivr.net.`);
  });
  return threeModules;
}

const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

function gradientTexture(THREE, top, bottom) {
  const c = document.createElement("canvas");
  c.width = 4;
  c.height = 256;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, top);
  grad.addColorStop(1, bottom);
  g.fillStyle = grad;
  g.fillRect(0, 0, 4, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * Create a stage inside `host` (which gets the canvas).
 * Returns helpers used by the scene builders.
 */
export async function createStage(host, { height = 560, cameraPosition = [7, 5.5, 8], target = [0, 0.8, 0], extent = 8 } = {}) {
  const T = await loadThree();
  const { THREE, OrbitControls } = T;
  const dark = isDark();
  const ink = cssVar("--ink") || (dark ? "#ffffff" : "#0b0b0b");
  const ink2 = cssVar("--ink-2") || (dark ? "#c3c2b7" : "#52514e");
  const ink3 = cssVar("--ink-3") || "#898781";

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  const canvas = renderer.domElement;
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "img");
  host.append(canvas);

  const scene = new THREE.Scene();
  scene.background = dark ? gradientTexture(THREE, "#1d2230", "#0d0d0d") : gradientTexture(THREE, "#fdfdfc", "#e9ecf2");
  scene.fog = new THREE.Fog(dark ? 0x0d0d0d : 0xeceef3, extent * 2.2, extent * 5);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envTarget = pmrem.fromScene(new T.RoomEnvironment(), 0.04);
  scene.environment = envTarget.texture;
  scene.environmentIntensity = dark ? 0.35 : 0.55;
  pmrem.dispose();

  scene.add(new THREE.HemisphereLight(dark ? 0xbfd4ff : 0xffffff, dark ? 0x1a1a19 : 0xd8d6cf, dark ? 0.9 : 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, dark ? 1.7 : 1.9);
  sun.position.set(extent * 0.6, extent * 1.2, extent * 0.8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const sc = sun.shadow.camera;
  sc.left = -extent; sc.right = extent; sc.top = extent; sc.bottom = -extent;
  sc.near = 0.5; sc.far = extent * 5;
  sun.shadow.bias = -0.0004;
  sun.shadow.radius = 4;
  scene.add(sun);
  const rim = new THREE.DirectionalLight(dark ? 0x6da7ec : 0xffffff, dark ? 0.6 : 0.35);
  rim.position.set(-extent, extent * 0.5, -extent);
  scene.add(rim);

  const camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.05, extent * 12);
  camera.position.set(...cameraPosition);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(...target);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.0;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = extent * 0.35;
  controls.maxDistance = extent * 3;
  controls.update();

  let cssWidth = 0;
  let cssHeight = height;
  let recording = false;
  const resize = () => {
    if (recording) return;
    cssWidth = Math.max(200, Math.floor(host.clientWidth));
    cssHeight = window.innerWidth < 700 ? Math.min(height, 420) : height;
    renderer.setSize(cssWidth, cssHeight);
    camera.aspect = cssWidth / cssHeight;
    camera.updateProjectionMatrix();
  };
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(host);

  // ---- ticking, tweens ----
  const tickers = new Set();
  const timer = new THREE.Timer();
  let autoRotateListeners = [];
  controls.addEventListener("start", () => {
    if (!recording && controls.autoRotate) {
      controls.autoRotate = false;
      autoRotateListeners.forEach((fn) => fn(false));
    }
  });
  const tween = (duration, fn) => new Promise((resolve) => {
    let t = 0;
    const tick = (dt) => {
      t = Math.min(1, t + dt / duration);
      fn(easeInOut(t), t);
      if (t >= 1) {
        tickers.delete(tick);
        resolve();
      }
    };
    tickers.add(tick);
  });

  // ---- picking ----
  const raycaster = new THREE.Raycaster();
  raycaster.params.Line = { threshold: 0.05 };
  const pointer = new THREE.Vector2();
  const pickables = [];
  let hoverHandler = null;
  let lastEvent = null;
  const pick = () => {
    if (!lastEvent || !pickables.length) return;
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((lastEvent.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((lastEvent.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(pickables.map((p) => p.object), false);
    const hit = hits[0];
    const entry = hit && pickables.find((p) => p.object === hit.object);
    const content = entry ? entry.describe(hit) : null;
    if (content) {
      showTip(lastEvent, content);
      canvas.style.cursor = "pointer";
    } else {
      hideTip();
      canvas.style.cursor = "";
    }
    if (hoverHandler) hoverHandler(entry ? { entry, hit } : null);
  };
  canvas.addEventListener("pointermove", (e) => { lastEvent = e; pick(); });
  canvas.addEventListener("pointerleave", () => {
    lastEvent = null;
    hideTip();
    if (hoverHandler) hoverHandler(null);
  });

  // ---- text sprites (rendered in WebGL so recordings include them) ----
  const sprite = (text, { color = ink2, size = 13, weight = 500, bg = null, align = "center", onTop = false } = {}) => {
    const scale = 2;
    const font = `${weight} ${size * scale}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    const c = document.createElement("canvas");
    const g = c.getContext("2d");
    g.font = font;
    const padX = bg ? 8 * scale : 2 * scale;
    const padY = bg ? 5 * scale : 2 * scale;
    const w = Math.ceil(g.measureText(text).width) + padX * 2;
    const h = Math.ceil(size * scale * 1.35) + padY * 2;
    c.width = w;
    c.height = h;
    g.font = font;
    if (bg) {
      g.fillStyle = bg;
      const r = 6 * scale;
      g.beginPath();
      g.roundRect(0, 0, w, h, r);
      g.fill();
    }
    g.fillStyle = color;
    g.textBaseline = "middle";
    g.fillText(text, padX, h / 2 + scale);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearFilter;
    // Axis labels are hidden behind geometry like everything else; key
    // annotations (`onTop`) always stay visible.
    const mat = new THREE.SpriteMaterial({ map: tex, depthTest: !onTop, depthWrite: false, transparent: true, sizeAttenuation: false });
    const s = new THREE.Sprite(mat);
    s.renderOrder = onTop ? 20 : 5;
    s.userData.pixelSize = [w / scale, h / scale];
    s.center.set(align === "left" ? 0 : align === "right" ? 1 : 0.5, 0.5);
    tickers.add(() => {
      // constant on-screen size
      const k = (2 * Math.tan((camera.fov * Math.PI) / 360)) / cssHeight;
      s.scale.set((w / scale) * k, (h / scale) * k, 1);
    });
    return s;
  };

  const fatMaterial = (opts) => new T.LineMaterial({ worldUnits: false, linewidth: 2, transparent: true, ...opts });

  // ---- loop ----
  let fixedDelta = null;
  renderer.setAnimationLoop((time) => {
    timer.update(time);
    const dt = fixedDelta ?? Math.min(0.05, timer.getDelta());
    for (const fn of [...tickers]) fn(dt);
    controls.update(dt);
    renderer.render(scene, camera);
  });

  /** Frame a bounding sphere from the current viewing direction. */
  const frame = (center, radius) => {
    const dir = camera.position.clone().sub(controls.target).normalize();
    const dist = radius / Math.sin((camera.fov * Math.PI) / 360) * 1.05;
    controls.target.set(...center);
    camera.position.copy(controls.target).addScaledVector(dir, dist);
    controls.minDistance = dist * 0.35;
    controls.maxDistance = dist * 2.5;
    camera.far = dist * 6;
    camera.updateProjectionMatrix();
    controls.update();
  };

  const dispose = () => {
    renderer.setAnimationLoop(null);
    timer.dispose();
    ro.disconnect();
    controls.dispose();
    hideTip();
    scene.traverse((obj) => {
      obj.geometry?.dispose?.();
      const mats = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
      for (const m of mats) {
        m.map?.dispose?.();
        m.dispose?.();
      }
    });
    scene.background?.dispose?.();
    envTarget.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
    canvas.remove();
  };

  const snapshot = (filename) => {
    renderer.render(scene, camera);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, "image/png");
  };

  /** Record one full orbit at 1920x1080. Resolves with {blob, extension}. */
  const record = async ({ seconds = 12, width = 1920, heightPx = 1080, onFrame } = {}) => {
    const types = ["video/mp4;codecs=avc1.640028", "video/mp4;codecs=avc1", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    const mimeType = types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
    if (!mimeType) throw new Error("This browser cannot record video from a canvas.");
    recording = true;
    const saved = { pr: renderer.getPixelRatio(), auto: controls.autoRotate, speed: controls.autoRotateSpeed, aspect: camera.aspect };
    renderer.setPixelRatio(1);
    renderer.setSize(width, heightPx, false);
    camera.aspect = width / heightPx;
    camera.updateProjectionMatrix();
    const k = (2 * Math.tan((camera.fov * Math.PI) / 360)) / heightPx;
    const spriteScale = () => scene.traverse((o) => {
      if (o.isSprite && o.userData.pixelSize) {
        const [w, h] = o.userData.pixelSize;
        o.scale.set(w * 1.6 * k, h * 1.6 * k, 1);
      }
    });
    tickers.add(spriteScale);
    controls.autoRotate = true;
    controls.autoRotateSpeed = 60 / seconds;
    fixedDelta = 1 / 60;
    const stream = canvas.captureStream(60);
    const chunks = [];
    const rec = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 16_000_000 });
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    const done = new Promise((resolve) => { rec.onstop = resolve; });
    rec.start(250);
    // Count rendered frames so the clip is exactly one orbit even when the
    // GPU renders below 60 fps.
    const total = Math.round(seconds * 60);
    await new Promise((resolve) => {
      let frames = 0;
      const counter = () => {
        frames += 1;
        onFrame?.(frames / total);
        if (frames >= total) {
          tickers.delete(counter);
          resolve();
        }
      };
      tickers.add(counter);
    });
    rec.stop();
    await done;
    stream.getTracks().forEach((t) => t.stop());
    tickers.delete(spriteScale);
    fixedDelta = null;
    renderer.setPixelRatio(saved.pr);
    controls.autoRotate = saved.auto;
    controls.autoRotateSpeed = saved.speed;
    recording = false;
    resize();
    return { blob: new Blob(chunks, { type: mimeType.split(";")[0] }), extension: mimeType.startsWith("video/mp4") ? "mp4" : "webm" };
  };

  return {
    T,
    THREE,
    scene,
    camera,
    controls,
    renderer,
    canvas,
    dark,
    colors: { ink, ink2, ink3 },
    sprite,
    fatMaterial,
    tween,
    onTick: (fn) => { tickers.add(fn); return () => tickers.delete(fn); },
    addPickable: (object, describe) => pickables.push({ object, describe }),
    onHover: (fn) => { hoverHandler = fn; },
    onAutoRotateChange: (fn) => { autoRotateListeners.push(fn); },
    setAutoRotate: (on) => { controls.autoRotate = on; autoRotateListeners.forEach((fn) => fn(on)); },
    snapshot,
    record,
    frame,
    dispose,
    refreshPick: pick,
  };
}

/** Bicubic (Catmull-Rom) resampling of a matrix with `null` gaps filled. */
export function fillGaps(M) {
  const nr = M.length, nc = M[0].length;
  const out = M.map((row) => row.slice());
  for (let pass = 0; pass < 8; pass++) {
    let missing = 0;
    for (let i = 0; i < nr; i++) {
      for (let j = 0; j < nc; j++) {
        if (out[i][j] !== null && Number.isFinite(out[i][j])) continue;
        let s = 0, n = 0;
        for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const v = out[i + di]?.[j + dj];
          if (v !== null && v !== undefined && Number.isFinite(v)) { s += v; n++; }
        }
        if (n) out[i][j] = s / n;
        else missing++;
      }
    }
    if (!missing) break;
  }
  return out.map((row) => row.map((v) => (v === null || !Number.isFinite(v) ? 0 : v)));
}

function catmull(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}

/** Upsample a full matrix by `k` in both directions with Catmull-Rom splines. */
export function upsample(M, k) {
  const nr = M.length, nc = M[0].length;
  const at = (i, j) => M[Math.max(0, Math.min(nr - 1, i))][Math.max(0, Math.min(nc - 1, j))];
  const rows = (nr - 1) * k + 1, cols = (nc - 1) * k + 1;
  const out = [];
  for (let r = 0; r < rows; r++) {
    const fi = r / k, i = Math.min(nr - 2, Math.floor(fi)), ti = nr === 1 ? 0 : fi - i;
    const row = [];
    for (let c = 0; c < cols; c++) {
      const fj = c / k, j = Math.min(nc - 2, Math.floor(fj)), tj = nc === 1 ? 0 : fj - j;
      const col = (ii) => catmull(at(ii, j - 1), at(ii, j), at(ii, j + 1), at(ii, j + 2), tj);
      row.push(nr === 1 ? col(0) : catmull(col(i - 1), col(i), col(i + 1), col(i + 2), ti));
    }
    out.push(row);
  }
  return out;
}

/**
 * Marching squares: iso-lines of a grid (rows = z, cols = x) at `level`.
 * Returns a flat array of segment endpoints [x0, z0, x1, z1, ...] in grid units.
 */
export function isoSegments(G, level) {
  const out = [];
  const nr = G.length, nc = G[0].length;
  const lerp = (a, b) => (level - a) / (b - a);
  for (let i = 0; i < nr - 1; i++) {
    for (let j = 0; j < nc - 1; j++) {
      const v = [G[i][j], G[i][j + 1], G[i + 1][j + 1], G[i + 1][j]];
      if (v.some((x) => !Number.isFinite(x))) continue;
      let idx = 0;
      v.forEach((x, k) => { if (x > level) idx |= 1 << k; });
      if (idx === 0 || idx === 15) continue;
      const edge = (e) => {
        switch (e) {
          case 0: return [j + lerp(v[0], v[1]), i];
          case 1: return [j + 1, i + lerp(v[1], v[2])];
          case 2: return [j + 1 - lerp(v[2], v[3]), i + 1];
          default: return [j, i + 1 - lerp(v[3], v[0])];
        }
      };
      const table = {
        1: [[3, 0]], 2: [[0, 1]], 3: [[3, 1]], 4: [[1, 2]], 5: [[3, 0], [1, 2]], 6: [[0, 2]], 7: [[3, 2]],
        8: [[2, 3]], 9: [[0, 2]], 10: [[0, 1], [2, 3]], 11: [[1, 2]], 12: [[1, 3]], 13: [[0, 1]], 14: [[3, 0]],
      };
      for (const [a, b] of table[idx]) {
        const p = edge(a), q = edge(b);
        out.push(p[0], p[1], q[0], q[1]);
      }
    }
  }
  return out;
}
