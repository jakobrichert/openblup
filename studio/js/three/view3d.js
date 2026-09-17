// The 3D tab: scene picker, per-scene controls, snapshot and clip recording.

import { h, clear, icon, ICONS, segmented, select, download } from "../dom.js";
import { likelihoodSurface } from "../engine.js";
import { duration } from "../format.js";
import { createStage } from "./stage.js";
import { buildLikelihood } from "./likelihood3d.js";
import { buildField, FIELD_MODES } from "./field3d.js";
import { buildPedigree } from "./pedigree3d.js";
import { buildGxe } from "./gxe3d.js";

const SCENES = {
  field: "Field landscape",
  gxe: "G×E landscape",
  pedigree: "Pedigree",
  likelihood: "Likelihood landscape",
};

function pedigreeTerm(fitState) {
  const m = fitState.model;
  if (!fitState.pedigreeName) return null;
  return m.pedigreeTerm || (m.random[0] ? m.random[0].b || m.random[0].a : null);
}

function parameters(fitState) {
  const fit = fitState.result.fit;
  return fit.variance_components.flatMap((c) => c.parameters.map((p, i) => ({
    name: `${c.name} ${p[0]}`,
    value: p[1],
    se: c.se[i] > 0 ? c.se[i] : 0,
    boundary: c.at_boundary[i],
  })));
}

export function scenesFor(fitState) {
  const r = fitState.result;
  const out = [];
  if (r.grid) out.push("field");
  const gxe = r.covariances.find((c) => r.model.random_terms.find((t) => t.label === c.term)?.columns.length === 2);
  if (gxe) out.push("gxe");
  const pt = pedigreeTerm(fitState);
  if (r.pedigree && pt && r.fit.random_effects.some((b) => b.term === pt)) out.push("pedigree");
  if (parameters(fitState).length >= 2) out.push("likelihood");
  return out;
}

function legendView(legend) {
  if (!legend) return null;
  return h("div", { class: "stage-legend" },
    legend.rampLabel ? h("span", { text: legend.rampLabel }) : null,
    legend.ramp ? h("span", { class: "ramp", style: { background: legend.ramp } }) : null,
    legend.ends ? h("span", { class: "ends" }, h("span", { text: legend.ends[0] }), h("span", { text: legend.ends[1] })) : null,
    (legend.keys || []).map((k) => h("span", { class: "key" },
      h("i", { class: k.dot ? "dot" : "", style: { background: k.color } }), k.label)));
}

/** Mount the 3D view into `root`; returns a dispose function. */
export function mount3D(root, fitState, view) {
  const scenes = scenesFor(fitState);
  view.scene = scenes.includes(view.scene) ? view.scene : scenes[0];
  const params = parameters(fitState);
  if (view.pa === undefined || view.pb === undefined) {
    const free = params.map((p, i) => [p, i]).filter(([p]) => !p.boundary).map(([, i]) => i);
    const pick = free.length >= 2 ? free : params.map((_, i) => i);
    [view.pa, view.pb] = pick;
  }
  view.fieldMode ??= "raw";

  const stageHost = h("div", { class: "stage" });
  const caption = h("div", { class: "stage-caption" });
  const legendHost = h("div", {});
  const status = h("div", { class: "stage-status", text: "Loading the 3D engine…" });
  stageHost.append(
    h("div", { class: "stage-overlay" }, caption, legendHost),
    status,
    h("div", { class: "stage-hint", text: "Drag to orbit · scroll to zoom · hover for values" }),
  );
  const sceneControls = h("div", { class: "filters" });
  const actions = h("div", { class: "filters", style: { marginLeft: "auto" } });
  const card = h("section", { class: "card figure" },
    h("div", { class: "figure-head" }, h("div", {},
      h("h3", { text: "3D view" }),
      h("p", { text: "Interactive 3D renderings of this fit. Every surface and colour is computed from the model." }))),
    h("div", { class: "filters", style: { marginBottom: "12px" } },
      segmented(scenes.map((s) => [s, SCENES[s]]), view.scene, (s) => { view.scene = s; renderControls(); show(); }, { label: "3D scene" }),
      sceneControls,
      actions),
    stageHost);
  root.append(card);

  let stage = null;
  let scene = null;
  let token = 0;
  let disposed = false;
  let recording = false;
  let playing = null;

  const setCaption = (title, text) => {
    clear(caption);
    caption.append(h("strong", { text: title }), h("span", { text }));
  };

  const renderActions = () => {
    clear(actions);
    const rotate = h("button", {
      class: "btn", type: "button", text: stage?.controls.autoRotate ? "Auto-rotate on" : "Auto-rotate off",
      attrs: { "aria-pressed": String(!!stage?.controls.autoRotate) },
      on: { click: () => { if (stage) { stage.setAutoRotate(!stage.controls.autoRotate); renderActions(); } } },
    });
    const png = h("button", {
      class: "btn", type: "button",
      on: { click: () => stage?.snapshot(`openblup-${view.scene}.png`) },
    }, icon(ICONS.download), "PNG");
    const rec = h("button", {
      class: `btn${recording ? " recording" : ""}`, type: "button", disabled: recording || !stage,
      attrs: { title: "Record one full orbit as a 1920 × 1080 video" },
      on: { click: recordClip },
    }, h("span", { class: "rec-dot" }), recording ? "Recording…" : "Record clip");
    actions.append(rotate, png, rec);
  };

  const renderControls = () => {
    clear(sceneControls);
    if (view.scene === "likelihood") {
      const opts = params.map((p, i) => [String(i), p.name]);
      sceneControls.append(
        h("label", {}, "x", select(opts, String(view.pa), (v) => {
          view.pa = Number(v);
          if (view.pa === view.pb) view.pb = params.findIndex((_, i) => i !== view.pa);
          renderControls();
          show();
        })),
        h("label", {}, "y", select(opts.filter(([v]) => Number(v) !== view.pa), String(view.pb), (v) => { view.pb = Number(v); show(); })));
    } else if (view.scene === "field") {
      sceneControls.append(
        segmented(FIELD_MODES, view.fieldMode, (m) => { stopStory(); view.fieldMode = m; renderControls(); applyFieldMode(); }, { label: "Field layer" }),
        h("button", {
          class: "btn", type: "button",
          on: { click: () => (playing ? stopStory() : playStory()) },
        }, icon(playing ? ICONS.x : ICONS.play), playing ? "Stop" : "Play story"));
    }
  };

  const applyFieldMode = () => {
    if (view.scene !== "field" || !scene) return;
    scene.setMode(view.fieldMode);
    const c = scene.captions[view.fieldMode];
    setCaption(c.title, c.text);
    clear(legendHost);
    legendHost.append(legendView(c.legend));
  };

  function stopStory() {
    if (playing) clearInterval(playing);
    playing = null;
    renderControls();
  }

  function playStory(interval = 3600) {
    if (view.scene !== "field") return;
    stopStory();
    const order = FIELD_MODES.map(([m]) => m);
    const step = () => {
      view.fieldMode = order[(order.indexOf(view.fieldMode) + 1) % order.length];
      renderControls();
      applyFieldMode();
    };
    playing = setInterval(step, interval);
    renderControls();
  }

  async function recordClip() {
    if (!stage || recording) return;
    recording = true;
    renderActions();
    const button = actions.lastChild;
    try {
      const isField = view.scene === "field";
      if (isField) {
        view.fieldMode = "raw";
        applyFieldMode();
        playStory(4000);
      }
      const titles = {
        likelihood: ["The REML likelihood landscape", "AI-REML climbing to the maximum, with the 95% joint confidence region"],
        field: ["Separating field trend from genetics", fitState.result.grid
          ? `An AR1 × AR1 spatial model on a ${fitState.result.grid.row_levels.length} × ${fitState.result.grid.col_levels.length} field trial`
          : "An AR1 × AR1 spatial model"],
        pedigree: ["Breeding values through a pedigree", "Information flows along parent–offspring links"],
        gxe: ["Genotype-by-environment reaction norms", "Factor-analytic model across environments"],
      };
      const [title, subtitle] = titles[view.scene];
      const { blob, extension } = await stage.record({
        title,
        subtitle,
        seconds: isField ? 16 : 12,
        onFrame: (p) => { if (button) button.lastChild.textContent = `Recording… ${Math.round(p * 100)}%`; },
      });
      if (isField) stopStory();
      download(`openblup-${view.scene}.${extension}`, blob, blob.type);
    } catch (err) {
      setCaption("Recording failed", err.message);
    } finally {
      recording = false;
      renderActions();
    }
  }

  async function show() {
    const my = ++token;
    stopStory();
    if (stage) { stage.dispose(); stage = null; scene = null; }
    clear(legendHost);
    status.hidden = false;
    status.textContent = "Loading the 3D engine…";
    setCaption(SCENES[view.scene], "");
    renderActions();
    try {
      let surface = null;
      if (view.scene === "likelihood") {
        // One computation per parameter pair: switching back is instant and
        // quick back-and-forth does not queue repeat jobs in the engine.
        const key = `${view.pa}:${view.pb}`;
        view.surfaces ??= new Map();
        if (!view.surfaces.has(key)) {
          const fit = fitState.result.fit;
          const perEval = Math.max(1, fitState.ms / Math.max(1, fit.n_iterations + 1));
          const n = Math.max(11, Math.min(41, Math.round(Math.sqrt(2200 / perEval))));
          const t0 = performance.now();
          const job = likelihoodSurface(fitState.request, {
            theta: params.map((p) => p.value),
            se: params.map((p) => p.se),
            params: [view.pa, view.pb],
            n,
            path: fit.history.map((it) => it.variance_params),
          }).then((res) => ({ ...res, ms: performance.now() - t0 }));
          job.catch(() => view.surfaces.delete(key));
          view.surfaces.set(key, job);
          status.textContent = `Evaluating the REML likelihood on a ${n} × ${n} grid…`;
        }
        surface = await view.surfaces.get(key);
        if (my !== token || disposed) return;
      }
      const opts = {
        likelihood: { cameraPosition: [7.6, 5.4, 8.6], target: [0, 0.7, 0], extent: 6 },
        field: { cameraPosition: [6.2, 7.0, 8.4], target: [0, 0.7, 0], extent: 7 },
        gxe: { cameraPosition: [7, 7.5, 10], target: [0, 1.2, 0], extent: 8 },
        pedigree: { cameraPosition: [8, 5, 8], target: [0, 2.3, 0], extent: 8 },
      }[view.scene];
      const s = await createStage(stageHost, { height: 600, ...opts });
      if (my !== token || disposed) { s.dispose(); return; }
      stage = s;
      stageHost.insertBefore(s.canvas, stageHost.firstChild);
      s.onAutoRotateChange(() => renderActions());

      if (view.scene === "likelihood") {
        const [pa, pb] = [params[view.pa], params[view.pb]];
        const info = buildLikelihood(s, surface, { labelA: pa.name, labelB: pb.name });
        setCaption("REML likelihood landscape",
          `The restricted log-likelihood over ${pa.name} and ${pb.name}` +
          `${surface.slice ? " (a slice: the other parameters stay at their estimates)" : ""}. ` +
          `The orange path is AI-REML climbing to the maximum; the bold contour is the 95% joint confidence region. ` +
          `${surface.a.length} × ${surface.b.length} REML evaluations in ${duration(surface.ms)}.`);
        legendHost.append(legendView(info.legend));
      } else if (view.scene === "field") {
        scene = buildField(s, fitState);
        applyFieldMode();
      } else if (view.scene === "pedigree") {
        const info = buildPedigree(s, fitState, pedigreeTerm(fitState));
        s.frame(info.center, info.radius);
        setCaption(info.caption.title, info.caption.text);
        legendHost.append(legendView(info.legend));
      } else if (view.scene === "gxe") {
        const cov = fitState.result.covariances.find((c) => fitState.result.model.random_terms.find((t) => t.label === c.term)?.columns.length === 2);
        const info = buildGxe(s, fitState, cov);
        s.frame(info.center, info.radius);
        setCaption(info.caption.title, info.caption.text);
        legendHost.append(legendView(info.legend));
      }
      status.hidden = true;
      renderActions();
    } catch (err) {
      if (my !== token) return;
      console.error(err);
      status.hidden = false;
      status.textContent = err.message;
    }
  }

  if (!scenes.length) {
    status.textContent = "No 3D view is available for this model.";
    return () => {};
  }
  renderControls();
  renderActions();
  show();
  card.redraw = () => show();

  return () => {
    disposed = true;
    token++;
    stopStory();
    if (stage) stage.dispose();
    stage = null;
  };
}
