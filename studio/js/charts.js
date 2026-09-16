// Hand-rolled SVG charts. Mark specs: bars <= 24px with a 4px rounded data
// end, 2px lines, >= 8px dots with a 2px surface ring, hairline solid grid,
// text in ink colours (never the series colour), a tooltip on every mark and a
// table view for every figure.

import { h, s, clear, segmented } from "./dom.js";
import { fmt, tickFormatter, isNum } from "./format.js";

// ---------------------------------------------------------------- scales

export function linear([d0, d1], [r0, r1]) {
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const f = (v) => r0 + (v - d0) * k;
  f.invert = (p) => (k === 0 ? d0 : d0 + (p - r0) / k);
  return f;
}

function tickStep(span, count) {
  const raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const err = raw / mag;
  return (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
}

/** Expand [min, max] to tick multiples and return {domain, ticks, format}. */
export function niceAxis(min, max, count = 5, { includeZero = false } = {}) {
  if (!isNum(min) || !isNum(max)) { min = 0; max = 1; }
  if (includeZero) { min = Math.min(min, 0); max = Math.max(max, 0); }
  if (min === max) { const pad = Math.abs(min) * 0.1 || 1; min -= pad; max += pad; }
  const step = tickStep(max - min, count);
  // The epsilon keeps values that are already tick multiples (like 0) as the bound.
  const lo = Math.floor(min / step + 1e-9) * step;
  const hi = Math.ceil(max / step - 1e-9) * step;
  const ticks = [];
  for (let i = 0, v = lo; v <= hi + step * 1e-6 && i < 200; i++, v = lo + i * step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { domain: [lo, hi], ticks, format: tickFormatter(step) };
}

// ---------------------------------------------------------------- colour

export function isDark() {
  const t = document.documentElement.dataset.theme;
  if (t) return t === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => c / 255);
}
const toLin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toGam = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

function rgbToOklab([r, g, b]) {
  [r, g, b] = [r, g, b].map(toLin);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const q = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * q,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * q,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * q,
  ];
}

function oklabToHex([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const q = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const rgb = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * q,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * q,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * q,
  ].map((c) => Math.round(Math.min(1, Math.max(0, toGam(c))) * 255));
  return `#${rgb.map((c) => c.toString(16).padStart(2, "0")).join("")}`;
}

function rampInterpolator(stops) {
  const labs = stops.map((c) => rgbToOklab(hexToRgb(c)));
  return (t) => {
    const x = Math.min(1, Math.max(0, t)) * (labs.length - 1);
    const i = Math.min(labs.length - 2, Math.floor(x));
    const f = x - i;
    return oklabToHex(labs[i].map((v, k) => v + (labs[i + 1][k] - v) * f));
  };
}

const BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

/** Sequential (one hue, more = darker; the anchor flips in dark mode). */
export function sequentialScale([lo, hi]) {
  const stops = isDark() ? [...BLUE_RAMP].reverse().slice(0, 11) : BLUE_RAMP;
  const ramp = rampInterpolator(stops);
  const f = (v) => ramp(hi === lo ? 0.5 : (v - lo) / (hi - lo));
  f.css = `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((t) => ramp(t)).join(",")})`;
  f.domain = [lo, hi];
  return f;
}

/** Diverging blue (negative) <-> grey midpoint <-> red (positive), symmetric. */
export function divergingScale(maxAbs) {
  const dark = isDark();
  const neg = rampInterpolator([dark ? "#383835" : "#f0efec", dark ? "#3987e5" : "#2a78d6"]);
  const pos = rampInterpolator([dark ? "#383835" : "#f0efec", dark ? "#e66767" : "#e34948"]);
  const m = maxAbs > 0 ? maxAbs : 1;
  const f = (v) => (v < 0 ? neg(-v / m) : pos(v / m));
  f.css = `linear-gradient(90deg, ${[-1, -0.5, 0, 0.5, 1].map((t) => f(t * m)).join(",")})`;
  f.domain = [-m, m];
  return f;
}

/** Ink colour (white or near-black) that reads on a given fill. */
export function inkOn(hex) {
  const [r, g, b] = hexToRgb(hex).map(toLin);
  const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return lum > 0.3 ? "#0b0b0b" : "#ffffff";
}

// ---------------------------------------------------------------- tooltip

const tipEl = () => document.getElementById("tooltip");

/** rows: [{value, label, color?}] - the value leads, the label follows. */
export function showTip(event, { title, rows = [] }) {
  const tip = tipEl();
  clear(tip);
  if (title) tip.append(h("div", { class: "tt-title", text: title }));
  for (const r of rows) {
    tip.append(h("div", { class: "tt-row" },
      r.color ? h("span", { class: "tt-key", style: { background: r.color } }) : null,
      h("span", { class: "tt-value", text: r.value }),
      r.label ? h("span", { class: "tt-label", text: r.label }) : null));
  }
  tip.hidden = false;
  let x, y;
  if (event && "clientX" in event && event.type !== "focus") { x = event.clientX; y = event.clientY; }
  else {
    const box = event.target.getBoundingClientRect();
    x = box.left + box.width / 2; y = box.top;
  }
  const { width, height } = tip.getBoundingClientRect();
  let left = x + 14;
  let top = y - height - 12;
  if (left + width > window.innerWidth - 8) left = x - width - 14;
  if (top < 8) top = y + 16;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}

export function hideTip() {
  const tip = tipEl();
  if (tip) tip.hidden = true;
}

function hoverable(node, content) {
  node.setAttribute("tabindex", "0");
  const show = (e) => showTip(e, content());
  node.addEventListener("pointermove", show);
  node.addEventListener("pointerenter", show);
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", show);
  node.addEventListener("blur", hideTip);
  return node;
}

// ---------------------------------------------------------------- figure

/**
 * A chart card with a Chart / Table toggle. `chart(host, width)` draws into
 * `host` and is re-run when the card is resized or the theme changes.
 */
export function figure({ title, subtitle, legend, chart, table, foot, className = "" }) {
  let mode = "chart";
  const body = h("div", { class: "figure-body" });
  const head = h("div", { class: "figure-head" },
    h("div", {}, h("h3", { text: title }), subtitle ? h("p", { text: subtitle }) : null));
  const toggleHost = h("div", { style: { marginLeft: "auto" } });
  if (chart && table) head.append(toggleHost);
  const card = h("figure", { class: `card figure ${className}` }, head,
    legend ? h("div", { class: "legend" }, legend) : null, body,
    foot ? h("figcaption", { class: "figure-foot", text: foot }) : null);

  let lastWidth = 0;
  const draw = () => {
    clear(body);
    if (mode === "table" || !chart) { body.append(table()); return; }
    const width = Math.floor(body.clientWidth);
    if (!width) return;
    lastWidth = width;
    chart(body, width);
  };
  const renderToggle = () => {
    clear(toggleHost);
    toggleHost.append(segmented([["chart", "Chart"], ["table", "Table"]], mode, (m) => {
      mode = m; renderToggle(); draw();
    }, { label: `${title} view` }));
  };
  if (chart && table) renderToggle();

  const ro = new ResizeObserver(() => {
    if (mode !== "chart" || !chart) return;
    const width = Math.floor(body.clientWidth);
    if (width && Math.abs(width - lastWidth) > 1) draw();
  });
  ro.observe(body);
  card.redraw = draw;
  card.addEventListener("studio:redraw", draw);
  queueMicrotask(draw);
  return card;
}

export function redrawAll(root) {
  root.querySelectorAll(".figure").forEach((f) => f.redraw && f.redraw());
}

export function legendItems(items) {
  return items.map(({ label, color, line }) => h("span", {},
    h("i", { class: line ? "line" : "", style: { background: color } }), label));
}

// ---------------------------------------------------------------- tables

/**
 * columns: [{key, label, num?, format?, render?}]; rows: array of objects.
 */
export function dataTable(columns, rows, { limit = 2000, empty = "No rows" } = {}) {
  const shown = rows.slice(0, limit);
  const table = h("table", { class: "data" },
    h("thead", {}, h("tr", {}, columns.map((c) => h("th", { class: c.num ? "num" : "", text: c.label, attrs: { scope: "col" } })))),
    h("tbody", {}, shown.length ? shown.map((r) => h("tr", {}, columns.map((c) => {
      const v = typeof c.key === "function" ? c.key(r) : r[c.key];
      const content = c.render ? c.render(v, r) : c.format ? c.format(v, r) : v ?? "—";
      return h("td", { class: c.num ? "num" : "" }, content);
    }))) : h("tr", {}, h("td", { class: "muted", text: empty, attrs: { colspan: columns.length } }))));
  const wrap = h("div", { class: "table-wrap" }, table);
  if (rows.length > limit) wrap.append(h("div", { class: "hint", style: { padding: "8px 10px" }, text: `Showing ${limit} of ${rows.length} rows (download for all).` }));
  return wrap;
}

// ---------------------------------------------------------------- axes

function truncate(text, maxPx, charPx = 6.7) {
  const max = Math.max(3, Math.floor(maxPx / charPx));
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function xAxis(svg, x, axis, top, bottom, title, { zeroLine = false } = {}) {
  for (const t of axis.ticks) {
    const px = x(t);
    svg.append(s("line", { class: zeroLine && t === 0 ? "baseline" : "gridline", x1: px, x2: px, y1: top, y2: bottom }));
    svg.append(s("text", { x: px, y: bottom + 16, "text-anchor": "middle", text: axis.format(t) }));
  }
  if (title) {
    const [r0, r1] = [x(axis.domain[0]), x(axis.domain[1])];
    svg.append(s("text", { class: "axis-title", x: (r0 + r1) / 2, y: bottom + 34, "text-anchor": "middle", text: title }));
  }
}

function yAxis(svg, y, axis, left, right, title, { zeroLine = false, titleX = 12 } = {}) {
  for (const t of axis.ticks) {
    const py = y(t);
    svg.append(s("line", { class: zeroLine && t === 0 ? "baseline" : "gridline", x1: left, x2: right, y1: py, y2: py }));
    svg.append(s("text", { x: left - 8, y: py + 4, "text-anchor": "end", text: axis.format(t) }));
  }
  if (title) {
    const [r0, r1] = [y(axis.domain[0]), y(axis.domain[1])];
    const cy = (r0 + r1) / 2;
    svg.append(s("text", { class: "axis-title", x: titleX, y: cy, "text-anchor": "middle", transform: `rotate(-90 ${titleX} ${cy})`, text: title }));
  }
}

function frame(host, width, height, label) {
  const svg = s("svg", { class: "chart", width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label });
  host.append(svg);
  return svg;
}

function labelWidthFor(labels, min = 48, max = 180) {
  const longest = labels.reduce((m, l) => Math.max(m, String(l).length), 0);
  return Math.min(max, Math.max(min, longest * 7.2 + 24));
}

/** Horizontal bar with a 4px rounded data end and a square baseline end. */
function hbarPath(x0, x1, y, t) {
  const len = Math.abs(x1 - x0);
  const r = Math.min(4, len, t / 2);
  if (len < 0.5) return `M${x0},${y}h0.5v${t}h-0.5z`;
  if (x1 >= x0) return `M${x0},${y}H${x1 - r}A${r},${r} 0 0 1 ${x1},${y + r}V${y + t - r}A${r},${r} 0 0 1 ${x1 - r},${y + t}H${x0}Z`;
  return `M${x0},${y}H${x1 + r}A${r},${r} 0 0 0 ${x1},${y + r}V${y + t - r}A${r},${r} 0 0 0 ${x1 + r},${y + t}H${x0}Z`;
}

/** Vertical column with a rounded top (or bottom, for negatives). */
function vbarPath(x, t, y0, y1) {
  const len = Math.abs(y1 - y0);
  const r = Math.min(4, len, t / 2);
  if (len < 0.5) return `M${x},${y0}h${t}v-0.5h${-t}z`;
  if (y1 <= y0) return `M${x},${y0}V${y1 + r}A${r},${r} 0 0 1 ${x + r},${y1}H${x + t - r}A${r},${r} 0 0 1 ${x + t},${y1 + r}V${y0}Z`;
  return `M${x},${y0}V${y1 - r}A${r},${r} 0 0 0 ${x + r},${y1}H${x + t - r}A${r},${r} 0 0 0 ${x + t},${y1 - r}V${y0}Z`;
}

// ---------------------------------------------------------------- charts

/**
 * Ranked estimates with intervals. items: [{label, value, lo, hi, tip?}]
 */
export function dotInterval(host, width, { items, xTitle, format = fmt, rowHeight = 22 }) {
  const left = labelWidthFor(items.map((d) => d.label));
  const m = { top: 6, right: 18, bottom: xTitle ? 44 : 26, left };
  const height = m.top + items.length * rowHeight + m.bottom;
  const svg = frame(host, width, height, xTitle || "Estimates with intervals");
  const lo = Math.min(...items.map((d) => d.lo ?? d.value));
  const hi = Math.max(...items.map((d) => d.hi ?? d.value));
  const axis = niceAxis(lo, hi, Math.max(3, Math.floor((width - left) / 90)), { includeZero: true });
  const x = linear(axis.domain, [m.left, width - m.right]);
  const bottom = height - m.bottom;
  xAxis(svg, x, axis, m.top, bottom, xTitle, { zeroLine: true });
  items.forEach((d, i) => {
    const cy = m.top + i * rowHeight + rowHeight / 2;
    const row = s("g", {});
    const hit = s("rect", { class: "hit", x: 0, y: cy - rowHeight / 2, width, height: rowHeight, rx: 4 });
    hoverable(hit, () => d.tip || { title: d.label, rows: [{ value: format(d.value), label: "estimate" }] });
    row.append(hit);
    row.append(s("text", { class: "label", x: m.left - 10, y: cy + 4, "text-anchor": "end", text: truncate(d.label, m.left - 14) }));
    if (isNum(d.lo) && isNum(d.hi)) row.append(s("line", { class: "interval", x1: x(d.lo), x2: x(d.hi), y1: cy, y2: cy, "pointer-events": "none" }));
    row.append(s("circle", { class: "dot", cx: x(d.value), cy, r: 4.5, "pointer-events": "none" }));
    svg.append(row);
  });
}

/**
 * Horizontal bars from zero. items: [{label, value, err?, tip?}]
 */
export function hbars(host, width, { items, xTitle, format = fmt, rowHeight = 30, showValues = true, color }) {
  const left = labelWidthFor(items.map((d) => d.label));
  const m = { top: 4, right: showValues ? 64 : 18, bottom: xTitle ? 44 : 26, left };
  const height = m.top + items.length * rowHeight + m.bottom;
  const svg = frame(host, width, height, xTitle || "Bar chart");
  const lo = Math.min(0, ...items.map((d) => d.value - (d.err || 0)));
  const hi = Math.max(0, ...items.map((d) => d.value + (d.err || 0)));
  const axis = niceAxis(lo, hi, Math.max(3, Math.floor((width - left - m.right) / 90)), { includeZero: true });
  const x = linear(axis.domain, [m.left, width - m.right]);
  const bottom = height - m.bottom;
  xAxis(svg, x, axis, m.top, bottom, xTitle, { zeroLine: true });
  const t = Math.min(24, rowHeight - 10);
  items.forEach((d, i) => {
    const y = m.top + i * rowHeight + (rowHeight - t) / 2;
    const cy = y + t / 2;
    const hit = s("rect", { class: "hit", x: 0, y: y - (rowHeight - t) / 2, width, height: rowHeight, rx: 4 });
    hoverable(hit, () => d.tip || { title: d.label, rows: [{ value: format(d.value) }, ...(d.err ? [{ value: `± ${format(d.err)}`, label: "SE" }] : [])] });
    svg.append(hit);
    svg.append(s("text", { class: "label", x: m.left - 10, y: cy + 4, "text-anchor": "end", text: truncate(d.label, m.left - 14) }));
    svg.append(s("path", { class: color ? "" : "mark", fill: color, d: hbarPath(x(0), x(d.value), y, t), "pointer-events": "none" }));
    if (d.err) {
      const a = x(d.value - d.err), b = x(d.value + d.err);
      svg.append(s("line", { class: "whisker", x1: a, x2: b, y1: cy, y2: cy, "pointer-events": "none" }));
      svg.append(s("line", { class: "whisker", x1: a, x2: a, y1: cy - 4, y2: cy + 4, "pointer-events": "none" }));
      svg.append(s("line", { class: "whisker", x1: b, x2: b, y1: cy - 4, y2: cy + 4, "pointer-events": "none" }));
    }
    if (showValues) {
      const end = d.value >= 0 ? Math.max(x(d.value), d.err ? x(d.value + d.err) : 0) : x(0);
      svg.append(s("text", { class: "value", x: end + 8, y: cy + 4, text: format(d.value), "pointer-events": "none" }));
    }
  });
}

/** Vertical columns. items: [{label, value, tip?}] */
export function columns(host, width, { items, yTitle, format = fmt, height = 240 }) {
  const m = { top: 12, right: 12, bottom: 34, left: 56 };
  const svg = frame(host, width, height, yTitle || "Column chart");
  const lo = Math.min(0, ...items.map((d) => d.value));
  const hi = Math.max(0, ...items.map((d) => d.value));
  const axis = niceAxis(lo, hi, 4, { includeZero: true });
  const y = linear(axis.domain, [height - m.bottom, m.top]);
  yAxis(svg, y, axis, m.left, width - m.right, yTitle, { zeroLine: true });
  const band = (width - m.left - m.right) / items.length;
  const t = Math.min(24, band - 8);
  items.forEach((d, i) => {
    const cx = m.left + band * (i + 0.5);
    const hit = s("rect", { class: "hit", x: cx - band / 2, y: m.top, width: band, height: height - m.top - m.bottom, rx: 4 });
    hoverable(hit, () => d.tip || { title: d.label, rows: [{ value: format(d.value) }] });
    svg.append(hit);
    svg.append(s("path", { class: "mark", d: vbarPath(cx - t / 2, t, y(0), y(d.value)), "pointer-events": "none" }));
    svg.append(s("text", { class: "label", x: cx, y: height - m.bottom + 18, "text-anchor": "middle", text: truncate(d.label, band) }));
  });
}

/**
 * Scatter with nearest-point hover. points: [{x, y, tip?}]
 * ref: "zero" | "identity" | {slope, intercept} | undefined
 */
export function scatter(host, width, { points, xTitle, yTitle, ref, height = 300, format = fmt, guides = [] }) {
  const m = { top: 12, right: 16, bottom: 46, left: 60 };
  const svg = frame(host, width, height, `${yTitle} against ${xTitle}`);
  const xs = points.map((p) => p.x).filter(isNum);
  const ys = points.map((p) => p.y).filter(isNum);
  const xAx = niceAxis(Math.min(...xs), Math.max(...xs), Math.max(3, Math.floor(width / 110)));
  let yLo = Math.min(...ys), yHi = Math.max(...ys);
  if (ref === "zero") { yLo = Math.min(yLo, 0); yHi = Math.max(yHi, 0); }
  for (const g of guides) { yLo = Math.min(yLo, g); yHi = Math.max(yHi, g); }
  if (ref === "identity") { yLo = Math.min(yLo, xAx.domain[0]); yHi = Math.max(yHi, xAx.domain[1]); }
  const yAx = niceAxis(yLo, yHi, 5);
  const x = linear(xAx.domain, [m.left, width - m.right]);
  const y = linear(yAx.domain, [height - m.bottom, m.top]);
  xAxis(svg, x, xAx, m.top, height - m.bottom, xTitle);
  yAxis(svg, y, yAx, m.left, width - m.right, yTitle, { zeroLine: ref === "zero" });
  const clip = (x1, y1, x2, y2) => s("line", { class: "ref", x1, y1, x2, y2 });
  for (const g of guides) svg.append(clip(m.left, y(g), width - m.right, y(g)));
  if (ref === "identity") {
    const a = Math.max(xAx.domain[0], yAx.domain[0]);
    const b = Math.min(xAx.domain[1], yAx.domain[1]);
    svg.append(clip(x(a), y(a), x(b), y(b)));
  } else if (ref && typeof ref === "object") {
    const [a, b] = xAx.domain;
    svg.append(clip(x(a), y(ref.intercept + ref.slope * a), x(b), y(ref.intercept + ref.slope * b)));
  }
  const pts = points.filter((p) => isNum(p.x) && isNum(p.y)).map((p) => ({ ...p, px: x(p.x), py: y(p.y) }));
  // Dense clouds keep a hairline ring only; a 2px ring would erase neighbours.
  const dense = pts.length > 120;
  const r = pts.length > 600 ? 3 : 4;
  const dots = s("g", { "pointer-events": "none" });
  for (const p of pts) dots.append(s("circle", { class: "dot", cx: p.px, cy: p.py, r, "stroke-width": dense ? 0.75 : 2 }));
  svg.append(dots);
  const ring = s("circle", { class: "focus-ring", r: r + 3, visibility: "hidden" });
  svg.append(ring);
  const overlay = s("rect", { x: m.left, y: m.top, width: width - m.left - m.right, height: height - m.top - m.bottom, fill: "transparent" });
  overlay.addEventListener("pointermove", (e) => {
    const box = svg.getBoundingClientRect();
    const mx = e.clientX - box.left, my = e.clientY - box.top;
    let best = null, bestD = 28 * 28;
    for (const p of pts) {
      const d = (p.px - mx) ** 2 + (p.py - my) ** 2;
      if (d < bestD) { bestD = d; best = p; }
    }
    if (!best) { ring.setAttribute("visibility", "hidden"); hideTip(); return; }
    ring.setAttribute("cx", best.px); ring.setAttribute("cy", best.py); ring.setAttribute("visibility", "visible");
    showTip(e, best.tip || { rows: [{ value: format(best.y), label: yTitle }, { value: format(best.x), label: xTitle }] });
  });
  overlay.addEventListener("pointerleave", () => { ring.setAttribute("visibility", "hidden"); hideTip(); });
  svg.append(overlay);
}

/**
 * Line chart with a snapping crosshair.
 * series: [{name, color, points: [{x, y}], muted?}], xLabels?: category labels for integer x
 */
export function lineChart(host, width, { series, xTitle, yTitle, height = 280, format = fmt, xLabels, endLabels = true, integerX = false }) {
  const labelled = series.filter((sr) => !sr.muted);
  const rightPad = endLabels && labelled.length ? Math.min(140, labelWidthFor(labelled.map((sr) => sr.name), 30, 140)) : 16;
  const m = { top: 12, right: rightPad, bottom: 46, left: 60 };
  const svg = frame(host, width, height, yTitle || "Line chart");
  const all = series.flatMap((sr) => sr.points);
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y).filter(isNum);
  const xDomain = [Math.min(...xs), Math.max(...xs)];
  const xAx = xLabels
    ? { domain: xDomain, ticks: xLabels.map((_, i) => i), format: (i) => xLabels[i] }
    : integerX
      ? (() => { const ax = niceAxis(xDomain[0], xDomain[1], Math.min(8, xDomain[1] - xDomain[0] || 1)); return { ...ax, domain: xDomain, ticks: ax.ticks.filter((t) => Number.isInteger(t) && t >= xDomain[0] && t <= xDomain[1]), format: (t) => String(t) }; })()
      : niceAxis(xDomain[0], xDomain[1], Math.max(3, Math.floor(width / 110)));
  const yAx = niceAxis(Math.min(...ys), Math.max(...ys), 5);
  const pad = xLabels ? 24 : 0;
  const x = linear(xAx.domain, [m.left + pad, width - m.right - pad]);
  const y = linear(yAx.domain, [height - m.bottom, m.top]);
  xAxis(svg, x, xAx, m.top, height - m.bottom, xTitle);
  yAxis(svg, y, yAx, m.left, width - m.right, yTitle);
  const path = (pts) => pts.filter((p) => isNum(p.y)).map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.y)}`).join("");
  for (const sr of series.filter((sr) => sr.muted)) svg.append(s("path", { class: "muted-line", d: path(sr.points) }));
  const ends = [];
  for (const sr of labelled) {
    svg.append(s("path", { class: "mark-stroke", d: path(sr.points), style: `stroke:${sr.color}` }));
    const last = [...sr.points].reverse().find((p) => isNum(p.y));
    if (last) {
      svg.append(s("circle", { cx: x(last.x), cy: y(last.y), r: 4, fill: sr.color, stroke: "var(--surface)", "stroke-width": 2 }));
      ends.push({ name: sr.name, y: y(last.y), x: x(last.x) });
    }
  }
  if (endLabels && labelled.length) {
    // Only label ends that don't collide; the legend and tooltip carry the rest.
    ends.sort((a, b) => a.y - b.y);
    let prev = -Infinity;
    for (const e of ends) {
      if (e.y - prev < 14) continue;
      svg.append(s("text", { class: "label", x: e.x + 10, y: e.y + 4, text: truncate(e.name, m.right - 14) }));
      prev = e.y;
    }
  }
  const cross = s("line", { class: "crosshair", y1: m.top, y2: height - m.bottom, visibility: "hidden" });
  const marks = s("g", { "pointer-events": "none" });
  svg.append(cross, marks);
  const xsUnique = [...new Set(xs)].sort((a, b) => a - b);
  const overlay = s("rect", { x: m.left, y: m.top, width: width - m.left - m.right, height: height - m.top - m.bottom, fill: "transparent" });
  overlay.addEventListener("pointermove", (e) => {
    const box = svg.getBoundingClientRect();
    const xv = x.invert(e.clientX - box.left);
    const nx = xsUnique.reduce((best, v) => (Math.abs(v - xv) < Math.abs(best - xv) ? v : best), xsUnique[0]);
    cross.setAttribute("x1", x(nx)); cross.setAttribute("x2", x(nx)); cross.setAttribute("visibility", "visible");
    clear(marks);
    const rows = [];
    for (const sr of labelled) {
      const p = sr.points.find((q) => q.x === nx);
      if (!p || !isNum(p.y)) continue;
      marks.append(s("circle", { cx: x(nx), cy: y(p.y), r: 4.5, fill: sr.color, stroke: "var(--surface)", "stroke-width": 2 }));
      rows.push({ value: format(p.y), label: sr.name, color: sr.color });
    }
    showTip(e, { title: `${xTitle}: ${xAx.format(nx)}`, rows });
  });
  overlay.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); clear(marks); hideTip(); });
  svg.append(overlay);
}

/**
 * Heatmap. value(i, j) -> number | null. color: scale fn with .css/.domain.
 */
export function heatmap(host, width, { rows, cols, value, color, cellText, tip, rowTitle, colTitle, maxCell = 44, legendTitle, format = fmt }) {
  const left = labelWidthFor(rows, 28, 120) + (rowTitle ? 18 : 0);
  const avail = width - left - 8;
  const cell = Math.max(8, Math.min(maxCell, Math.floor(avail / cols.length)));
  const top = 28 + (colTitle ? 16 : 0);
  const gridW = cell * cols.length;
  const height = top + cell * rows.length + 8;
  const svg = frame(host, Math.max(width, left + gridW + 8), height, legendTitle || "Heatmap");
  const colEvery = Math.ceil(28 / cell);
  const rowEvery = Math.ceil(14 / cell);
  if (colTitle) svg.append(s("text", { class: "axis-title", x: left + gridW / 2, y: 12, "text-anchor": "middle", text: colTitle }));
  cols.forEach((c, j) => {
    if (j % colEvery) return;
    svg.append(s("text", { x: left + j * cell + cell / 2, y: top - 8, "text-anchor": "middle", text: truncate(String(c), cell * colEvery - 2) }));
  });
  if (rowTitle) {
    const cy = top + (cell * rows.length) / 2;
    svg.append(s("text", { class: "axis-title", x: 10, y: cy, "text-anchor": "middle", transform: `rotate(-90 10 ${cy})`, text: rowTitle }));
  }
  rows.forEach((r, i) => {
    if (i % rowEvery) return;
    svg.append(s("text", { x: left - 8, y: top + i * cell + cell / 2 + 4, "text-anchor": "end", text: truncate(String(r), left - 12 - (rowTitle ? 18 : 0)) }));
  });
  const gap = cell >= 14 ? 2 : 1;
  for (let i = 0; i < rows.length; i++) {
    for (let j = 0; j < cols.length; j++) {
      const v = value(i, j);
      const has = isNum(v);
      const fill = has ? color(v) : null;
      const rect = s("rect", {
        class: `cell-rect cell-hit${has ? "" : " missing"}`,
        x: left + j * cell + gap / 2, y: top + i * cell + gap / 2,
        width: cell - gap, height: cell - gap, rx: cell >= 20 ? 3 : 1,
        fill,
      });
      hoverable(rect, () => (tip ? tip(i, j, v) : { title: `${rows[i]} · ${cols[j]}`, rows: [{ value: has ? format(v) : "missing" }] }));
      svg.append(rect);
      const txt = has && cellText ? cellText(v) : null;
      if (txt && cell >= 30) {
        svg.append(s("text", { class: "cell", x: left + j * cell + cell / 2, y: top + i * cell + cell / 2 + 4, "text-anchor": "middle", fill: inkOn(fill), style: `fill:${inkOn(fill)}`, "pointer-events": "none", text: txt }));
      }
    }
  }
  if (color.css) {
    const [lo, hi] = color.domain;
    host.append(h("div", { class: "scale-legend" },
      h("span", { text: format(lo) }),
      h("span", { class: "ramp", style: { background: color.css } }),
      h("span", { text: format(hi) }),
      legendTitle ? h("span", { style: { marginLeft: "6px" }, text: legendTitle }) : null));
  }
}

/**
 * One stacked horizontal bar for part-to-whole (variance partition).
 * parts: [{label, value, color}]
 */
export function partitionBar(host, width, { parts, format = fmt }) {
  const total = parts.reduce((a, p) => a + Math.max(0, p.value), 0) || 1;
  const height = 34;
  const svg = frame(host, width, height, "Variance partition");
  let x0 = 0;
  const gap = 2;
  parts.forEach((p, i) => {
    const w = (Math.max(0, p.value) / total) * width;
    const last = i === parts.length - 1;
    const segW = Math.max(0, w - (last ? 0 : gap));
    const y = 4, t = 26;
    const d = last ? hbarPath(x0, x0 + segW, y, t) : `M${x0},${y}h${segW}v${t}h${-segW}z`;
    const seg = s("path", { d, fill: p.color });
    hoverable(seg, () => ({ title: p.label, rows: [{ value: format(p.value), label: `${((p.value / total) * 100).toFixed(1)}% of total`, color: p.color }] }));
    svg.append(seg);
    const label = `${((p.value / total) * 100).toFixed(0)}%`;
    if (segW > label.length * 7.5 + 16) {
      svg.append(s("text", { class: "cell", x: x0 + 8, y: y + t / 2 + 4, style: `fill:${inkOn(p.color)}`, "pointer-events": "none", text: label }));
    }
    x0 += w;
  });
}

export const SERIES = (i) => `var(--series-${i + 1})`;
export function seriesHex(i) {
  return cssVar(`--series-${i + 1}`) || "#2a78d6";
}
