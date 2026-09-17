// Tiny DOM helpers. Data-derived strings (column names, levels) are always
// inserted as text nodes, never as HTML.

const SVG_NS = "http://www.w3.org/2000/svg";

function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

/**
 * Create an HTML element.
 * props: class, text, on: {event: fn}, attrs: {name: value}, style: {prop: value},
 * plus any direct DOM property (value, checked, type, ...).
 */
export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "on") for (const [ev, fn] of Object.entries(value)) node.addEventListener(ev, fn);
    else if (key === "attrs") for (const [a, v] of Object.entries(value)) { if (v !== undefined && v !== null && v !== false) node.setAttribute(a, v === true ? "" : v); }
    else if (key === "style") Object.assign(node.style, value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else node[key] = value;
  }
  append(node, children);
  return node;
}

/** Create an SVG element; attributes are set verbatim, `text` sets textContent. */
export function s(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "text") node.textContent = value;
    else if (key === "on") for (const [ev, fn] of Object.entries(value)) node.addEventListener(ev, fn);
    else node.setAttribute(key, value);
  }
  append(node, children);
  return node;
}

/** Build an inline SVG icon from path data (24x24 stroke icons). */
export function icon(paths, size = 16) {
  return s("svg", {
    width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
    "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true",
  }, ...[paths].flat().map((d) => s("path", { d })));
}

export const ICONS = {
  upload: ["M12 16V4", "M7 9l5-5 5 5", "M4 20h16"],
  tree: ["M12 3v6", "M6 15v-3h12v3", "M12 9v3", "M4 15h4v5H4z", "M16 15h4v5h-4z", "M10 3h4v4h-4z"],
  grid: ["M3 3h18v18H3z", "M3 9h18", "M3 15h18", "M9 3v18", "M15 3v18"],
  lines: ["M3 17l5-6 4 3 7-9", "M3 21h18"],
  sprout: ["M12 21v-9", "M12 12C12 7 8 5 4 5c0 5 3 7 8 7z", "M12 14c0-4 3-6 8-6 0 4-3 6-8 6z"],
  alert: ["M12 9v4", "M12 17h.01", "M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"],
  info: ["M12 22a10 10 0 100-20 10 10 0 000 20z", "M12 16v-4", "M12 8h.01"],
  download: ["M12 4v12", "M7 11l5 5 5-5", "M4 20h16"],
  play: ["M7 4l13 8-13 8z"],
  sun: ["M12 17a5 5 0 100-10 5 5 0 000 10z", "M12 1v2", "M12 21v2", "M4.2 4.2l1.4 1.4", "M18.4 18.4l1.4 1.4", "M1 12h2", "M21 12h2", "M4.2 19.8l1.4-1.4", "M18.4 5.6l1.4-1.4"],
  moon: ["M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"],
  check: ["M20 6L9 17l-5-5"],
  x: ["M18 6L6 18", "M6 6l12 12"],
};

export function clear(node) {
  while (node.firstChild) node.firstChild.remove();
  return node;
}

export function download(filename, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function csvField(value) {
  const str = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
}

export function toCsv(header, rows) {
  return [header, ...rows].map((r) => r.map(csvField).join(",")).join("\n") + "\n";
}

export function segmented(options, current, onChange, { block = false, label } = {}) {
  // The control tracks its own selection, so it stays correct whether or not
  // the caller re-renders it after a change.
  let selected = current;
  const buttons = options.map(([value, text]) => h("button", {
    type: "button",
    text,
    attrs: { "aria-pressed": String(value === selected) },
    on: {
      click: () => {
        if (value === selected) return;
        selected = value;
        buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i][0] === selected)));
        onChange(value);
      },
    },
  }));
  return h("div", { class: `segmented${block ? " block" : ""}`, attrs: { role: "group", "aria-label": label } }, buttons);
}

export function select(options, current, onChange, props = {}) {
  const node = h("select", { ...props, on: { change: (e) => onChange(e.target.value) } },
    options.map(([value, text]) => h("option", { value, text, selected: value === current })));
  return node;
}
