// Number formatting.

const isNum = (v) => typeof v === "number" && Number.isFinite(v);

/** Format with `digits` significant digits (fixed notation for ordinary magnitudes). */
export function fmt(v, digits = 4) {
  if (!isNum(v)) return "—";
  if (v === 0) return "0";
  const abs = Math.abs(v);
  if (abs >= 1e6 || abs < 1e-4) return v.toExponential(Math.max(0, digits - 1));
  const decimals = Math.max(0, digits - 1 - Math.floor(Math.log10(abs)));
  return v.toLocaleString("en-US", { minimumFractionDigits: Math.min(decimals, 6), maximumFractionDigits: Math.min(decimals, 6) });
}

/** Fixed number of decimals, thousands separated. */
export function fixed(v, decimals = 2) {
  if (!isNum(v)) return "—";
  return v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

export function int(v) {
  return isNum(v) ? Math.round(v).toLocaleString("en-US") : "—";
}

export function pct(v, decimals = 0) {
  return isNum(v) ? `${(v * 100).toFixed(decimals)}%` : "—";
}

export function pValue(p) {
  if (!isNum(p)) return "—";
  if (p < 1e-4) return "< 0.0001";
  return p.toFixed(4);
}

export function stars(p) {
  if (!isNum(p)) return "";
  return p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : p < 0.1 ? "." : "";
}

/** Tick formatter for an axis with the given step. */
export function tickFormatter(step) {
  const decimals = step > 0 ? Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9))) : 0;
  return (v) => {
    const r = Math.abs(v) < step * 1e-6 ? 0 : v;
    return r.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };
}

export function duration(ms) {
  if (!isNum(ms)) return "";
  return ms < 1000 ? `${Math.max(1, Math.round(ms))} ms` : `${(ms / 1000).toFixed(2)} s`;
}

export { isNum };
