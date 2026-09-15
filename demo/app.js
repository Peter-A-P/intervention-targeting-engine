// The static demo: loads precomputed JSON and moves a cutoff through it.
//
// There is no modelling here on purpose. Every number this page can display was computed by
// `itx demo build` from the same benchmark that produces the README table, because the demo
// is hosted as static files with no backend (PLAN.md section 7). The only arithmetic below
// is reading an index out of an array and drawing a line.

"use strict";

const state = { index: null, payload: null, dataset: null, estimator: null, step: 10 };

const el = (id) => document.getElementById(id);

async function boot() {
  state.index = await fetch("data/index.json").then((r) => r.json());
  const picker = el("dataset");
  for (const entry of state.index.datasets) {
    const option = document.createElement("option");
    option.value = entry.key;
    option.textContent = `${entry.key} (${entry.n_test.toLocaleString()} test rows)`;
    picker.append(option);
  }
  picker.addEventListener("change", () => loadDataset(picker.value));
  el("estimator").addEventListener("change", (e) => {
    state.estimator = e.target.value;
    render();
  });
  el("budget").addEventListener("input", (e) => {
    state.step = Number(e.target.value);
    render();
  });
  await loadDataset(state.index.datasets[0].key);
}

async function loadDataset(key) {
  state.dataset = key;
  state.payload = await fetch(`data/${key}.json`).then((r) => r.json());

  const picker = el("estimator");
  picker.textContent = "";
  // Ordered so the two baselines sit at the bottom of the list: the whole point of the
  // project is that they are what everything else has to beat.
  const names = Object.keys(state.payload.estimators).sort(baselinesLast);
  for (const name of names) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    picker.append(option);
  }
  state.estimator = names[0];
  picker.value = state.estimator;

  const slider = el("budget");
  slider.max = String(state.payload.budgets.length);
  state.step = Math.min(state.step, state.payload.budgets.length);
  slider.value = String(state.step);
  el("level-label").textContent = `${Math.round(state.payload.level * 100)}%`;
  render();
}

function baselinesLast(a, b) {
  const rank = (n) => (n === "outcome-ranking" ? 1 : n === "random" ? 2 : 0);
  return rank(a) - rank(b) || a.localeCompare(b);
}

function render() {
  const p = state.payload;
  const i = state.step - 1;
  const budget = p.budgets[i];
  const curve = p.estimators[state.estimator];
  const selected = curve.dr;
  const random = p.random.dr;
  // ceil, like rank_and_cut: the set the gain describes is the smallest one covering the
  // budget share, so a 10% budget on 961 rows is 97 rows, not 96.
  const treated = Math.max(1, Math.ceil(p.n_test * budget));

  el("budget-label").textContent = `${Math.round(budget * 100)}%`;

  const gain = selected.value[i];
  const versus = gain - random[i];
  const digits = p.outcome_is_binary ? 4 : 3;
  el("headline").innerHTML = `
    <div class="stat"><span class="figure">${treated.toLocaleString()}</span>
      <span class="caption">units treated, of ${p.n_test.toLocaleString()}</span></div>
    <div class="stat"><span class="figure">${fmt(gain, digits)}</span>
      <span class="caption">outcome bought per head
        (${fmt(selected.low[i], digits)} to ${fmt(selected.high[i], digits)})</span></div>
    <div class="stat ${versus >= 0 ? "up" : "down"}"><span class="figure">${fmt(versus, digits)}</span>
      <span class="caption">against spending the same budget at random
        (${fmt(selected.low[i] - random[i], digits)} to ${fmt(selected.high[i] - random[i], digits)})</span></div>`;

  el("list-note").textContent =
    treated <= curve.top.length
      ? `The ${treated.toLocaleString()} rows this budget treats, best first.`
      : `The first ${curve.top.length} of ${treated.toLocaleString()} rows this budget treats.` +
        ` The rest are not shipped: the whole ranking would be a multi-megabyte page load.`;

  const body = el("units").querySelector("tbody");
  body.textContent = "";
  for (const [rank, unit] of curve.top.slice(0, treated).entries()) {
    const tr = document.createElement("tr");
    for (const cell of [rank + 1, unit.row, fmt(unit.uplift, 4), unit.cost.toFixed(1)]) {
      const td = document.createElement("td");
      td.textContent = String(cell);
      tr.append(td);
    }
    body.append(tr);
  }

  el("provenance").textContent =
    `${p.dataset}, one split at seed ${p.seed} of the five the results table averages,` +
    ` ${p.n_test.toLocaleString()} held-out rows, ${p.n_treated.toLocaleString()} of them treated in the data.` +
    ` Bands are ${Math.round(p.level * 100)}% percentile intervals from ${p.n_resamples} bootstrap resamples` +
    ` of this split, so they cover sampling within it and not variation between splits.`;

  drawChart(p.budgets, selected, random, i);
}

function fmt(x, digits) {
  return (x >= 0 ? "+" : "") + x.toFixed(digits);
}

// Inline SVG rather than a charting library: one line, one band and one rule do not justify
// a dependency on a page whose whole argument is that it is a static file you can read.
function drawChart(budgets, selected, random, cursor) {
  const W = 720;
  const H = 300;
  const pad = { top: 16, right: 16, bottom: 40, left: 64 };

  const ys = [...selected.low, ...selected.high, ...random, 0];
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  const span = hi - lo || 1;
  const x = (i) => pad.left + (i / (budgets.length - 1)) * (W - pad.left - pad.right);
  const y = (v) => H - pad.bottom - ((v - lo) / span) * (H - pad.top - pad.bottom);

  const line = (vals) => vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const band =
    selected.high.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("") +
    selected.low.map((v, i) => `L${x(selected.low.length - 1 - i).toFixed(1)},${y(selected.low[selected.low.length - 1 - i]).toFixed(1)}`).join("") +
    "Z";

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const v = lo + f * span;
    return `<line class="grid" x1="${pad.left}" y1="${y(v).toFixed(1)}" x2="${W - pad.right}" y2="${y(v).toFixed(1)}"></line>
            <text class="tick" x="${pad.left - 8}" y="${(y(v) + 4).toFixed(1)}" text-anchor="end">${v.toFixed(3)}</text>`;
  });
  const xticks = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const i = Math.round(f * (budgets.length - 1));
    return `<text class="tick" x="${x(i).toFixed(1)}" y="${H - pad.bottom + 20}" text-anchor="middle">${Math.round(budgets[i] * 100)}%</text>`;
  });

  el("chart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
      ${ticks.join("")}
      ${lo < 0 && hi > 0 ? `<line class="zero" x1="${pad.left}" y1="${y(0).toFixed(1)}" x2="${W - pad.right}" y2="${y(0).toFixed(1)}"></line>` : ""}
      <path class="band" d="${band}"></path>
      <path class="random" d="${line(random)}"></path>
      <path class="selected" d="${line(selected.value)}"></path>
      <line class="cursor" x1="${x(cursor).toFixed(1)}" y1="${pad.top}" x2="${x(cursor).toFixed(1)}" y2="${H - pad.bottom}"></line>
      <circle class="dot" cx="${x(cursor).toFixed(1)}" cy="${y(selected.value[cursor]).toFixed(1)}" r="4"></circle>
      ${xticks.join("")}
      <text class="axis" x="${(W / 2).toFixed(0)}" y="${H - 6}" text-anchor="middle">share of the population treated</text>
    </svg>`;
}

boot();
