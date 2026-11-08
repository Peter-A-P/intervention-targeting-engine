// The static demo: loads precomputed JSON and moves a cutoff through it.
//
// There is no modelling here on purpose. Every number that came out of a fitted model was
// computed by `itx demo build` from the same benchmark that produces the README table, because
// the demo is hosted as static files with no backend (PLAN.md section 7). The one exception is
// the sample-size calculator in `power.js`, which is closed-form arithmetic rather than a model
// and is tested against the Python it mirrors.

"use strict";

const state = {
  index: null,
  payload: null,
  dataset: null,
  estimator: null,
  highlight: "",
  step: 10,
};

const el = (id) => document.getElementById(id);

// One hue per family, and the hues live in style.css rather than here, so dark mode follows
// the page. Each method's marks carry a class `c-<method>` that sets the CSS `color`, and the
// series, band, dot, label and swatch rules paint with currentColor. The five LightGBM
// meta-learners are steps of one blue; the neural network and the two baselines each get
// their own hue. Random is red and dashed: it is the line everything has to clear, and it must
// be findable by someone who cannot separate red from green, which is why it is also the only
// dashed line and is labelled directly on the chart.
function colourClass(name) {
  return `c-${name in ESTIMATORS ? name : "other"}`;
}

// The page's whole state is three values, so it can live in the URL. Sharing a view then
// costs nothing and the back button works.
function readUrl() {
  const q = new URLSearchParams(location.search);
  return {
    dataset: q.get("dataset"),
    method: q.get("method"),
    budget: q.get("budget") ? Number(q.get("budget")) : null,
    highlight: q.get("highlight"),
  };
}

function writeUrl() {
  const q = new URLSearchParams();
  q.set("dataset", state.dataset);
  q.set("method", state.estimator);
  q.set("budget", String(state.step));
  if (state.highlight) q.set("highlight", state.highlight);
  history.replaceState(null, "", `${location.pathname}?${q}`);
}

async function boot() {
  state.index = await fetch("data/index.json").then((r) => {
    if (!r.ok) throw new Error(`data/index.json answered ${r.status}`);
    return r.json();
  });
  const wanted = readUrl();

  const picker = el("dataset");
  for (const entry of orderedDatasets(state.index.datasets)) {
    const meta = DATASETS[entry.key];
    const option = document.createElement("option");
    option.value = entry.key;
    option.textContent = meta
      ? `${meta.title} (${entry.n_test.toLocaleString()} people)`
      : entry.key;
    picker.append(option);
  }
  picker.addEventListener("change", () => loadDataset(picker.value));


  el("highlight").addEventListener("change", (e) => {
    state.highlight = e.target.value;
    renderAll();
    writeUrl();
  });
  el("estimator").addEventListener("change", (e) => {
    state.estimator = e.target.value;
    render();
    writeUrl();
  });
  el("budget").addEventListener("input", (e) => {
    state.step = Number(e.target.value);
    render();
    writeUrl();
  });

  renderGlossary();
  bootPower();
  watchChartKeys();

  const known = state.index.datasets.map((d) => d.key);
  const first = known.includes(wanted.dataset)
    ? wanted.dataset
    : orderedDatasets(state.index.datasets)[0].key;
  if (wanted.method) state.estimator = wanted.method;
  if (wanted.highlight) state.highlight = wanted.highlight;
  if (wanted.budget) state.step = wanted.budget;
  await loadDataset(first, { keepSelection: true });
}

// Hillstrom first: it is the most ordinary case and the easiest to understand cold. The two
// that refuted something come next, then the failure, then the simulated pair.
function orderedDatasets(entries) {
  const order = ["hillstrom", "criteo", "ieee-fraud", "acic", "lenta", "ihdp"];
  return [...entries].sort(
    (a, b) => order.indexOf(a.key) - order.indexOf(b.key) || a.key.localeCompare(b.key)
  );
}

async function loadDataset(key, { keepSelection = false } = {}) {
  state.dataset = key;
  state.payload = await fetch(`data/${key}.json`).then((r) => {
    if (!r.ok) throw new Error(`data/${key}.json answered ${r.status}`);
    return r.json();
  });
  state.cursorAll = null;
  el("dataset").value = key;

  const names = presentEstimators();

  const method = el("estimator");
  method.textContent = "";
  for (const name of names) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${ESTIMATORS[name]?.title ?? name}: ${ESTIMATORS[name]?.short ?? ""}`;
    method.append(option);
  }
  state.estimator = names.includes(state.estimator) ? state.estimator : names[0];
  method.value = state.estimator;

  const highlight = el("highlight");
  highlight.textContent = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "all methods";
  highlight.append(none);
  for (const name of names) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = ESTIMATORS[name]?.title ?? name;
    highlight.append(option);
  }
  if (!keepSelection || !names.includes(state.highlight)) state.highlight = "";
  highlight.value = state.highlight;

  const slider = el("budget");
  slider.max = String(state.payload.budgets.length);
  state.step = Math.min(Math.max(1, state.step), state.payload.budgets.length);
  slider.value = String(state.step);

  renderDatasetCard();
  renderAll();
  render();
  writeUrl();
}

function presentEstimators() {
  return ESTIMATOR_ORDER.filter((n) => n in state.payload.estimators);
}

// ---------------------------------------------------------------- units and formatting

function meta() {
  return DATASETS[state.dataset] ?? { units: "outcome per head", scale: 1, decimals: 3 };
}

// The number as a reader can hold it. "+0.0055 outcome per head" is a quantity with no noun
// attached; "5.5 extra conversions per 1,000 people" is the same fact, readable.
function scaled(value) {
  const m = meta();
  const shown = value * (m.scale ?? 1);
  const digits = m.decimals ?? 3;
  if (m.currency) {
    const sign = shown < 0 ? "-" : "+";
    return `${sign}$${Math.abs(shown).toFixed(digits)}`;
  }
  return (shown >= 0 ? "+" : "") + shown.toFixed(digits);
}

function unitLabel() {
  return meta().units ?? "outcome per head";
}

// The same unit with the qualifying phrase dropped, for the three hero tiles where it appears
// three times over and the full form takes two lines each. The dataset card states it in full.
function shortUnitLabel() {
  return meta().unitsShort ?? unitLabel();
}

function axisLabel() {
  const m = meta();
  if (m.currency) return "dollars saved per transaction";
  if ((m.scale ?? 1) > 1) return unitLabel();
  return unitLabel();
}

// ---------------------------------------------------------------- cards and copy

function renderDatasetCard() {
  const m = meta();
  const p = state.payload;
  el("dataset-card").innerHTML = `
    <div class="card-head">
      <h3>${m.title ?? state.dataset}</h3>
      <span class="tag ${m.kind?.startsWith("Real randomised") ? "tag-real" : "tag-sim"}">${m.kind ?? ""}</span>
    </div>
    <p>${m.what ?? ""}</p>
    <dl class="facts">
      <div><dt>Outcome measured</dt><dd>${m.outcome ?? ""}</dd></div>
      <div><dt>Numbers below are</dt><dd>${m.units ?? ""}</dd></div>
      <div><dt>People in this split</dt><dd>${p.n_test.toLocaleString()} held out, ${p.n_treated.toLocaleString()} of them treated</dd></div>
    </dl>
    <p class="why"><strong>Why it is here.</strong> ${m.why ?? ""}</p>
    <p class="honest"><strong>What to watch out for.</strong> ${m.honest ?? ""}</p>`;

  el("units-explainer").innerHTML =
    `On this dataset the numbers are <strong>${m.units}</strong>. ${
      m.currency
        ? "A value of +$2.00 means the budget saved two dollars of fraud loss for every transaction in the population, not for every transaction reviewed."
        : (m.scale ?? 1) > 1
          ? `A value of +${(2).toFixed(m.decimals ?? 1)} means the budget bought two extra of them per thousand people in the population, counting everybody, not only the people treated.`
          : "The outcome is simulated, so the units are arbitrary: they are useful for comparing methods against each other and mean nothing in the world."
    }`;
}

function renderMethodCard() {
  const m = ESTIMATORS[state.estimator] ?? {};
  el("method-card").innerHTML = `
    <div class="card-head">
      <h3>${m.title ?? state.estimator}</h3>
      <span class="tag tag-${m.family ?? "other"}">${familyLabel(m.family)}</span>
    </div>
    <p>${m.what ?? ""}</p>
    <div class="pros">
      <p><strong>Strength.</strong> ${m.strength ?? ""}</p>
      <p><strong>Weakness.</strong> ${m.weakness ?? ""}</p>
    </div>`;
}

function familyLabel(family) {
  return (
    { lightgbm: "Gradient-boosted trees", neural: "Neural network", baseline: "Baseline", random: "Baseline" }[
      family
    ] ?? "Method"
  );
}

function renderGlossary() {
  el("glossary").innerHTML = Object.entries(GLOSSARY)
    .map(([term, definition]) => `<dt>${term}</dt><dd>${definition}</dd>`)
    .join("");
}

// ---------------------------------------------------------------- the overview chart

function renderAll() {
  const p = state.payload;
  const names = presentEstimators().filter((n) => n !== "random");
  const series = names.map((name) => ({
    name,
    values: p.estimators[name].dr.value,
    colour: colourClass(name),
  }));
  const random = p.random.dr;

  drawMulti("chart-all", p.budgets, series, random);

  // The colour is a class, set in style.css. A style attribute in markup would be blocked by
  // the page's own content security policy, which once left every swatch blank on the live
  // site while looking correct on a local server that sends no policy.
  el("legend-all").innerHTML =
    series
      .map(
        (s) =>
          `<button type="button" class="legend-item ${state.highlight === s.name ? "on" : ""}" data-name="${s.name}" aria-pressed="${state.highlight === s.name}">
             <span class="swatch ${s.colour}"></span>${ESTIMATORS[s.name]?.title ?? s.name}
           </button>`
      )
      .join("") +
    `<span class="legend-item static"><span class="swatch dashed c-random"></span>Random targeting</span>`;
  for (const button of el("legend-all").querySelectorAll("button")) {
    button.addEventListener("click", () => {
      state.highlight = state.highlight === button.dataset.name ? "" : button.dataset.name;
      el("highlight").value = state.highlight;
      renderAll();
    });
  }

  el("caption-all").textContent =
    `Every method on ${meta().title ?? state.dataset}, in ${unitLabel()}. A line above the red ` +
    `dashed one beats random targeting at that budget. Pick a method under "Highlight" to pull ` +
    `it out of the bundle.`;
  readAll(state.cursorAll ?? tenPercent());

  renderVerdict(series, random);
  renderHero(series, random);
}

// The sentence a visitor should leave with, computed rather than written, so it cannot go stale
// when a number changes.
function tenPercent() {
  return Math.max(0, Math.round(0.1 * (state.payload.budgets.length - 1)) - 1);
}

function renderVerdict(series, random) {
  const p = state.payload;
  const i = tenPercent();
  const best = series.reduce((a, b) => (b.values[i] > a.values[i] ? b : a));
  const risk = series.find((s) => s.name === "outcome-ranking");
  if (!risk) {
    el("verdict").innerHTML = "";
    return;
  }

  // Every claim below is gated on the interval rather than the point estimate. A single split
  // will happily put one line under another by an amount that means nothing, and announcing
  // from that that somebody's risk model destroys value is precisely the failure this project
  // exists to demonstrate. Where the interval covers the random line, the honest answer is
  // that this split cannot tell, and that is what it says.
  const band = p.estimators["outcome-ranking"].dr;
  const bestBand = p.estimators[best.name].dr;
  const clearlyWorse = band.high[i] < random[i];
  const clearlyBetter = band.low[i] > random[i];
  const upliftWins = bestBand.low[i] > band.high[i];
  const title = (n) => ESTIMATORS[n]?.title ?? n;

  let line;
  let tone = "";
  if (clearlyWorse) {
    tone = "bad";
    line =
      `<strong>On this dataset, ranking by risk is worse than picking at random.</strong> ` +
      `At a 10% budget it buys ${scaled(risk.values[i])} against random targeting's ` +
      `${scaled(random[i])}, and its whole confidence range sits below the random line. An ` +
      `organisation doing the ordinary thing here would be destroying value while its report ` +
      `looked healthy.`;
  } else if (!clearlyBetter) {
    line =
      `<strong>On this dataset, ranking by risk cannot be told apart from picking at ` +
      `random.</strong> At a 10% budget it buys ${scaled(risk.values[i])} against random ` +
      `targeting's ${scaled(random[i])}, and its confidence range covers the random line, so ` +
      `the difference has not been shown. ` +
      (upliftWins
        ? `${title(best.name)} does clear it, at ${scaled(best.values[i])}.`
        : `Nor is any other method here measurably ahead of it at this budget.`);
  } else if (upliftWins) {
    tone = "good";
    line =
      `<strong>On this dataset, uplift modelling is worth the work.</strong> ` +
      `${title(best.name)} buys ${scaled(best.values[i])} at a 10% budget, clear of ranking ` +
      `by risk at ${scaled(risk.values[i])}, which in turn clears random targeting at ` +
      `${scaled(random[i])}.`;
  } else {
    line =
      `<strong>On this dataset, ranking by risk is as good as anything else.</strong> ` +
      `It buys ${scaled(risk.values[i])} at a 10% budget, clear of random targeting, and no ` +
      `method here is measurably ahead of it, though the best point estimate is ` +
      `${scaled(best.values[i])}. On data like this, the uplift models cost more to build and ` +
      `buy nothing measurable over one risk model.`;
  }
  el("verdict").innerHTML = line;
  el("verdict").className = `verdict ${tone}`;
}

function renderHero(series, random) {
  const p = state.payload;
  const i = tenPercent();
  const best = series.reduce((a, b) => (b.values[i] > a.values[i] ? b : a));
  const risk = series.find((s) => s.name === "outcome-ranking");
  el("hero-best").textContent = scaled(best.values[i]);
  el("hero-risk").textContent = risk ? scaled(risk.values[i]) : "...";
  el("hero-random").textContent = scaled(random[i]);
  // Every estimate carries its interval. Random targeting's line is the closed-form
  // expectation, so it has none to show.
  const band = (name) => p.estimators[name].dr;
  const range = (name) => `range ${scaled(band(name).low[i])} to ${scaled(band(name).high[i])}`;
  el("hero-best-label").textContent =
    `${ESTIMATORS[best.name]?.title ?? best.name}, the best of the eight at a 10% budget, ${range(best.name)}`;
  el("hero-risk-label").textContent = risk
    ? `ranking people by risk, same budget, ${range("outcome-ranking")}`
    : "ranking people by risk is not in this dataset's results";
  // The tone goes on the tile, which colours both its top rule and its figure.
  el("hero-best").parentElement.className = `point ${best.values[i] > random[i] ? "up" : "down"}`;
  el("hero-risk").parentElement.className = `point ${
    risk && risk.values[i] < random[i] ? "down" : "up"
  }`;

  // A figure with no noun on it is not a measurement. All three are in the same unit, and the
  // unit changes with the dataset, so it is written under each one rather than assumed.
  for (const node of document.querySelectorAll(".point-unit")) node.textContent = shortUnitLabel();
  // Whether the gap between the three figures means anything is decided by the same intervals
  // the verdict below uses, so the first screen cannot claim more than the section does.
  const riskBand = risk ? band("outcome-ranking") : null;
  const separated =
    band(best.name).low[i] > random[i] ||
    (riskBand !== null && (riskBand.low[i] > random[i] || riskBand.high[i] < random[i]));
  const context = el("hero-context");
  context.textContent = "";
  const strong = document.createElement("strong");
  strong.textContent = meta().title ?? state.dataset;
  context.append(
    "These three figures are for ",
    strong,
    separated
      ? ", the dataset selected below."
      : ", the dataset selected below, and on it the gap between them is inside the noise."
  );
}

// ---------------------------------------------------------------- the single-method chart

function render() {
  const p = state.payload;
  const i = state.step - 1;
  const budget = p.budgets[i];
  const curve = p.estimators[state.estimator];
  const selected = curve.dr;
  const random = p.random.dr;
  const treated = Math.max(1, Math.ceil(p.n_test * budget));

  el("budget-label").textContent = `${Math.round(budget * 100)}%`;
  renderMethodCard();

  const gain = selected.value[i];
  const versus = gain - random[i];
  const clearsRandom = selected.low[i] > random[i];

  el("headline").innerHTML = `
    <div class="stat"><span class="figure">${treated.toLocaleString()}</span>
      <span class="caption">people treated, out of ${p.n_test.toLocaleString()}</span></div>
    <div class="stat"><span class="figure">${scaled(gain)}</span>
      <span class="caption">${unitLabel()}<br><span class="range">range ${scaled(selected.low[i])} to ${scaled(selected.high[i])}</span></span></div>
    <div class="stat ${versus >= 0 ? "up" : "down"}"><span class="figure">${scaled(versus)}</span>
      <span class="caption">better than random targeting<br>
        <span class="range">${clearsRandom ? "clear of random at this budget" : "not clear of random: the range overlaps"}</span></span></div>`;

  el("list-note").textContent =
    treated <= curve.top.length
      ? `The ${treated.toLocaleString()} rows this budget treats, best first.`
      : `The first ${curve.top.length} of ${treated.toLocaleString()} rows this budget treats.` +
        ` The rest are not shipped: the whole ranking would be a multi-megabyte page load.`;

  const body = el("units").querySelector("tbody");
  body.textContent = "";
  for (const [rank, unit] of curve.top.slice(0, treated).entries()) {
    const tr = document.createElement("tr");
    for (const cell of [rank + 1, unit.row, unit.uplift.toFixed(4), unit.cost.toFixed(1)]) {
      const td = document.createElement("td");
      td.textContent = String(cell);
      tr.append(td);
    }
    body.append(tr);
  }

  el("caption-one").textContent =
    `${ESTIMATORS[state.estimator]?.title ?? state.estimator} on ${meta().title ?? state.dataset}. ` +
    `Where the shaded band, the 95% confidence range, sits clear above the red dashed line, the ` +
    `method beats random targeting. The vertical line is your selected budget.`;

  el("provenance").textContent =
    `${p.dataset}, one split at seed ${p.seed} of the five the results table averages,` +
    ` ${p.n_test.toLocaleString()} held-out rows, ${p.n_treated.toLocaleString()} of them treated in the data.` +
    ` Bands are ${Math.round(p.level * 100)}% percentile intervals from ${p.n_resamples} bootstrap resamples` +
    ` of this split, so they cover sampling within it and not variation between splits.`;

  drawOne(p.budgets, selected, random, i);
  readOne(i);
}

// ---------------------------------------------------------------- readouts
//
// A line under each chart says the value under the pointer, the keyboard focus or a tap. The
// numbers are read from the same JSON the lines are drawn from; nothing is computed here.

function readAll(i) {
  const p = state.payload;
  const names = [state.highlight, "outcome-ranking"].filter(
    (n, k, all) => n && p.estimators[n] && all.indexOf(n) === k
  );
  const best = presentEstimators()
    .filter((n) => n !== "random")
    .reduce((a, b) => (p.estimators[b].dr.value[i] > p.estimators[a].dr.value[i] ? b : a));
  if (!names.includes(best)) names.push(best);
  const out = el("readout-all");
  out.textContent = "";
  const when = document.createElement("span");
  when.className = "when";
  when.textContent = `Treat the top ${Math.round(p.budgets[i] * 100)}%`;
  out.append(when);
  const pair = (label, text) => {
    const span = document.createElement("span");
    span.className = "pair";
    const key = document.createElement("span");
    key.className = "key";
    key.textContent = `${label} `;
    span.append(key, text);
    out.append(span, " ");
  };
  for (const name of names) {
    const band = p.estimators[name].dr;
    pair(
      ESTIMATORS[name]?.title ?? name,
      `${scaled(band.value[i])} (${scaled(band.low[i])} to ${scaled(band.high[i])})`
    );
  }
  pair("Random", scaled(p.random.dr[i]));
}

function readOne(i) {
  const p = state.payload;
  const band = p.estimators[state.estimator].dr;
  el("readout-one").textContent =
    `Treat the top ${Math.round(p.budgets[i] * 100)}%: ` +
    `${ESTIMATORS[state.estimator]?.title ?? state.estimator} ${scaled(band.value[i])} ` +
    `(${scaled(band.low[i])} to ${scaled(band.high[i])}), random ${scaled(p.random.dr[i])}`;
}

// Pointer, touch and arrow keys all move the same cursor. On the overview chart the cursor is
// the chart's own; on the single-method chart it is the budget slider, so the two stay in step.
function budgetUnder(target, clientX) {
  const svg = el(target).querySelector("svg");
  if (!svg) return null;
  const box = svg.getBoundingClientRect();
  const x = ((clientX - box.left) / box.width) * W;
  const n = state.payload.budgets.length;
  const f = (x - PAD.left) / (W - PAD.left - PAD.right);
  return Math.min(n - 1, Math.max(0, Math.round(f * (n - 1))));
}

function watchChartKeys() {
  const all = el("chart-all");
  const one = el("chart");
  const moveAll = (i) => {
    if (i === null || !state.payload) return;
    state.cursorAll = i;
    drawCursorAll(i);
    readAll(i);
  };
  const moveOne = (i) => {
    if (i === null || !state.payload) return;
    state.step = i + 1;
    el("budget").value = String(state.step);
    render();
    writeUrl();
  };
  all.addEventListener("pointermove", (e) => moveAll(budgetUnder("chart-all", e.clientX)));
  all.addEventListener("pointerdown", (e) => moveAll(budgetUnder("chart-all", e.clientX)));
  one.addEventListener("pointerdown", (e) => moveOne(budgetUnder("chart", e.clientX)));
  one.addEventListener("pointermove", (e) => {
    if (e.buttons) moveOne(budgetUnder("chart", e.clientX));
  });
  const keys = (move, current) => (e) => {
    if (!state.payload) return;
    const n = state.payload.budgets.length;
    const step = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
    if (e.key === "Home") move(0);
    else if (e.key === "End") move(n - 1);
    else if (step) move(Math.min(n - 1, Math.max(0, current() + step)));
    else return;
    e.preventDefault();
  };
  all.addEventListener("keydown", keys(moveAll, () => state.cursorAll ?? tenPercent()));
  one.addEventListener("keydown", keys(moveOne, () => state.step - 1));
}

function drawCursorAll(i) {
  const line = el("chart-all").querySelector("svg .cursor");
  if (!line) return;
  const x = PAD.left + (i / (state.payload.budgets.length - 1)) * (W - PAD.left - PAD.right);
  line.setAttribute("x1", x.toFixed(1));
  line.setAttribute("x2", x.toFixed(1));
  line.classList.remove("off");
}

// ---------------------------------------------------------------- drawing
//
// Inline SVG rather than a charting library: a handful of lines, a band and a rule do not
// justify a dependency on a page whose whole argument is that it is a static file you can read.

const W = 760;
const H = 360;
const PAD = { top: 20, right: 116, bottom: 56, left: 78 };

// Round numbers on the axis, because 11.1 and 22.3 are the arithmetic showing through rather
// than anything a reader wants to read.
function niceTicks(lo, hi, count = 6) {
  const raw = (hi - lo) / (count - 1);
  const magnitude = 10 ** Math.floor(Math.log10(Math.abs(raw) || 1));
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((candidate) => candidate >= raw) ??
    magnitude * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
  return out.length >= 2 ? out : [lo, hi];
}

function axes(lo, span, budgets, yLabel) {
  const y = (v) => H - PAD.bottom - ((v - lo) / span) * (H - PAD.top - PAD.bottom);
  const x = (i) => PAD.left + (i / (budgets.length - 1)) * (W - PAD.left - PAD.right);
  const m = meta();
  const fmtTick = (v) => {
    const shown = v * (m.scale ?? 1);
    return m.currency ? `$${shown.toFixed(0)}` : shown.toFixed(m.decimals ?? 2);
  };
  const ticks = niceTicks(lo, lo + span)
    .map(
      (v) =>
        `<line class="grid" x1="${PAD.left}" y1="${y(v).toFixed(1)}" x2="${W - PAD.right}" y2="${y(v).toFixed(1)}"></line>
         <text class="tick" x="${PAD.left - 10}" y="${(y(v) + 4).toFixed(1)}" text-anchor="end">${fmtTick(v)}</text>`
    )
    .join("");
  // Round budgets on the axis, found by value rather than by position: the budgets start at
  // 2% and go up in twos, so a quarter of the way along is 26%, which is the arithmetic
  // showing through.
  const xticks = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    .map((share) => {
      const i = budgets.reduce(
        (best, b, k) => (Math.abs(b - share) < Math.abs(budgets[best] - share) ? k : best),
        0
      );
      return `<text class="tick" x="${x(i).toFixed(1)}" y="${H - PAD.bottom + 22}" text-anchor="middle">${Math.round(budgets[i] * 100)}%</text>`;
    })
    .join("");
  const labels = `
    <text class="axis" x="${((W - PAD.right + PAD.left) / 2).toFixed(0)}" y="${H - 10}" text-anchor="middle">share of the population treated</text>
    <text class="axis" transform="rotate(-90 18 ${(H / 2).toFixed(0)})" x="18" y="${(H / 2).toFixed(0)}" text-anchor="middle">${yLabel}</text>`;
  return { x, y, svg: ticks + xticks + labels };
}

function drawMulti(target, budgets, series, random) {
  const all = [...series.flatMap((s) => s.values), ...random, 0];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo || 1;
  const { x, y, svg } = axes(lo, span, budgets, axisLabel());
  const path = (vals) =>
    vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");

  const dim = state.highlight !== "";
  const lines = series
    .map(
      (s) =>
        `<path class="series ${s.colour} ${dim && state.highlight !== s.name && s.name !== "outcome-ranking" ? "faded" : ""}" d="${path(s.values)}"
               stroke-width="${state.highlight === s.name ? 3.5 : 2}"></path>`
    )
    .join("");

  // Direct labels at the right edge so the chart is readable without cross-referencing a
  // legend, and so it survives being read by someone who cannot distinguish the blues.
  // Only the lines a reader needs to find by name: random, which is the floor, risk ranking,
  // which is the comparison, and whatever is highlighted. Labelling all eight stacks them on
  // top of each other wherever the curves converge, which is most of the right-hand side.
  const named = [
    { name: "random", values: random, colour: colourClass("random") },
    ...series.filter((s) => s.name === "outcome-ranking" || s.name === state.highlight),
  ];
  let previous = -Infinity;
  const labels = named
    .map((s) => ({ s, yy: y(s.values[s.values.length - 1]) }))
    .sort((a, b) => a.yy - b.yy)
    .map(({ s, yy }) => {
      const place = Math.max(yy, previous + 14);
      previous = place;
      return `<text class="end-label halo ${s.colour}" x="${W - PAD.right + 8}" y="${place.toFixed(1)}">${ESTIMATORS[s.name]?.title ?? s.name}</text>`;
    })
    .join("");

  el(target).innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
      ${svg}
      ${lo < 0 && hi > 0 ? `<line class="zero" x1="${PAD.left}" y1="${y(0).toFixed(1)}" x2="${W - PAD.right}" y2="${y(0).toFixed(1)}"></line>` : ""}
      ${lines}
      <path class="random-line c-random" d="${path(random)}"></path>
      ${labels}
      <line class="cursor off" x1="0" y1="${PAD.top}" x2="0" y2="${H - PAD.bottom}"></line>
    </svg>`;
  if (state.cursorAll !== null && state.cursorAll !== undefined) drawCursorAll(state.cursorAll);
}

function drawOne(budgets, selected, random, cursor) {
  const ys = [...selected.low, ...selected.high, ...random, 0];
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  const span = hi - lo || 1;
  const { x, y, svg } = axes(lo, span, budgets, axisLabel());
  const colour = colourClass(state.estimator);
  const path = (vals) =>
    vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const band =
    selected.high.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("") +
    selected.low
      .map(
        (v, i) =>
          `L${x(selected.low.length - 1 - i).toFixed(1)},${y(selected.low[selected.low.length - 1 - i]).toFixed(1)}`
      )
      .join("") +
    "Z";

  el("chart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
      ${svg}
      ${lo < 0 && hi > 0 ? `<line class="zero" x1="${PAD.left}" y1="${y(0).toFixed(1)}" x2="${W - PAD.right}" y2="${y(0).toFixed(1)}"></line>` : ""}
      <path class="band ${colour}" d="${band}"></path>
      <path class="random-line c-random" d="${path(random)}"></path>
      <path class="series ${colour}" d="${path(selected.value)}" stroke-width="3"></path>
      <line class="cursor" x1="${x(cursor).toFixed(1)}" y1="${PAD.top}" x2="${x(cursor).toFixed(1)}" y2="${H - PAD.bottom}"></line>
      <circle class="dot ${colour}" cx="${x(cursor).toFixed(1)}" cy="${y(selected.value[cursor]).toFixed(1)}" r="5"></circle>
      <text class="end-label halo ${colour}" x="${W - PAD.right + 8}" y="${y(selected.value[selected.value.length - 1]).toFixed(1)}">${ESTIMATORS[state.estimator]?.title ?? state.estimator}</text>
      <text class="end-label halo c-random" x="${W - PAD.right + 8}" y="${(y(random[random.length - 1]) + 14).toFixed(1)}">Random</text>
    </svg>`;
}

// ---------------------------------------------------------------- the sample-size calculator

function bootPower() {
  for (const id of ["p-base", "p-lift", "p-budget", "p-holdout"]) {
    el(id).addEventListener("input", renderPower);
  }
  renderPower();
}

function renderPower() {
  const base = Number(el("p-base").value) / 100;
  const relative = Number(el("p-lift").value) / 100;
  const budget = Number(el("p-budget").value) / 100;
  const treated = Number(el("p-holdout").value) / 100;

  el("p-base-label").textContent = `${(base * 100).toFixed(0)}%`;
  el("p-lift-label").textContent = `${(relative * 100).toFixed(0)}%`;
  el("p-budget-label").textContent = `${(budget * 100).toFixed(0)}%`;
  el("p-holdout-label").textContent = `${(treated * 100).toFixed(0)}%`;

  const outcomeSd = binaryOutcomeSd(base);
  const averageEffect = relative * base;
  const report = requirementTable({ outcomeSd, averageEffect, budget, treatedShare: treated });

  const rows = report.rows
    .map(
      (r) => `<tr>
        <td>${r.liftRatio.toFixed(2)}x the average</td>
        <td class="num">${r.units.toLocaleString()}</td>
        <td class="num">${r.multipleOfEffectDetection < 10 ? r.multipleOfEffectDetection.toFixed(1) : Math.round(r.multipleOfEffectDetection).toLocaleString()}x</td>
      </tr>`
    )
    .join("");

  el("power-output").innerHTML = `
    <div class="power-floor">
      <span class="figure">${report.toDetectAnEffect.toLocaleString()}</span>
      <span class="caption">people, just to show the intervention does <em>anything</em></span>
    </div>
    <p class="power-lead">To show that <strong>targeting beats random</strong>, it depends
      entirely on how much better your top group responds, which is the thing you are trying to
      find out:</p>
    <div class="table-wrap">
      <table class="power-table">
        <thead><tr><th>If the top ${(budget * 100).toFixed(0)}% responds</th><th class="num">You need</th><th class="num">Times the easy question</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <p class="power-note">A base rate of ${(base * 100).toFixed(0)}% with a
      ${(relative * 100).toFixed(0)}% relative effect means the intervention moves the outcome
      from ${(base * 100).toFixed(1)}% to ${((base + averageEffect) * 100).toFixed(1)}%.</p>`;
}

// ---------------------------------------------------------------- the section tabs

// The pill for the section being read is marked current. A section counts as being read once
// its top has passed the upper third of the window.
function watchTabs() {
  const links = [...document.querySelectorAll(".tabs a[href^='#']")];
  const sections = links.map((a) => document.getElementById(a.hash.slice(1)));
  const mark = () => {
    const line = window.innerHeight / 3;
    let current = 0;
    sections.forEach((s, k) => {
      if (s && s.getBoundingClientRect().top <= line) current = k;
    });
    links.forEach((a, k) => {
      if (k === current) a.setAttribute("aria-current", "true");
      else a.removeAttribute("aria-current");
    });
  };
  window.addEventListener("scroll", mark, { passive: true });
  mark();
}

watchTabs();
boot().catch((error) => {
  const failure = el("failure");
  failure.textContent =
    `The results could not be loaded (${error.message}). The same numbers are in the ` +
    `repository's README, as tables.`;
  failure.hidden = false;
});
