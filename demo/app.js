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

// One hue per family. The five LightGBM meta-learners are shades of a single blue so they read
// as variations on one idea; the neural network and the two baselines each get their own hue,
// because those are the comparisons the project is about. Random is red and dashed: it is the
// line everything has to clear, and it must be findable by someone who cannot separate red from
// green, which is why it is also the only dashed line and is labelled directly on the chart.
const COLOURS = {
  "s-learner": "#2563eb",
  "t-learner": "#3b82f6",
  "x-learner": "#60a5fa",
  "dr-learner": "#1d4ed8",
  "r-learner": "#93c5fd",
  dragonnet: "#7c3aed",
  "outcome-ranking": "#0f5c6e",
  random: "#dc2626",
};

// Only the ones that fail on a dark background. The page's teal is chosen to sit on paper and
// vanishes on near-black, and it is risk ranking's line, which is the one a reader is here to
// find. The deepest blue and the purple go up a step with it so the family still reads as a
// family.
const COLOURS_DARK = {
  "dr-learner": "#4f7dff",
  dragonnet: "#a78bfa",
  "outcome-ranking": "#2dd4bf",
  random: "#f2645a",
};

const darkMode = window.matchMedia?.("(prefers-color-scheme: dark)");

function colourOf(name) {
  return (darkMode?.matches ? COLOURS_DARK[name] : null) ?? COLOURS[name] ?? "#888";
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
  state.index = await fetch("data/index.json").then((r) => r.json());
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
  darkMode?.addEventListener?.("change", () => {
    renderAll();
    render();
  });

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
  state.payload = await fetch(`data/${key}.json`).then((r) => r.json());
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
    colour: colourOf(name),
  }));
  const random = p.random.dr;

  drawMulti("chart-all", p.budgets, series, random);

  // The colour travels as a data attribute and is applied below rather than written into a
  // style attribute here. Under the page's own content security policy a style attribute in
  // markup is blocked, which left every swatch blank on the live site while looking correct
  // on a local server that sends no policy. Setting the property from script is allowed.
  el("legend-all").innerHTML =
    series
      .map(
        (s) =>
          `<button class="legend-item ${state.highlight === s.name ? "on" : ""}" data-name="${s.name}">
             <span class="swatch" data-colour="${s.colour}"></span>${ESTIMATORS[s.name]?.title ?? s.name}
           </button>`
      )
      .join("") +
    `<span class="legend-item static"><span class="swatch dashed" data-colour="${colourOf("random")}"></span>Random targeting</span>`;
  for (const swatch of el("legend-all").querySelectorAll(".swatch[data-colour]")) {
    swatch.style.background = swatch.dataset.colour;
  }
  for (const button of el("legend-all").querySelectorAll("button")) {
    button.addEventListener("click", () => {
      state.highlight = state.highlight === button.dataset.name ? "" : button.dataset.name;
      el("highlight").value = state.highlight;
      renderAll();
    });
  }

  el("caption-all").textContent =
    `Every method on ${meta().title ?? state.dataset}, measured in ${unitLabel()}. ` +
    `Higher is better. The red dashed line is random targeting.`;

  renderVerdict(series, random);
  renderHero(series, random);
}

// The sentence a visitor should leave with, computed rather than written, so it cannot go stale
// when a number changes.
function renderVerdict(series, random) {
  const p = state.payload;
  const i = Math.max(0, Math.round(0.1 * (p.budgets.length - 1)) - 1);
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
  if (clearlyWorse) {
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
      `${scaled(best.values[i])}. The sophisticated machinery earns nothing measurable here, ` +
      `which is worth knowing before you build it.`;
  }
  el("verdict").innerHTML = line;
}

function renderHero(series, random) {
  const i = Math.max(0, Math.round(0.1 * (state.payload.budgets.length - 1)) - 1);
  const best = series.reduce((a, b) => (b.values[i] > a.values[i] ? b : a));
  const risk = series.find((s) => s.name === "outcome-ranking");
  el("hero-best").textContent = scaled(best.values[i]);
  el("hero-risk").textContent = risk ? scaled(risk.values[i]) : "—";
  el("hero-random").textContent = scaled(random[i]);
  el("hero-best").className = `point-figure ${best.values[i] > random[i] ? "up" : "down"}`;
  el("hero-risk").className = `point-figure ${
    risk && risk.values[i] < random[i] ? "down" : "up"
  }`;

  // A figure with no noun on it is not a measurement. All three are in the same unit, and the
  // unit changes with the dataset, so it is written under each one rather than assumed.
  for (const node of document.querySelectorAll(".point-unit")) node.textContent = shortUnitLabel();
  el("hero-context").innerHTML =
    `These three figures are for <strong>${meta().title ?? state.dataset}</strong>, the dataset ` +
    `selected below.`;
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
    `The shaded band is the 95% confidence range. The vertical line is your selected budget.`;

  el("provenance").textContent =
    `${p.dataset}, one split at seed ${p.seed} of the five the results table averages,` +
    ` ${p.n_test.toLocaleString()} held-out rows, ${p.n_treated.toLocaleString()} of them treated in the data.` +
    ` Bands are ${Math.round(p.level * 100)}% percentile intervals from ${p.n_resamples} bootstrap resamples` +
    ` of this split, so they cover sampling within it and not variation between splits.`;

  drawOne(p.budgets, selected, random, i);
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
  const xticks = [0, 0.25, 0.5, 0.75, 1]
    .map((f) => {
      const i = Math.round(f * (budgets.length - 1));
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
        `<path class="series ${dim && state.highlight !== s.name && s.name !== "outcome-ranking" ? "faded" : ""}" d="${path(s.values)}"
               stroke="${s.colour}" stroke-width="${state.highlight === s.name ? 3.5 : 2}"></path>`
    )
    .join("");

  // Direct labels at the right edge so the chart is readable without cross-referencing a
  // legend, and so it survives being read by someone who cannot distinguish the blues.
  // Only the lines a reader needs to find by name: random, which is the floor, risk ranking,
  // which is the comparison, and whatever is highlighted. Labelling all eight stacks them on
  // top of each other wherever the curves converge, which is most of the right-hand side.
  const named = [
    { name: "random", values: random, colour: colourOf("random") },
    ...series.filter((s) => s.name === "outcome-ranking" || s.name === state.highlight),
  ];
  let previous = -Infinity;
  const labels = named
    .map((s) => ({ s, yy: y(s.values[s.values.length - 1]) }))
    .sort((a, b) => a.yy - b.yy)
    .map(({ s, yy }) => {
      const place = Math.max(yy, previous + 14);
      previous = place;
      return `<text class="end-label" x="${W - PAD.right + 8}" y="${place.toFixed(1)}"
                    fill="${s.colour}">${ESTIMATORS[s.name]?.title ?? s.name}</text>`;
    })
    .join("");

  el(target).innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
      ${svg}
      ${lo < 0 && hi > 0 ? `<line class="zero" x1="${PAD.left}" y1="${y(0).toFixed(1)}" x2="${W - PAD.right}" y2="${y(0).toFixed(1)}"></line>` : ""}
      ${lines}
      <path class="random-line" d="${path(random)}" stroke="${colourOf("random")}"></path>
      ${labels}
    </svg>`;
}

function drawOne(budgets, selected, random, cursor) {
  const ys = [...selected.low, ...selected.high, ...random, 0];
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  const span = hi - lo || 1;
  const { x, y, svg } = axes(lo, span, budgets, axisLabel());
  const colour = colourOf(state.estimator);
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
      <path class="band" d="${band}" fill="${colour}"></path>
      <path class="random-line" d="${path(random)}" stroke="${colourOf("random")}"></path>
      <path class="series" d="${path(selected.value)}" stroke="${colour}" stroke-width="3"></path>
      <line class="cursor" x1="${x(cursor).toFixed(1)}" y1="${PAD.top}" x2="${x(cursor).toFixed(1)}" y2="${H - PAD.bottom}"></line>
      <circle class="dot" cx="${x(cursor).toFixed(1)}" cy="${y(selected.value[cursor]).toFixed(1)}" r="5" fill="${colour}"></circle>
      <text class="end-label" x="${W - PAD.right + 8}" y="${y(selected.value[selected.value.length - 1]).toFixed(1)}" fill="${colour}">${ESTIMATORS[state.estimator]?.title ?? state.estimator}</text>
      <text class="end-label" x="${W - PAD.right + 8}" y="${(y(random[random.length - 1]) + 14).toFixed(1)}" fill="${colourOf("random")}">Random</text>
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

boot();
