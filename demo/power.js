// The sample-size arithmetic, in the browser.
//
// This is a second implementation of `src/itx/metrics/power.py`, which is a thing worth being
// nervous about: two copies of a formula drift, and the drifting one is always the copy nobody
// runs. So `tests/test_power_js.py` feeds a grid of inputs through both this file and the Python
// and fails if any answer differs by more than a rounding step. If you change one, change both,
// and the test will tell you if you did not.
//
// It is arithmetic rather than modelling, which is why it is allowed to run here at all: the
// page's rule is that no model is fitted in your browser, and a closed-form sample size is not
// a model. Every number that came out of a fitted model on this page was precomputed.

"use strict";

const POWER_DEFAULTS = { alpha: 0.05, power: 0.8 };
const LIFT_RATIOS = [1.25, 1.5, 2.0, 3.0];

// Inverse normal CDF: Wichura's AS241 (PPND16), accurate to about 1e-16 across the whole
// range. An earlier version here used a cheaper rational approximation with one Halley
// refinement, which was good to 1e-8, and that turned out not to be good enough: the parity
// test against the Python found the two implementations disagreeing by 217 people out of 621
// million, because a sample size is rounded up and an error that small can still land on the
// wrong side of an integer. Precision was cheaper than a caveat.
function normalQuantile(p) {
  if (!(p > 0 && p < 1)) throw new RangeError(`p must be in (0, 1), got ${p}`);
  const q = p - 0.5;
  if (Math.abs(q) <= 0.425) {
    const r = 0.180625 - q * q;
    return (
      (q *
        (((((((2.5090809287301226727e3 * r + 3.3430575583588128105e4) * r +
          6.7265770927008700853e4) * r + 4.5921953931549871457e4) * r +
          1.3731693765509461125e4) * r + 1.9715909503065514427e3) * r +
          1.3314166789178437745e2) * r + 3.3871328727963666080)) /
      (((((((5.2264952788528545610e3 * r + 2.8729085735721942674e4) * r +
        3.9307895800092710610e4) * r + 2.1213794301586595867e4) * r +
        5.3941960214247511077e3) * r + 6.8718700749205790830e2) * r +
        4.2313330701600911252e1) * r + 1)
    );
  }
  let r = q < 0 ? p : 1 - p;
  r = Math.sqrt(-Math.log(r));
  let value;
  if (r <= 5) {
    r -= 1.6;
    value =
      (((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r +
        2.41780725177450611770e-1) * r + 1.27045825245236838258) * r +
        3.64784832476320460504) * r + 5.76949722146069140550) * r +
        4.63033784615654529590) * r + 1.42343711074968357734) /
      (((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r +
        1.51986665636164571966e-2) * r + 1.48103976427480074590e-1) * r +
        6.89767334985100004550e-1) * r + 1.67638483018380384940) * r +
        2.05319162663775882187) * r + 1);
  } else {
    r -= 5;
    value =
      (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r +
        1.24266094738807843860e-3) * r + 2.65321895265761230930e-2) * r +
        2.96560571828504891230e-1) * r + 1.78482653991729133580) * r +
        5.46378491116411436990) * r + 6.65790464350110377720) /
      (((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r +
        1.84631831751005468180e-5) * r + 7.86869131145613259100e-4) * r +
        1.48753612908506148525e-2) * r + 1.36929880922735805310e-1) * r +
        5.99832206555887937690e-1) * r + 1);
  }
  return q < 0 ? -value : value;
}

function binaryOutcomeSd(baseRate) {
  if (!(baseRate > 0 && baseRate < 1)) {
    throw new RangeError(`base rate must be in (0, 1), got ${baseRate}`);
  }
  return Math.sqrt(baseRate * (1 - baseRate));
}

function multiplier(alpha, power) {
  return (normalQuantile(1 - alpha / 2) + normalQuantile(power)) ** 2;
}

function armFactor(treatedShare) {
  if (!(treatedShare > 0 && treatedShare < 1)) {
    throw new RangeError(`treated share must be in (0, 1), got ${treatedShare}`);
  }
  return 1 / treatedShare + 1 / (1 - treatedShare);
}

// Units needed to tell the average effect from zero. The floor, and the question most pilots
// are sized for.
function unitsToDetectAnEffect({ outcomeSd, averageEffect, treatedShare = 0.5,
                                 alpha = POWER_DEFAULTS.alpha, power = POWER_DEFAULTS.power }) {
  if (!(outcomeSd > 0)) throw new RangeError(`outcome_sd must be positive, got ${outcomeSd}`);
  if (averageEffect === 0) throw new RangeError("average_effect must be non-zero");
  return Math.ceil(
    (multiplier(alpha, power) * outcomeSd ** 2 * armFactor(treatedShare)) / averageEffect ** 2
  );
}

// Units needed to show that targeting the top `budget` share beats random targeting. The null
// is that the targeted group responds exactly like the population, which is what random
// targeting delivers.
function unitsToRank({ outcomeSd, averageEffect, liftRatio, budget, treatedShare = 0.5,
                       alpha = POWER_DEFAULTS.alpha, power = POWER_DEFAULTS.power }) {
  if (!(budget > 0 && budget <= 1)) throw new RangeError(`budget must be in (0, 1], got ${budget}`);
  if (!(liftRatio > 1)) throw new RangeError(`lift_ratio must be above 1, got ${liftRatio}`);
  if (!(outcomeSd > 0)) throw new RangeError(`outcome_sd must be positive, got ${outcomeSd}`);
  const gap = (liftRatio - 1) * averageEffect;
  return Math.ceil(
    (multiplier(alpha, power) * outcomeSd ** 2 * armFactor(treatedShare)) / (budget * gap ** 2)
  );
}

// The smallest lift a study of a given size could have detected: the inverse of unitsToRank,
// and the more useful direction once the data already exists.
function detectableLift({ nUnits, outcomeSd, averageEffect, budget, treatedShare = 0.5,
                          alpha = POWER_DEFAULTS.alpha, power = POWER_DEFAULTS.power }) {
  if (!(nUnits > 0)) throw new RangeError(`n_units must be positive, got ${nUnits}`);
  const gap = Math.sqrt(
    (multiplier(alpha, power) * outcomeSd ** 2 * armFactor(treatedShare)) / (budget * nUnits)
  );
  return 1 + gap / Math.abs(averageEffect);
}

function requirementTable({ outcomeSd, averageEffect, budget, treatedShare = 0.5,
                            liftRatios = LIFT_RATIOS, alpha = POWER_DEFAULTS.alpha,
                            power = POWER_DEFAULTS.power }) {
  const floor = unitsToDetectAnEffect({ outcomeSd, averageEffect, treatedShare, alpha, power });
  return {
    toDetectAnEffect: floor,
    rows: liftRatios.map((liftRatio) => {
      const units = unitsToRank({ outcomeSd, averageEffect, liftRatio, budget, treatedShare,
                                  alpha, power });
      return { liftRatio, units, multipleOfEffectDetection: units / floor };
    }),
  };
}

// Exported for the parity test, which runs this file under Node.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { normalQuantile, binaryOutcomeSd, unitsToDetectAnEffect, unitsToRank,
                     detectableLift, requirementTable, LIFT_RATIOS };
}
