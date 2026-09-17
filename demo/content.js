// Plain-language copy for the page: what each dataset is, what each ranking method does, and
// what the numbers are counted in.
//
// This lives in its own file, and it is prose rather than data, on purpose. A dropdown reading
// "acic" tells a visitor nothing, and a page whose whole argument is that results should be
// checkable by a stranger cannot then assume the stranger knows what a DR-learner is. None of
// this is generated: it is written by hand and reviewed with the README, which is why it is
// not in the JSON that `itx demo build` writes.

"use strict";

// Every dataset on the page. `units` is the single most important field here: without it
// "+0.214 outcome bought per head" is a number with no noun attached.
const DATASETS = {
  hillstrom: {
    title: "Hillstrom email campaign",
    kind: "Real randomised experiment",
    what:
      "64,000 customers of an online retailer who had bought something in the previous year. " +
      "Two thirds were sent a marketing email at random and a third were sent nothing. The " +
      "page uses the men's-email arm against the control.",
    outcome: "Whether the customer visited the website in the following two weeks.",
    units: "extra site visits per 1,000 people in the population",
    scale: 1000,
    decimals: 1,
    why:
      "The everyday case: a real campaign, properly randomised, with a modest effect. It is " +
      "the dataset where the uplift models most clearly earn their keep.",
    honest:
      "Randomised by the retailer, so the comparison is clean. The outcome is a site visit " +
      "rather than a purchase, which is the easier thing to move.",
  },
  criteo: {
    title: "Criteo advertising",
    kind: "Real randomised experiment",
    what:
      "13.98 million advertising impressions from Criteo, an ad company, with 85% of people " +
      "shown the ad and 15% held back at random. The page uses the committed 10% subsample.",
    outcome: "Whether the person converted, meaning they took the action the advertiser wanted.",
    units: "extra conversions per 1,000 people in the population",
    scale: 1000,
    decimals: 2,
    why:
      "The dataset that refuted this project's own premise. Ranking people by risk was " +
      "supposed to be the wrong thing to do, and here it matches every uplift model.",
    honest:
      "Randomised, and very large, so the estimates are tight. Conversions are rare, which " +
      "is why the numbers look small until you scale them per thousand.",
  },
  lenta: {
    title: "Lenta grocery SMS",
    kind: "Real randomised experiment",
    what:
      "687,029 supermarket loyalty customers, randomly assigned to receive a marketing SMS " +
      "or not.",
    outcome: "Whether the customer made a purchase in the following period.",
    units: "extra purchases per 1,000 people in the population",
    scale: 1000,
    decimals: 2,
    why:
      "The honest failure. Nothing here separates from random targeting, including every " +
      "uplift model. The effect is real but far too small to rank people on at this size.",
    honest:
      "This dataset cannot answer the targeting question and the page says so rather than " +
      "hiding it. It is the reason the sample-size calculator further down exists.",
  },
  ihdp: {
    title: "IHDP infant health",
    kind: "Simulated outcomes on real covariates",
    what:
      "747 infants from a real health study, with the outcome simulated so that the true " +
      "effect of the intervention on every single child is known.",
    outcome: "A simulated cognitive test score.",
    units: "simulated test-score points per head",
    scale: 1,
    decimals: 3,
    why:
      "Because the truth is written down, this is one of only two datasets here where you " +
      "can check whether a method recovered the right answer rather than merely a plausible one.",
    honest:
      "The outcome is simulated, so the units are arbitrary and mean nothing in the world. " +
      "Only 747 people, which is far too few to rank on, and the page shows that too.",
  },
  acic: {
    title: "ACIC 2016 benchmark",
    kind: "Simulated outcomes on real covariates",
    what:
      "4,802 people from a real survey, with treatment and outcome simulated. Unlike the " +
      "others, who gets treated here depends on their characteristics, on purpose.",
    outcome: "A simulated continuous outcome.",
    units: "simulated outcome units per head",
    scale: 1,
    decimals: 3,
    why:
      "The opposite of Criteo, and the reason this project has a headline at all. Here " +
      "ranking by risk is worse than doing nothing: it destroys value, while looking fine.",
    honest:
      "Simulated, and deliberately confounded, so it is a test of the methods rather than a " +
      "finding about the world. That confounding is what makes it a hard case.",
  },
  "ieee-fraud": {
    title: "Card fraud review",
    kind: "Real transactions, simulated intervention",
    what:
      "590,540 real card transactions from the IEEE-CIS competition. The intervention, an " +
      "analyst manually reviewing a transaction, is simulated on top of them, because no " +
      "public dataset records who was reviewed and what it changed.",
    outcome: "Dollars of fraud loss prevented by the review.",
    units: "dollars per transaction",
    scale: 1,
    decimals: 2,
    currency: true,
    why:
      "The case built to make risk ranking lose, where it won anyway. The dollars make it " +
      "the easiest result here to read, and the one whose conclusion is least comfortable.",
    honest:
      "The review effect is written by the simulation, not measured from the world, and this " +
      "page says so wherever the dataset appears. The transactions and the fraud labels are real.",
  },
};

// The rankings. `family` drives the colour: one hue for the five LightGBM meta-learners so
// they read as variations on a theme, and separate ones for the neural network and the two
// baselines, which is the comparison the whole project is about.
const ESTIMATORS = {
  "s-learner": {
    family: "lightgbm",
    title: "S-learner",
    short: "One model, treatment as a feature",
    what:
      "Fits a single model to everybody with 'did they get the treatment' as one more input " +
      "column, then asks it the same question twice: once as if treated, once as if not. The " +
      "difference is the predicted effect.",
    strength: "Simplest thing that can work, and uses all the data at once.",
    weakness:
      "A tree can ignore the treatment column if other features predict the outcome better, " +
      "and then every predicted effect collapses toward zero.",
  },
  "t-learner": {
    family: "lightgbm",
    title: "T-learner",
    short: "Two models, one per group",
    what:
      "Fits one model to the treated people and a separate model to the untreated, then " +
      "subtracts one prediction from the other.",
    strength: "The treatment can never be ignored, because the two models are separate.",
    weakness:
      "Each model sees only half the data, and two separately noisy predictions subtracted " +
      "from each other are noisier still.",
  },
  "x-learner": {
    family: "lightgbm",
    title: "X-learner",
    short: "Two models, then a repair step",
    what:
      "Starts like the T-learner, then uses each group's model to impute what would have " +
      "happened to the other group, and fits a second stage to those imputed effects.",
    strength: "Designed for the common case where one group is much larger than the other.",
    weakness: "More moving parts, so more places for an error to enter.",
  },
  "dr-learner": {
    family: "lightgbm",
    title: "DR-learner",
    short: "Doubly robust two-stage",
    what:
      "Builds a corrected outcome for every person that stays honest if either the outcome " +
      "model or the model of who got treated is right, then fits the effect to that.",
    strength:
      "Two chances to be right instead of one, which matters when treatment was not random.",
    weakness: "Needs the most nuisance machinery, and is the slowest to fit here.",
  },
  "r-learner": {
    family: "lightgbm",
    title: "R-learner",
    short: "Residual-on-residual",
    what:
      "Strips the predictable part out of both the outcome and the treatment, then looks for " +
      "the effect in what is left over.",
    strength: "Elegant, and isolates the effect from everything the features already explain.",
    weakness: "Its accuracy depends on the first-stage models being good.",
  },
  dragonnet: {
    family: "neural",
    title: "Dragonnet",
    short: "Neural network",
    what:
      "A small neural network with three heads sharing one representation: one for the " +
      "untreated outcome, one for the treated outcome, and one for the chance of being " +
      "treated, so the representation keeps what matters for the effect.",
    strength: "Learns the shared structure rather than treating the two groups separately.",
    weakness:
      "The only method here that cannot take missing values natively, and the only one where " +
      "the same data has produced different answers in different runs. That was a defect in " +
      "how one dataset was encoded, and it is written up in the repository.",
  },
  "outcome-ranking": {
    family: "baseline",
    title: "Risk ranking",
    short: "What most organisations actually do",
    what:
      "No uplift modelling at all. Predict who is most likely to have the outcome anyway and " +
      "treat them, from highest risk down. This is the baseline everything else has to beat.",
    strength:
      "Cheap, familiar, needs no experiment, and on some datasets here it is as good as " +
      "anything else.",
    weakness:
      "It finds the people most likely to have the outcome, not the people your action would " +
      "change. On ACIC that is worse than spending the budget at random.",
  },
  random: {
    family: "random",
    title: "Random targeting",
    short: "The floor",
    what:
      "Pick people out of a hat. This is what a budget buys with no targeting at all, and " +
      "nothing that fails to beat it is worth building.",
    strength: "Free, unbiased, and impossible to overfit.",
    weakness: "Uses none of the information you have.",
  },
};

// Order for the charts and the dropdown: the five LightGBM learners, the network, then the two
// baselines last, because they are the comparison rather than the contenders.
const ESTIMATOR_ORDER = [
  "s-learner",
  "t-learner",
  "x-learner",
  "dr-learner",
  "r-learner",
  "dragonnet",
  "outcome-ranking",
  "random",
];

const GLOSSARY = {
  "policy value":
    "How much extra outcome a budget buys, per person in the whole population, compared with " +
    "treating nobody. Per population rather than per person treated, so budgets of different " +
    "sizes can be compared on the same axis.",
  "doubly robust":
    "A way of estimating that outcome which stays honest if either the model of the outcome " +
    "or the model of who got treated is right. It does not need both.",
  "confidence interval":
    "The range the number would likely fall in if the experiment were repeated. A wide band " +
    "means the data cannot pin the answer down, and a result whose band crosses zero has not " +
    "been shown to do anything at all.",
  uplift:
    "The change your action causes for a specific person: what happens if you act, minus what " +
    "would have happened if you did not. It is never observed for anybody, because only one " +
    "of the two ever happens, which is what makes this hard.",
  "risk ranking":
    "Sorting people by how likely the outcome is and treating the top of the list. It answers " +
    "'who is most likely to churn', not 'whose mind would we change'.",
  "held-out rows":
    "People the models never saw while learning. Every number here is measured on them, which " +
    "is the only way to find out whether a model learned something real.",
};
