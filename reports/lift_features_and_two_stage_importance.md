# Lift features and their role in the two-stage model

## Short answer

The lift features summarize how demand changes during promotions or around calendar
events relative to normal demand. The two pooled event features were designed to
match the two-stage decomposition:

- `event_lift_pooled_occurrence` describes how an event changes the chance of a
  positive sale.
- `event_lift_pooled_quantity` describes how an event changes the quantity sold,
  conditional on a positive sale.

This conceptual match does not guarantee high LightGBM feature importance. In the
current data, the pooled features are coarse event-level numbers, are missing on
most rows, have very few distinct values, and overlap with richer calendar and
annual-history features. Consequently, they offer little *additional* split gain to
the trees even though their statistical interpretation is appropriate for the
two-stage model.

## What “lift” means

A lift is a ratio between demand under a special condition and a normal baseline.
A value of 1.0 means “no change,” 1.25 means “25% higher than normal,” and 0.8 means
“20% lower than normal.” The promotion feature is stored after subtracting one, so
its neutral value is 0 rather than 1.

The feature builder creates five columns whose names contain `lift`:

| Feature | Level | Meaning |
| --- | --- | --- |
| `mean_action_lift_in_sourcing_group` | sourcing group and origin | Historical promotion uplift |
| `event_lift_series` | article-store and target date | Prior-year event demand relative to that series' recent demand |
| `event_lift_pooled_occurrence` | event and origin | Event effect on the probability of positive demand |
| `event_lift_pooled_quantity` | event and origin | Event effect on demand size when demand is positive |
| `event_lift_pooled_total` | event and origin | Product of pooled occurrence and quantity lift |

All historical inputs are restricted to dates strictly before the forecast origin.
The features therefore do not use future demand.

## How each feature is calculated

### Historical promotion lift

For each sourcing group, the builder takes all active rows before the origin and
calculates:

```text
mean demand on promoted rows / mean demand on non-promoted rows - 1
```

Inactive rows are excluded, while active rows with zero demand remain in both
means. For example, a value of 0.30 says that promoted rows historically had 30%
higher mean demand in that sourcing group. This number can be used together with
`action_on_forecast_day`: the action flag says whether a promotion is scheduled,
while the lift summarizes the historical strength of promotions in that group.

### Series-specific event lift

`event_lift_series` is calculated for an article-store series when the target lies
within three calendar days of an event. The target is mapped to the corresponding
prior-year event anchor. At that anchor, the builder calculates:

```text
demand on the prior-year event anchor
-------------------------------------
mean demand on the 24 active rows immediately before that anchor
```

The value is missing if the mapped anchor is inactive or if 24 earlier active rows
are not available. This feature is specific to an article-store, so it can express
that the same event affects two products or stores differently. Its weakness is
that its numerator is a single historical event-anchor observation and can therefore
be noisy.

### Pooled occurrence and positive-quantity lift

The pooled calculation first separates demand into the same two pieces as the
two-stage model:

```text
expected demand = probability of positive demand
                x mean quantity when demand is positive
```

For every historical active row in an event window, the builder constructs a normal
non-event baseline for that article and weekday:

- baseline occurrence rate = historical share of non-event rows with demand above
  zero;
- baseline positive mean = historical mean demand among positive non-event rows.

Both event observations and non-event baseline observations enter this calculation
only after their article-store series has at least 24 earlier active rows. This
avoids comparing mature histories with observations that have almost no preceding
demand context.

The article baseline pools stores. It is used after at least 30 baseline
observations are available and positive demand has been observed. Otherwise, the
builder falls back to a broader baseline with the same weekday, sourcing group, and
product category.

Historical event rows are then pooled by exact event name and by the length of the
associated store-closure block. They are not pooled with other events. For each
event cell, the occurrence lift is:

```text
actual number of positive event observations
---------------------------------------------------------
sum of the normal occurrence probabilities for those rows
```

The quantity lift is:

```text
actual quantity summed over positive event observations
----------------------------------------------------------------
sum of the normal positive-demand means for those observations
```

Zero-demand event observations affect occurrence lift but are excluded from the
quantity-lift numerator and denominator. A pooled value is exposed only when the
cell contains at least four distinct historical event-window dates. “Four dates”
means four daily dates in event windows, not necessarily four annual occurrences of
the holiday.

As a simple example, suppose an event cell contains 100 observations. Their normal
occurrence probabilities sum to 40, but 50 observations actually have positive
demand. The occurrence lift is `50 / 40 = 1.25`. If the normal positive means for
those 50 positive observations sum to 150 units and actual positive demand sums to
180 units, quantity lift is `180 / 150 = 1.20`. The implied total lift is
`1.25 x 1.20 = 1.50`.

`event_lift_pooled_total` is exactly the product of the two pooled components. It is
provided to the direct L2 and Tweedie models because those models predict demand in
one step.

## How the two-stage model uses the features

The occurrence booster is a binary classifier trained on `actual > 0`. It receives
`event_lift_pooled_occurrence` but not pooled quantity or pooled total lift.

The positive-quantity booster is a Gamma model trained only on rows where
`actual > 0`. It receives `event_lift_pooled_quantity` but not pooled occurrence or
pooled total lift. Its label is demand normalized by the article-store target mean;
the predicted quantity is converted back to the original scale afterward.

Both boosters may also use `event_lift_series`, calendar context, action features,
recent demand, annual demand, article and store identifiers, and the other shared
features. At prediction time, the final daily forecast is:

```text
clipped occurrence probability x predicted positive quantity
```

The pooled lifts are ordinary numeric inputs to LightGBM. They do not directly
multiply or otherwise force-adjust the prediction. A tree uses them only if a split
on their values improves its training objective.

## Why the pooled features have low feature importance

### 1. They apply to a minority of rows

Across the 72 materialized production origins currently in the feature store,
pooled lift is non-missing on 15.27% of active, training-eligible rows and 16.40% of
positive quantity-stage rows. It is available on about 89% of eligible event-window
rows, but non-event rows are much more common. Global feature importance is
therefore dominated by ordinary trading days.

This also means a feature can be useful specifically near events and still have
small importance when gain is summed over the complete training set.

### 2. The pooled signal is extremely coarse

There are only 22 distinct non-missing pooled occurrence values and 22 distinct
pooled quantity values across 10,106,012 materialized rows. For a given origin and
event, every article-store row receives the same value. The value also stays the
same before, on, and after that event because relative event position and weekday
are not part of the final pooled event key.

The weekday is used to create a fair normal baseline, but it is not retained when
event outcomes are finally pooled. The feature can therefore say, for example,
“Karfreitag historically raises occurrence overall,” but it cannot say that the
effect is different for a particular article, store, or day within the event
window. Most row-to-row prediction variation must come from other features.

### 3. Other features already carry much of the same information

The model also receives `event_name`, `holiday_event_window`,
`days_to_nearest_event`, neighboring closure counts, annual demand features, and
`event_lift_series`. Trees can reconstruct much of the event pattern from these
features. When correlated features are interchangeable, LightGBM often chooses one
for a split and assigns all gain from that split to it. Low gain for one feature can
therefore mean “redundant after the other features,” not “unrelated to demand.”

The series-specific lift is richer than the pooled lift because it varies by
article-store. In the positive-quantity stage it consequently receives noticeably
more gain than pooled quantity lift, despite its noisier single-anchor construction.

### 4. Leakage protection and minimum-history rules reduce availability

Only event dates before each origin are used, article baselines require 30
observations, and an event cell requires four historical event-window dates. These
are sensible safeguards, but they produce missing values early in the history and
for thin cells. LightGBM can route missing values, but it cannot learn detailed
numeric thresholds where few values exist.

### 5. “Designed for the model” means aligned, not guaranteed to be predictive

The feature design correctly mirrors the model's occurrence-times-quantity
structure. That prevents the occurrence booster from receiving a quantity-only
summary and vice versa. It does not ensure that the estimates are precise, granular,
or incrementally informative after all other inputs are present. Feature engineering
provides a hypothesis; model fitting tests whether the hypothesis improves the
objective in this dataset.

## What the current gain importance actually shows

The saved importance is LightGBM's split gain, normalized separately within each
stage and refit block. It measures how much the fitted trees reduced their training
objective through splits on a feature. It is neither a causal effect nor a direct
measure of forecast degradation when a feature is removed.

Mean values across the five current expanding-window refits are:

| Stage and feature | Mean gain share | Maximum gain share | Mean rank out of 47 |
| --- | ---: | ---: | ---: |
| occurrence: pooled occurrence lift | 0.0077% | 0.0124% | 41.6 |
| positive quantity: pooled quantity lift | 0.0210% | 0.0419% | 41.2 |
| occurrence: series event lift | 0.0252% | 0.0277% | 33.8 |
| positive quantity: series event lift | 0.3210% | 0.3918% | 20.8 |
| occurrence: sourcing-group action lift | 0.0530% | 0.0671% | 28.8 |
| positive quantity: sourcing-group action lift | 0.2490% | 0.3455% | 23.6 |

Thus, the result supports a narrow conclusion: the pooled stage-specific features
provide very little additional *global split gain* in the current specification.
It does not establish that event lift is useless on event days.

The cleanest test of practical value would be a repeated-origin ablation: refit the
same two-stage model without the pooled occurrence and quantity features, then
compare both overall metrics and event-window-only metrics. Event-window permutation
importance would provide a complementary diagnostic. Those tests answer a different
question from gain importance: how much out-of-sample forecast quality is lost when
the lift signal is unavailable.

## Implementation and result references

- Feature formulas and thresholds:
  [`src/models/lightgbm/features/builder.py`](../src/models/lightgbm/features/builder.py)
- Stage-specific feature routing and final multiplication:
  [`src/models/lightgbm/two_stage/model.py`](../src/models/lightgbm/two_stage/model.py)
- Gain-importance calculation:
  [`src/models/lightgbm/base.py`](../src/models/lightgbm/base.py)
- Current two-stage importance artifact:
  [`reports/results/feature_importance/artikel_markt_multi7days_lightgbm_two_stage.csv`](results/feature_importance/artikel_markt_multi7days_lightgbm_two_stage.csv)
