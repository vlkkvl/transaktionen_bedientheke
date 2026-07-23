# Feature Source Assessment: Write-Offs and Goods Receipts

Yes, both sources can be used, but not equally.

Goods receipts are the more interesting new signal, but coverage is uneven. Write-offs are usable only with careful filtering, because most "Abschriften" rows are not spoilage but action, discount, or process reasons.

The profiling below compares both raw sources against the current modelling population: FCM/Pseudo, `WGR_ID IN (890, 900)`, prepared daily rows from `transactions_dst_daily_min_demand_no_outliers`.

## Current Modelling Data

| Source | Rows / Series | Date range |
|---|---:|---|
| current model population | 19,934,887 rows / 25,169 series | 2023-07-22 to 2026-07-21 |

## Write-Offs

| Metric | Value |
|---|---:|
| raw rows | 46,012,963 |
| raw series | 555,742 |
| date range | 2010-01-01 to 2026-06-25 |
| negative quantity rate | 28.4% |
| prepared series with any write-off history | 21,548 / 25,169 = 85.6% |

## Goods Receipts

| Metric | Value |
|---|---:|
| raw rows | 13,335,004 |
| raw series | 243,185 |
| date range | 2023-06-27 to 2026-06-24 |
| negative quantity rate | 23.5% |
| prepared series with any receipt history | 12,420 / 25,169 = 49.3% |

## Coverage By WGR

| WGR | Prepared series | With write-offs | With receipts |
|---|---:|---:|---:|
| 890 | 22,336 | 18,947 = 84.8% | 9,634 = 43.1% |
| 900 | 2,833 | 2,601 = 91.8% | 2,786 = 98.3% |

Receipts are almost complete for `900`, but only partial for `890`, which is the problematic group. That does not make them useless, but missingness indicators are required.

## Write-Off Interpretation

Do not add total write-offs blindly.

The most frequent write-off reasons are:

| Reason | Rows | Meaning |
|---|---:|---|
| `E` | 21.9M | Zentrale Aktionen |
| `Q` | 12.4M | Bruch/Verderb |
| `8` | 4.8M | Mitarbeiterkarte |
| `L` | 3.3M | Dezentrale Preisreduzierungen |
| `F` | 1.2M | Dezentrale Aktionen |

This means "Abschriften" are not a clean spoilage signal. They mix promotions, discounts, employee cards, actions, and actual spoilage.

Use them like this:

- `spoilage_qty_last_28d`: only `ABSCHRIFT_ART = 'Q'`
- `spoilage_days_last_28d`
- `days_since_last_spoilage`
- `discount_writeoff_qty_last_28d`: maybe `L`, maybe `E/F` depending on business meaning
- `has_any_writeoff_history`
- `writeoff_missing_flag`

Avoid using a single `total_writeoff_qty_last_28d` unless the reason groups are separated. It will probably duplicate action effects and add noise.

## Goods Receipt Interpretation

Goods receipts are worth testing as an ablation.

They have direct operational meaning: article, market, date, quantity, unit, delivery path. But they are not stock. Also 23.5% of receipt rows have negative quantities, so corrections and returns are common.

Use these features:

- `receipt_qty_pos_last_7d`
- `receipt_qty_pos_last_14d`
- `receipt_qty_pos_last_28d`
- `receipt_days_last_28d`
- `days_since_last_receipt`
- `receipt_qty_net_last_28d`
- `receipt_correction_qty_last_28d`
- `has_receipt_history`
- `receipt_missing_flag`
- `BEZUGSWEG` aggregates, if stable enough

For negative quantities, do not just sum raw `WE_MENGE_VKE`. Keep both:

- positive receipts as supply signal
- negative/correction quantities as process-noise signal

## Recommended Experiment

Try them as one clean ablation:

1. Current rolling-origin LightGBM.
2. `+ goods receipt features`.
3. `+ spoilage-only write-off features`.
4. `+ goods receipts + spoilage write-offs`.

Do not add all write-off reasons as one lump.

Expected outcome:

- Goods receipts may help `WGR 900` more than `890`, because receipt coverage for `900` is much better.
- Write-offs may help only if `Bruch/Verderb` is isolated.
- If `FCM 890` still fails after receipt features, the issue is probably not missing supply context alone. Then segmentation or calibration is more promising than more feature engineering.
