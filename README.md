# Team vcts | Microburbs Hackathon Task 4: weekly asking rents

We predict the weekly asking rent (AUD) for **65,777 Sydney flats**, once with a car space and once without. Both predictions use the same flat attributes and history; only the parking indicator changes. The information cutoff is **31 July 2026**.

**Prediction file:** [rents_vcts.csv](rents_vcts.csv). Columns: `gnaf_id`, `rent_with_car_space`, `rent_without_car_space`. SHA-256: `ecfc32c8a28850d4b248cd7d581070db4a617b20bbfe78363b7c2390b5ac0cc2`.

**Report:** [report.html](report.html), a self-contained English report with findings, dated evaluation, the final-refit distinction and links to saved evidence.

## Method

The final histogram gradient boosting regressor uses bedrooms, bathrooms, parking, coordinates, suburb, property type and a market-time feature. Rental history adds the last advertised rent for that flat, days since that listing, and the count and median of comparable advertisements from the preceding 365 days. Comparables share a suburb, building address and bedroom count; their median requires at least three advertisements. Same-day and future rents are excluded. All predictions for each validation window use history frozen at the start of that window.

The chosen model uses absolute-error loss, 63 maximum leaves, at least 30 training examples per leaf, learning rate 0.04, 400 iterations and L2 regularization of 1. Transit distances were tested but not used in the final model. At the final cutoff we trained on **12,209** recent advertisements, with **112,457** earlier and recent advertisements available for strictly historical features.

## Local evaluation

| Evaluation | Scored ads | Median absolute error (AUD/week) | Within AUD 100/week | R² |
|---|---:|---:|---:|---:|
| Three development windows, equally weighted | 2,359 / 2,297 / 1,917 | 47.82 | 76.33% | 0.8683 |
| Final local holdout: May–July 2026 | 3,956 of 4,083 | **47.58** | **76.57%** | **0.8430** |

In the March–April development window, the grouped rent-median baseline had MedAE AUD 100.00; the selected HGB had AUD 48.45 on the same 1,917 advertisements. The [saved aligned comparison](reports/hgb_cv/development_comparison.json) compares complete pipelines, not an isolated history-feature ablation. The finalization script validates the saved no-transit configuration before the May–July evaluation. Ads with unobserved parking are excluded from local scoring; predictions are supplied for every requested flat. These are local results, not official hidden-test scores. The parking difference is a model scenario estimate, not a causal parking premium; 1,618 flats have a lower estimate in the with-parking scenario.

**Evaluation versus final refit:** the May–July scores describe a model fitted on July 2025–April 2026 adverts, with history frozen at 1 May. The root prediction CSV comes from a separate refit of the same settings on October 2025–July 2026 adverts. This includes the earlier local-check period, so those scores are not an independent evaluation of the submitted fitted model. No August–September judging labels were used.

## Reproduction

Code: [scripts/finalize_hgb.py](scripts/finalize_hgb.py). The repository contains aggregate validation metrics and the final predictions, not the organiser's input data, private per-ad predictions, API credentials or API responses.

Use Python 3.14.6 and the versions in `requirements.txt`. In a fresh Python environment run `python -m pip install -r requirements.txt`, then `python scripts/finalize_hgb.py --root . --output-dir reproduced`. Keep the output folder separate from the submitted CSV. No API key is needed.

Authorised participants can place their `flats_clean_transit_v2.csv` and `leases_clean_transit_v2.csv` in `data/processed/` to use the same inputs as the submitted run (distance columns are ignored). The script also accepts the team's base cleaned pair or the original `rents-input-flats-city-of-sydney.csv` and `rents-past-leases-city-of-sydney.csv` under `data/raw/`, applying the model-column cleaning to raw inputs. The script must reproduce the saved March–April development result before it proceeds; an input or runtime difference may fail that check. For the predictions actually submitted, use the CSV at the repository root and its checksum above.

Team presentation files use the root filenames `report.html`, `pitch.mp4` and `demo.mp4`.

## Sources

Microburbs supplied the flat attributes and historical asking-rent adverts for the event; its inputs are not redistributed. Address identifiers use Open G-NAF, © Geoscape Australia, licensed by the Commonwealth. The supplied block data contains ABS 2021 Census counts and Microburbs 2026 estimates; block counts are not inputs to the final model.

- [Official rents task](https://www.microburbs.com.au/p/hackathon-task-rents)
- [Participant and submission guide](https://www.microburbs.com.au/p/hackathon-guide)
- [Open G-NAF](https://data.gov.au/data/dataset/geocoded-national-address-file-g-naf)
- [ABS Census](https://www.abs.gov.au/census)
