# Atlas Charts for the sorter (manual, ~10 minutes in the Atlas UI)

Charts can only be created in the Atlas web UI. Do this once `MONGODB_URI` works and at least one attempt
is in Atlas (run `python scripts/atlas_setup.py`, then one pick).

Every doc the Store writes to Atlas has `at` (a Date) next to `ts` (epoch seconds); use `at` for time axes.
Attempt docs: `success` (bool), `label`, `mode` (`trial | demo | teach | reset`), `duration_s`, `fail_phase`.

## 1. Data source

1. Atlas → your project → **Charts** (left nav). First time: **Activate Charts**.
2. **Data Sources** → **Add Data Source** → pick the hackathon cluster → database `sorter` → tick `attempts` → **Finish**.

## 2. Dashboard and a 0/1 field

1. **Dashboards** → **Add Dashboard** → name `Sorter` → **Create**.
2. **Add Chart** → data source `sorter.attempts`.
3. In the field list click **Add Field** → **Calculated** → name `ok`, value `{ $cond: ["$success", 1, 0] }` → **Save**.
   (Mean of `ok` is the success rate.)
4. Filter for every chart below: **Filter** tab → drag `mode` → **Not** `reset` (reset passes are not picks).

## 3. Charts

| Chart | Type | Encode | Settings |
|---|---|---|---|
| **Success rate over time** | Line | X: `at`, binning **Minute**, bin size 10 · Y: `ok`, aggregate **Mean** | Y axis 0–1, format percent |
| **Attempts over time** | Column | X: `at`, binning **Minute**, bin size 10 · Y: `_id`, **Count** | put next to the line chart |
| **Success by object** | Bar | Y (category): `label` · X: `ok`, **Mean** | sort by value, descending |
| **Failure phases** | Donut | Label: `fail_phase` · Arc: `_id`, **Count** | extra filter `success` = false |
| **Total attempts** | Number | `_id`, **Count** | |

Save each chart into the `Sorter` dashboard. Set the dashboard **Auto-refresh** to 1 minute.

## 4. Embed into our dashboard

1. On each chart: **…** → **Embed** → **Unauthenticated** → enable **Unauthenticated access** for the chart
   (this makes that chart viewable by anyone with the link; it shows only aggregated attempt data).
2. Copy the **iframe `src`** URL.
3. Start the dashboard with the URLs, comma-separated:
   `ATLAS_CHART_URLS="https://charts.mongodb.com/...&chartId=...,https://charts.mongodb.com/...&chartId=..." python dashboard/app.py`
   The "Atlas Charts" section appears under the attempts chart.
