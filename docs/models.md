

## Feature engineering
We built 39 features in six groups: calendar (month, week of year, monsoon flag), lagged weather (rainfall, temperature and humidity from 1-4 weeks earlier), rolling weather (4, 8 and 12 week rainfall, 4 week temperature and humidity), weather combinations (temperature range, temperature x humidity), past cases (lags, rolling averages, trend) and ward information (density, water logging, sanitation, city).

Rainfall over the previous 4 weeks was the strongest predictor: it has the highest correlation with high-risk weeks (0.82) and the highest XGBoost importance (0.31), followed by 4 week average humidity (importance 0.23). This matches how dengue spreads: mosquitoes breed after rain, so cases rise with a delay. Correlation shows a link, not proof of cause.

We compared three feature sets (30 base features, 39 with extras, and the top 15) using yearly time-based cross-validation. All three reached the same AUC (0.9845) and F1 scores between 0.895 and 0.898, so the extra features did not improve accuracy. We kept the full set and note that the top-15 set is a simpler alternative with the same score.

No feature uses the current week's dengue cases. Case features use earlier weeks only, and we split the data by time (train 2016-2023, test 2024-2025) so the models never see the future.

## Data cleaning
Before modelling, `src/clean_data.py` checks the dataset (10,440 weekly records, 20 wards) for unparseable dates, missing values, duplicate ward-week rows, gaps in the weekly timeline, impossible values (negative rainfall or cases, humidity outside 0-100, minimum temperature above average temperature, scores outside 0-1) and wards that change city or static values over time. It found no problems. It also flagged statistical outliers in rainfall (516 weeks) and dengue cases (864 weeks). These fall in the monsoon (May-Sep) and dengue season (Jun-Oct), so they are real extreme weeks, not errors, and we kept them. The cleaned file is saved as `data/clean_dataset.csv` and the full report is in `docs/data_cleaning_report.md`.