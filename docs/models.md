

## Feature engineering
We built 39 features in six groups: calendar (month, week of year, monsoon flag), lagged weather (rainfall, temperature and humidity from 1-4 weeks earlier), rolling weather (4, 8 and 12 week rainfall, 4 week temperature and humidity), weather combinations (temperature range, temperature x humidity), past cases (lags, rolling averages, trend) and ward information (density, water logging, sanitation, city).

Rainfall over the previous 4 weeks was the strongest predictor (correlation about 0.82 with high-risk weeks), followed by 4 week average humidity. This matches how dengue spreads: mosquitoes breed after rain, so cases rise with a delay.

We compared three feature sets (30 base features, 39 with extras, and the top 15) using yearly time-based cross-validation. Their AUC differed by only about 0.001, so the extra features did not improve accuracy. We kept the full set and note that the top-15 set is a simpler alternative with a similar score.

No feature uses the current week's dengue cases. Case features use earlier weeks only, and we split the data by time (train 2016-2023, test 2024-2025) so the models never see the future.