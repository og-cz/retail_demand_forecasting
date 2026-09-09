# Retail Demand Forecasting & Inventory Optimization

Forecasts daily `Units Sold` per store-product from the [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset), then uses that forecast to flag stockout risk and recommend reorder quantities.

![Pipeline overview](/images/retail_forecasting_pipeline.png)

## Setup notes

The dataset ships a `Demand Forecast` column, which is left out of both the features and the target it's someone else's model output, and training on it would just teach this model to imitate that one. Price, discount, promotion, holiday, and weather are treated as known ahead of time, since retailers set these in advance. Only `Units Sold` gets lagged, so no row sees its own future.

## Pipeline

Data is loaded through a column resolver (headers vary across dataset versions), then parsed, sorted by store/product/date, deduplicated, and normalized. Features combine calendar fields (day of week, month, quarter) with lags (1/7/14/28 days) and rolling mean/std (7/14/28-day windows) of `Units Sold`, all shifted by one day so nothing sees itself.

A leakage check on `Inventory Level` came next it correlates with same-day sales, but you can't actually know today's inventory before today's sales happen. Three variants were tested (same-day, dropped, lagged), and the switch to `inventory_lag_1` was made regardless of the numbers, since same-day just isn't obtainable at prediction time.

Training used a chronological 70/15/15 split and compared:
- naive lag-1 and 7-day moving average baselines
- Linear Regression
- tuned Random Forest, XGBoost, and LightGBM (`RandomizedSearchCV` + `TimeSeriesSplit`)

An ablation study added feature groups one at a time (calendar → lags → rolling → price/promo → weather) to see which actually moved validation MAE. The best model by validation score was refit on train+val and scored once on test that's the number that counts. Rolling-origin backtesting across four origins confirmed it wasn't a fluke of one split.

## Inventory decision layer

The forecast feeds a small decision layer. A 7-day forecast uses a recursive rollout predict day 1, feed that back into the lag/rolling features, predict day 2, and so on rather than just repeating one day's prediction, which would miss day-of-week effects. Safety stock assumes a fixed lead time and 95% service level (z ≈ 1.65) applied to historical forecast error, standing in for real demand uncertainty since neither figure is in the dataset. From there, `stockout_risk` flags when current inventory falls short of forecasted lead-time demand plus the safety buffer, and `recommended_order_qty` closes that gap using the most recent *actual* inventory reading, which is fine since it's observed, not predicted.

## What actually helped

Engineered features (lags, rolling stats, price/promo/weather) added very little lift EDA, the ablation study, and the backtest all pointed the same way, likely a quirk of this particular synthetic dataset rather than a general fact about retail demand. Linear Regression stayed competitive with the tree models, consistent with not much nonlinear structure to exploit.

## Limitations

- Single combined time-series split rather than per-series CV
- Lead time / service level in the reorder math are placeholders, not real supply-chain numbers
- Store/product IDs are ordinal-encoded, so this won't generalize to unseen stores or products
- Safety stock assumes normal-distributed error; a stricter version would use quantile forecasting instead

## Running it

Needs `numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm joblib kagglehub` (the notebook installs the last three itself if missing). Looks for a local CSV first, falls back to `kagglehub`. Final model saves to `retail_demand_forecast_model.joblib`.