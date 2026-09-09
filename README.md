# Retail Demand Forecasting & Inventory Optimization

Forecasts daily `Units Sold` per store-product from the [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset), then uses that forecast to flag stockout risk and recommend reorder quantities.

![Pipeline overview](/images/retail_forecasting_pipeline.png)

## Setup notes

- The dataset ships a `Demand Forecast` column. Not using it as a feature or target it's someone else's model output, and training on it would just be learning to imitate that model instead of learning from actual sales.
- Price, discount, promotion, holiday, and weather are treated as known ahead of time for a given day, since retailers set these in advance and weather forecasts exist. Only `Units Sold` itself gets lagged, so no row sees its own future.

## Pipeline

1. **Load & clean** - column names vary across dataset variants, so there's a resolver that maps whatever headers show up to the fields we need. Parses dates, sorts by store/product/date, drops duplicate rows, normalizes yes/no-style columns to 0/1.
2. **Features** - calendar fields (day of week, month, quarter, week of year) plus lags (1/7/14/28 days) and rolling mean/std (7/14/28-day windows) of `Units Sold`, computed on `shift(1)` so nothing sees itself.
3. **Leakage check** - same-day `Inventory Level` correlates with sales, but you can't actually know it before today's sales happen. Ran three variants of a Linear Regression pipeline (same-day inventory, inventory dropped, yesterday's inventory) to confirm the swap doesn't tank performance, then switched to `inventory_lag_1` regardless of the numbers, since same-day isn't obtainable at prediction time in a real deployment.
4. **Model training** - chronological 70/15/15 train/val/test split (no shuffling, it's time series). Compared naive lag-1, 7-day moving average, Linear Regression, and tuned Random Forest / XGBoost / LightGBM (`RandomizedSearchCV` + `TimeSeriesSplit`).
5. **Ablation study** - added feature groups one at a time (calendar → + lag → + rolling → + price/promo → + weather) and tracked validation MAE at each step, to see which groups are actually pulling weight.
6. **Evaluate** - best model by validation MAE gets refit on train+val, scored once on test (that's the number that counts, everything before it is validation). Also ran rolling-origin backtesting across 4 origins to check the score isn't a fluke of one split, plus residual and feature-importance plots.

## Inventory decision layer

7. **7-day forecast** - recursive rollout: predict day 1, feed that prediction back into the lag/rolling features to build day 2's inputs, predict day 2, and so on. Just repeating a single day's prediction 7x would miss day-of-week effects and get the error compounding wrong.
8. **Safety stock** - assumes a fixed lead time and a 95% service level (z ≈ 1.65) applied to historical forecast error, as a stand-in for real demand uncertainty. Neither of these is in the dataset - they're placeholders for what would normally come from actual supplier agreements.
9. **Reorder qty** - flags `stockout_risk` when current inventory falls short of forecasted lead-time demand plus the safety buffer, and computes `recommended_order_qty` to close it. Uses the most recent actual inventory reading, which is a different situation from step 3's leakage - that was about not knowing today's inventory before today's sales; this is the stock on the shelf right now, at decision time, which is observed, not predicted.

## What actually helped

- Engineered features (lags, rolling stats, price/promo/weather) added very little - EDA, the ablation study, and the backtest all pointed the same way. Probably a property of this particular synthetic dataset, not a general claim about retail demand.
- Linear Regression was competitive with the tree models on validation, consistent with there not being much nonlinear structure for them to exploit.

## Limitations

- Single combined time-series split rather than per-series CV.
- Lead time / service level in the reorder math are placeholders, not real supply-chain numbers.
- Store/product IDs are ordinal-encoded, so this won't generalize to unseen stores or products.
- Safety stock assumes normal-distributed forecast error; a more rigorous version would use quantile/probabilistic forecasting instead of a point forecast plus an assumed error distribution.