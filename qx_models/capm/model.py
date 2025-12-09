import pandas as pd
from qx.engine.base_model import BaseModel

class CAPMModel(BaseModel):
    def run_impl(self, inputs, params, **kwargs) -> pd.DataFrame:
        prices = inputs["equity_prices"].sort_values(["symbol", "date"]).copy()
        rfr = inputs["risk_free"].groupby("date")["rate"].mean().reset_index().rename(columns={"rate": "rf"})
        prices["ret"] = prices.groupby("symbol")["close"].pct_change()
        df = prices.merge(rfr, on="date", how="left").fillna({"rf": 0.0})
        beta_method = params["beta_method"]
        if beta_method == "simple":
            df["market_premium"] = df["ret"].rolling(252, min_periods=10).mean() - df["rf"]
        elif beta_method == "rolling_ols":
            # Placeholder; implement rolling regression vs market index returns
            df["market_premium"] = df["ret"].rolling(252, min_periods=10).mean() - df["rf"]
        else:
            df["market_premium"] = df["ret"].ewm(span=252, adjust=False).mean() - df["rf"]

        latest = df.groupby("symbol").tail(1)
        latest["predicted_return"] = latest["rf"] + 1.0 * latest["market_premium"]
        latest["predicted_price"] = pd.NA
        latest["confidence"] = 0.5
        latest["horizon_d"] = params["horizon_d"]
        return latest[["symbol", "horizon_d", "predicted_return", "predicted_price", "confidence"]]
