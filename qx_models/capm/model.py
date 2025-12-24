import numpy as np
import pandas as pd

from qx.engine.base_model import BaseModel


class CAPMModel(BaseModel):
    """
    CAPM Expected Return Model.

    Calculates expected returns using CAPM formula:
        E(R_i) = R_f + β_i * (E(R_m) - R_f)

    Where:
        - E(R_i) = Expected return for asset i
        - R_f = Risk-free rate
        - β_i = Market beta for asset i (calculated via rolling regression)
        - E(R_m) = Expected market return (historical rolling average)
    """

    def run_impl(self, inputs, params, **kwargs) -> pd.DataFrame:
        """
        Calculate CAPM expected returns for all months.

        Args:
            inputs: Dict with 'equity_prices' and 'risk_free' DataFrames
            params: Dict with 'horizon_d' and 'beta_method'

        Returns:
            DataFrame with columns: symbol, date, beta_market, rf_rate,
            market_premium, ER_monthly, ER_annual
        """
        # Get parameters
        beta_method = params.get("beta_method", "simple")
        horizon_d = params.get("horizon_d", 252)

        # Prepare data
        prices = inputs["equity_prices"].sort_values(["symbol", "date"]).copy()
        rfr = (
            inputs["risk_free"]
            .groupby("date")["rate"]
            .mean()
            .reset_index()
            .rename(columns={"rate": "rf_rate"})
        )

        # Calculate returns
        prices["ret"] = prices.groupby("symbol")["adj_close"].pct_change()

        # Merge with risk-free rate
        df = prices.merge(rfr, on="date", how="left")
        df["rf_rate"] = df["rf_rate"].fillna(0.0) / 100.0  # Convert from % to decimal

        # Calculate market return proxy (equal-weighted portfolio of all stocks)
        market_returns = (
            df.groupby("date")["ret"]
            .mean()
            .reset_index()
            .rename(columns={"ret": "market_ret"})
        )
        df = df.merge(market_returns, on="date", how="left")

        # Calculate excess returns
        df["excess_ret"] = df["ret"] - df["rf_rate"]
        df["market_excess_ret"] = df["market_ret"] - df["rf_rate"]

        # Calculate rolling beta for each symbol
        window = 60  # 60-month rolling window (5 years)
        min_periods = 24  # Minimum 2 years of data

        betas = []
        for symbol, group in df.groupby("symbol"):
            group = group.sort_values("date").copy()

            # Rolling covariance and variance
            cov = (
                group["excess_ret"]
                .rolling(window=window, min_periods=min_periods)
                .cov(group["market_excess_ret"])
            )
            var = (
                group["market_excess_ret"]
                .rolling(window=window, min_periods=min_periods)
                .var()
            )

            # Beta = Cov(R_i, R_m) / Var(R_m)
            group["beta_market"] = cov / var
            group["beta_market"] = group["beta_market"].fillna(1.0)  # Default beta = 1

            betas.append(group)

        df = pd.concat(betas, ignore_index=True)

        # Calculate expected market premium (historical rolling average)
        df["market_premium"] = (
            df.groupby("symbol")["market_excess_ret"]
            .rolling(window=window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df["market_premium"] = df["market_premium"].fillna(
            df["market_excess_ret"]
        )  # Use current if not enough history

        # CAPM expected return formula: E(R) = Rf + β * Market_Premium
        df["ER_monthly"] = df["rf_rate"] + df["beta_market"] * df["market_premium"]

        # Annualize expected return: (1 + r_monthly)^12 - 1
        df["ER_annual"] = (1 + df["ER_monthly"]) ** 12 - 1

        # Select and clean output
        output = df[
            [
                "symbol",
                "date",
                "beta_market",
                "rf_rate",
                "market_premium",
                "ER_monthly",
                "ER_annual",
            ]
        ].copy()

        # Remove rows with NaN (early periods without enough data)
        output = output.dropna()

        return output
