import numpy as np
import pandas as pd

from qx.engine.base_model import BaseModel


class ExcessReturnModel(BaseModel):
    """
    Calculate excess returns from OHLCV price data.

    This model:
    1. Loads OHLCV data for multiple symbols
    2. Resamples to specified frequency (weekly, monthly, quarterly, yearly)
    3. Calculates returns (simple or log)
    4. Optionally subtracts risk-free rate for excess returns
    5. Outputs DataFrame with date, symbol, return, frequency
    """

    def run_impl(self, inputs, params, **kwargs) -> pd.DataFrame:
        """
        Calculate returns at specified frequency.

        Args:
            inputs: Dict with 'equity_prices' (required) and 'risk_free' (optional)
            params: Dict with 'frequency', 'return_type', 'min_periods'

        Returns:
            DataFrame with columns: date, symbol, return, frequency
        """
        # Get parameters
        frequency = params["frequency"]  # weekly, monthly, quarterly, yearly
        return_type = params["return_type"]  # simple, log
        min_periods = params["min_periods"]

        # Get price data
        prices = inputs["equity_prices"].copy()
        prices = prices.sort_values(["symbol", "date"])

        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(prices["date"]):
            prices["date"] = pd.to_datetime(prices["date"])

        # Map frequency to pandas resample rule
        freq_map = {
            "weekly": "W",
            "monthly": "ME",  # Month End
            "quarterly": "QE",  # Quarter End
            "yearly": "YE",  # Year End
        }
        resample_freq = freq_map[frequency]

        # Calculate returns for each symbol
        results = []

        for symbol, group in prices.groupby("symbol"):
            # Set date as index for resampling
            df = group.set_index("date").sort_index()

            # Resample to target frequency (use last close price of period)
            resampled = df["close"].resample(resample_freq).last()

            # Remove NaN values
            resampled = resampled.dropna()

            if len(resampled) < min_periods + 1:
                continue  # Not enough data for return calculation

            # Calculate returns
            if return_type == "simple":
                returns = resampled.pct_change()
            elif return_type == "log":
                returns = np.log(resampled / resampled.shift(1))
            else:
                raise ValueError(f"Unknown return_type: {return_type}")

            # Drop NaN (first period has no return)
            returns = returns.dropna()

            if len(returns) == 0:
                continue

            # Create result DataFrame
            result_df = pd.DataFrame(
                {
                    "date": returns.index,
                    "symbol": symbol,
                    "return": returns.values,
                    "frequency": frequency,
                }
            )

            results.append(result_df)

        if not results:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=["date", "symbol", "return", "frequency"])

        # Combine all symbols
        all_returns = pd.concat(results, ignore_index=True)

        # If risk-free rate provided, calculate excess returns
        if "risk_free" in inputs and inputs["risk_free"] is not None:
            rf_data = inputs["risk_free"].copy()

            # Ensure date is datetime
            if not pd.api.types.is_datetime64_any_dtype(rf_data["date"]):
                rf_data["date"] = pd.to_datetime(rf_data["date"])

            # Resample risk-free rate to same frequency
            # Use average rate over the period
            rf_data = rf_data.set_index("date").sort_index()
            rf_resampled = rf_data["rate"].resample(resample_freq).mean()

            # Convert annual rate to period return
            # Assuming rate is annual percentage (e.g., 5.0 for 5%)
            if frequency == "weekly":
                rf_resampled = rf_resampled / 100 / 52  # Annual to weekly
            elif frequency == "monthly":
                rf_resampled = rf_resampled / 100 / 12  # Annual to monthly
            elif frequency == "quarterly":
                rf_resampled = rf_resampled / 100 / 4  # Annual to quarterly
            elif frequency == "yearly":
                rf_resampled = rf_resampled / 100  # Annual to annual

            # Merge with returns
            all_returns["date"] = pd.to_datetime(all_returns["date"])
            rf_df = pd.DataFrame(
                {"date": rf_resampled.index, "rf_rate": rf_resampled.values}
            )

            all_returns = all_returns.merge(rf_df, on="date", how="left")

            # Calculate excess return
            all_returns["return"] = all_returns["return"] - all_returns[
                "rf_rate"
            ].fillna(0)

            # Drop temporary column
            all_returns = all_returns.drop(columns=["rf_rate"])

        # Sort by date and symbol
        all_returns = all_returns.sort_values(["date", "symbol"]).reset_index(drop=True)

        return all_returns[["date", "symbol", "return", "frequency"]]
