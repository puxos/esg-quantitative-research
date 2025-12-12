import pandas as pd

from qx.common.contracts import DatasetRegistry
from qx.common.types import AssetClass, DatasetType, Domain
from qx.storage.pathing import PathResolver
from qx.storage.table_format import TableFormatAdapter


class ProcessedWriterBase:
    def __init__(
        self,
        adapter: TableFormatAdapter,
        resolver: PathResolver,
        registry: DatasetRegistry,
    ):
        self.adapter, self.resolver, self.registry = adapter, resolver, registry

    def write(self, df: pd.DataFrame, output_type: str, model: str) -> str:
        run_date = pd.to_datetime(df["run_ts"].iloc[0]).strftime("%Y-%m-%d")
        out_dt = DatasetType(
            Domain.DERIVED_METRICS, AssetClass.EQUITY, output_type, None, None, None
        )
        c = self.registry.find(out_dt)
        rel_dir = self.resolver.processed_dir(
            output_type=output_type, model=model, run_date=run_date, c=c
        )
        filename = f"part-{df['run_id'].iloc[0]}.parquet"
        return self.adapter.write_batch(df, rel_dir, filename)
