from dataclasses import dataclass

import pandas as pd

from qx.storage.backend_local import LocalParquetBackend


@dataclass
class TableFormatAdapter:
    backend: LocalParquetBackend
    table_format: str = "parquet"
    write_mode: str = "append"

    def write_batch(self, df: pd.DataFrame, rel_dir: str, filename: str) -> str:
        rel_file = f"{rel_dir}/{filename}"
        self.backend.write_parquet(df, rel_file)
        return rel_file

    def list_parts(self, rel_dir: str):
        return self.backend.list_files(rel_dir, suffix=".parquet")
