from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class LocalParquetBackend:
    """Local filesystem backend using PyArrow Parquet."""

    def __init__(self, base_uri: str = "file://."):
        self.root = (
            Path(base_uri.replace("file://", ""))
            if base_uri.startswith("file://")
            else Path(base_uri)
        )
        self.root = self.root.resolve()

    def make_dir(self, rel_path: str):
        (self.root / rel_path).mkdir(parents=True, exist_ok=True)

    def list_files(self, rel_path: str, suffix: str = ".parquet") -> List[str]:
        dirp = self.root / rel_path
        if not dirp.exists():
            return []
        return [str(p.relative_to(self.root)) for p in dirp.glob(f"*{suffix}")]

    def write_parquet(self, df: pd.DataFrame, rel_file_path: str):
        out = self.root / rel_file_path
        out.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, str(out))

    def read_parquet(
        self,
        rel_file_path: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None,
    ) -> pd.DataFrame:
        """
        Read Parquet file with optional column selection and filtering.

        Args:
            rel_file_path: Relative path to Parquet file
            columns: List of column names to read (None = all columns)
            filters: PyArrow filter expressions as list of tuples
                     Format: [(column, operator, value), ...]
                     Operators: '=', '!=', '<', '>', '<=', '>=', 'in', 'not in'
                     Example: [('date', '>=', pd.Timestamp('2020-01-01')),
                              ('ticker', 'in', ['AAPL', 'MSFT'])]

        Returns:
            DataFrame with filtered data
        """
        fp = self.root / rel_file_path
        table = pq.read_table(str(fp), columns=columns, filters=filters)
        return table.to_pandas()
