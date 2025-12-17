import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qx.storage.backend_local import LocalParquetBackend


@dataclass
class TableFormatAdapter:
    backend: LocalParquetBackend
    table_format: str = "parquet"
    write_mode: str = "append"  # "append" or "overwrite"

    def write(
        self,
        data: pd.DataFrame,
        path: str,
        write_mode: str = None,
    ) -> str:
        """
        Write data to storage path (high-level API).

        Args:
            data: DataFrame to write
            path: Directory path (relative or absolute) where data should be written
            write_mode: Write mode - "append" or "overwrite"

        Returns:
            Full path to written file
        """
        from datetime import datetime

        mode = write_mode or self.write_mode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"part-{timestamp}.parquet"

        # Path is already the directory - just write to it
        self.write_batch(data, path, filename, mode=mode)

        return f"{path}/{filename}"

    def write_batch(
        self, df: pd.DataFrame, rel_dir: str, filename: str, mode: str = None
    ) -> str:
        """
        Write a batch of data to storage.

        Args:
            df: DataFrame to write
            rel_dir: Relative directory path
            filename: Output filename
            mode: Write mode - "append" (keep existing) or "overwrite" (delete existing)
                  If None, uses self.write_mode

        Returns:
            Relative path to written file
        """
        write_mode = mode or self.write_mode

        # If overwrite mode, clean the directory first
        if write_mode == "overwrite":
            full_dir = Path(self.backend.root) / rel_dir
            if full_dir.exists():
                # Remove all parquet files in this partition directory
                for pq_file in full_dir.glob("*.parquet"):
                    pq_file.unlink()

        rel_file = f"{rel_dir}/{filename}"
        self.backend.write_parquet(df, rel_file)
        return rel_file

    def list_parts(self, rel_dir: str):
        return self.backend.list_files(rel_dir, suffix=".parquet")
