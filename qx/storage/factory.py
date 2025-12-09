import yaml
from qx.storage.backend_local import LocalParquetBackend
from qx.storage.table_format import TableFormatAdapter

def storage_from_yaml(cfg_path: str):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)["storage"]
    backend = LocalParquetBackend(cfg.get("base_uri","file://."))
    adapter = TableFormatAdapter(backend=backend, table_format=cfg.get("table_format","parquet"), write_mode=cfg.get("write_mode","append"))
    return backend, adapter, cfg
