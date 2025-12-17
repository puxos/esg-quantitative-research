# Configuration Naming Fixes - Summary

## Issues Fixed

### 1. ❌ **Naming Conflict: `write_mode`**

**Problem**: The name `write_mode` was used for two different purposes:
- **File write method**: `append` vs `overwrite` (how to write files)
- **Environment selection**: `dev` vs `prod` (where to write files)

**Solution**: Renamed environment mode fields to avoid conflict:
- `read_mode` → `env_read`
- `write_mode` → `env_write`
- `mode` → `env` (backward compat)

**Environment variables renamed**:
- `QX_READ_MODE` → `QX_ENV_READ`
- `QX_WRITE_MODE` → `QX_ENV_WRITE`
- `QX_MODE` → still supported (backward compat)

### 2. ❌ **Config File Confusion**

**Problem**: Two config files with overlapping purposes:
- `config/storage.yaml` - Storage configuration
- `config/settings.yaml` - API keys, universe settings, expected returns, etc.

**Solution**: Merged into single `config/config.yaml`:
- Contains storage settings (backend, paths, environments)
- Contains API keys (Tiingo, FRED)
- Contains universe definitions (SP500, NASDAQ100, etc.)
- Contains expected returns configuration
- Contains logging and scheduling configuration

## Changes Made

### Files Modified

1. **✅ config/config.yaml** (merged from storage.yaml + settings.yaml)
   - Changed: `read_mode` → `env_read`
   - Changed: `write_mode` → `env_write`
   - Changed: `mode` → `env` (backward compat)
   - Merged all settings into single config file

2. **✅ qx/storage/pathing.py**
   - Updated field names: `mode` → `env`, `read_mode` → `env_read`, `write_mode` → `env_write`
   - Updated environment variable names: `QX_READ_MODE` → `QX_ENV_READ`, `QX_WRITE_MODE` → `QX_ENV_WRITE`
   - Updated method names: `effective_mode` → `effective_env`
   - Updated docstrings to reflect new terminology
   - Updated config path: `config/storage.yaml` → `config/config.yaml`

3. **✅ tests/test_env_modes.py** (renamed from test_read_write_mode.py)
   - Updated all test cases to use new naming
   - Changed environment variable references
   - All 6 tests passing ✅

### Files Removed

- ❌ `config/storage.yaml` - Merged into config.yaml
- ❌ `config/settings.yaml` - Merged into config.yaml
- ❌ `tests/test_read_write_mode.py` - Renamed to test_env_modes.py

## New Usage

### Configuration File

```yaml
# config/config.yaml
storage:
  write_mode: append     # File write method: "append" or "overwrite"
  
  # Environment selection
  env_read: prod         # Read from prod environment
  env_write: prod        # Write to prod environment
```

### Environment Variables

```bash
# Separate read/write environments
export QX_ENV_READ=prod
export QX_ENV_WRITE=dev

# Backward compatible single environment
export QX_MODE=prod
```

### Python Code

```python
from qx.storage.pathing import PathResolver

# Explicit separate environments
resolver = PathResolver(env_read="prod", env_write="dev")

# Backward compatible single environment
resolver = PathResolver(env="prod")

# Check configuration
print(f"Read from:  {resolver.env_read}")
print(f"Write to:   {resolver.env_write}")
print(f"Effective:  {resolver.effective_env}")  # "prod", "dev", or "mixed"
```

## Benefits

✅ **No naming conflicts**: File write method (`write_mode: append`) is separate from environment (`env_write: dev`)

✅ **Clearer terminology**: 
- `env_read` / `env_write` clearly indicate environment selection
- `write_mode` clearly indicates file write method

✅ **Single config file**: All configuration in `config/config.yaml`

✅ **Backward compatible**: `QX_MODE` and `env` parameter still work

## Migration Guide

### Old Way (Before)

```bash
# Environment variables
export QX_READ_MODE=prod
export QX_WRITE_MODE=dev

# Config file
# config/storage.yaml
storage:
  read_mode: prod
  write_mode: prod
```

```python
# Python code
resolver = PathResolver(read_mode="prod", write_mode="dev")
print(resolver.effective_mode)
```

### New Way (After)

```bash
# Environment variables
export QX_ENV_READ=prod
export QX_ENV_WRITE=dev

# Config file
# config/config.yaml
storage:
  env_read: prod
  env_write: prod
```

```python
# Python code
resolver = PathResolver(env_read="prod", env_write="dev")
print(resolver.effective_env)
```

## Verification

All tests passing ✅:

```
TEST 1: Default Configuration (from config/config.yaml) ✅
TEST 2: Separate Read/Write Environments (read=prod, write=dev) ✅
TEST 3: Environment Variables (QX_ENV_READ=prod, QX_ENV_WRITE=dev) ✅
TEST 4: Backward Compatibility (env='dev' parameter) ✅
TEST 5: Convenience Constructors ✅
TEST 6: Priority Order (explicit > env > config) ✅
```

Run tests:
```bash
.venv/bin/python tests/test_env_modes.py
```

## Priority Order

Environment settings are determined by (highest to lowest):

1. **Explicit parameters**: `PathResolver(env_read="X", env_write="Y")`
2. **Environment variables**: `QX_ENV_READ`, `QX_ENV_WRITE`
3. **Config file**: `storage.env_read`, `storage.env_write` in `config/config.yaml`
4. **Backward compat**: `QX_MODE` env var or `storage.env` (applies to both)
5. **Default**: `"prod"`

## Documentation Status

- ✅ Core implementation fixed and tested
- ⏳ Documentation files still reference old naming (read_mode, write_mode, storage.yaml)
- 📋 Next step: Update all documentation files

## Files Needing Documentation Updates

Run this to find files with old naming:
```bash
grep -r "read_mode\|write_mode\|QX_READ_MODE\|QX_WRITE_MODE\|storage\.yaml" docs/ examples/
```

These will need updating in a follow-up:
- docs/SEPARATE_READ_WRITE_MODE_*.md
- docs/DEV_PROD_MODE_*.md
- examples/separate_read_write_modes.py
- examples/dev_loader_workflow.py
- SEPARATE_READ_WRITE_MODE_SUMMARY.md

---

**Status**: ✅ Core changes complete and tested  
**Date**: December 17, 2025  
**Breaking Changes**: None (fully backward compatible via `env` parameter and `QX_MODE` env var)
