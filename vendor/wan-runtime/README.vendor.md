# Vendored Wan runtime snapshot

This directory contains the `wan/` Python runtime package from the official Wan2.2 repository.

- Upstream: https://github.com/Wan-Video/Wan2.2
- Commit: 42bf4cfaa384bc21833865abc2f9e6c0e67233dc
- License: see `LICENSE.txt`
- Vendoring policy: setup.py installs dependencies only; it does not clone the upstream repository and does not download model weights.

Local patch:
- `wan/__init__.py` is reduced to the TI2V import surface used by this extension, avoiding optional Wan tasks that are not part of this extension.
