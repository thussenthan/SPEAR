#!/usr/bin/env python3
"""Thin shim — use 'spear generate-manifest' instead."""

from spear.manifest import generate_manifest_main

if __name__ == "__main__":
    raise SystemExit(generate_manifest_main())
