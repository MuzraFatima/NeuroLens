#!/usr/bin/env python3
"""CLI Entrypoint to serve the TopoCircuit-SAE interactive explorer and REST API."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.server.serve import main

if __name__ == "__main__":
    main()
