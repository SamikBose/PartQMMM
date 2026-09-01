#!/usr/bin/env python3
"""Legacy compatibility notice for PartQMMM.

Direct ORCA-input generation was removed from the current V1 workflow.
PartQMMM now writes, per snapshot:
  * frame_XXXXXX_qm.xyz
  * frame_XXXXXX_mm.pc

Use generate_partitions.py and let the downstream QM-label workflow consume
those files. This stub intentionally prevents accidental use of the obsolete
pre-V1 ORCA driver.
"""

raise SystemExit(
    "orca_inputs.py is deprecated in PartQMMM V1. "
    "Use generate_partitions.py to write QM XYZ + MM point-charge files."
)
