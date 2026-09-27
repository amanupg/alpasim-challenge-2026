# NOTICE

- `trajectory.py` is copied from NVlabs/alpasim, `e2e_challenge/sample_submission_vavam/vavam_challenge/trajectory.py` (branch `e2e_challenge`, commit 2023534), Apache-2.0, Copyright (c) 2026 NVIDIA Corporation. Only the module docstring was changed.
- `driver.py` follows the structure of the same sample's `driver.py` (Apache-2.0, NVIDIA). The multi-camera buffering, ego-history construction and Alpamayo policy wiring are ours.
- Camera slot indices and the ego-history convention follow `src/driver/src/alpasim_driver/models/alpamayo_base.py` (Apache-2.0, NVIDIA).
- The model is NVIDIA Alpamayo 1.5 (`nvidia/Alpamayo-1.5-10B`), used under its own license; the code is from `NVlabs/alpamayo1.5` (commit 36aeb4c).
