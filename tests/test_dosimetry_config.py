from typing import Dict

from pytheranostics.dosimetry.config import build_roi_fit_config


class DummyLongitudinalStudy:
    def __init__(self, masks: Dict[int, Dict[str, object]]):
        self.masks = masks


def test_build_roi_fit_config_merges_organs_and_lesions():
    # Minimal masks with two timepoints, includes lesions
    masks = {
        0: {"Liver": object(), "Lesion_1": object()},
        1: {"Kidney_Left": object(), "Lesion_2": object()},
    }
    longSPECT = DummyLongitudinalStudy(masks=masks)

    roi_cfg = build_roi_fit_config(longSPECT)

    # Organs: expect defaults applied
    assert "Liver" in roi_cfg
    assert "Kidney_Left" in roi_cfg
    assert "fit_order" in roi_cfg["Liver"]
    assert "param_init" in roi_cfg["Liver"]

    # Lesions: auto-discovered and have lesion defaults applied
    for lesion in ("Lesion_1", "Lesion_2"):
        assert lesion in roi_cfg
        assert "param_init" in roi_cfg[lesion]
        assert "fit_order" in roi_cfg[lesion]

    # No unexpected keys
    assert all(isinstance(v, dict) for v in roi_cfg.values())


def test_build_roi_fit_config_respects_pattern_toggle():
    # Custom masks with non-standard lesion naming
    masks = {
        0: {"Liver": object(), "Tumor_01": object()},
    }
    longSPECT = DummyLongitudinalStudy(masks=masks)

    # Use explicit config with custom pattern and disable auto-discovery
    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "dosimetry_fit_defaults.json"
        cfg = {
            "organ_defaults": {"fit_order": 1, "param_init": {"A1": 10, "A2": 0.01}},
            "lesion_defaults": {"fit_order": 1, "param_init": {"A1": 100, "A2": 0.1}},
            "lesions": {"auto_discover": False, "pattern": "^Tumor_(\\d+)$"},
        }
        cfg_path.write_text(json.dumps(cfg))

        roi_cfg = build_roi_fit_config(longSPECT, config_path=cfg_path)
        assert "Liver" in roi_cfg
        # auto_discover disabled; Tumor_01 should not be present
        assert "Tumor_01" not in roi_cfg
