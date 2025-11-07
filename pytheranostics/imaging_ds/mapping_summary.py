"""Utilities to summarize applied ROI mappings in longitudinal workflows.

Provides a compact console summary and an optional JSON artifact
without bloating notebook output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _split_modalities(
    mapping: Dict[str, str],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    ct = [(k, v) for k, v in mapping.items() if str(k).endswith("_m")]
    spect = [(k, v) for k, v in mapping.items() if str(k).endswith("_a")]
    other = [
        (k, v)
        for k, v in mapping.items()
        if not (str(k).endswith("_m") or str(k).endswith("_a"))
    ]
    return ct, spect, other


def summarize_used_mappings(
    used_mappings: Dict[int, Dict[str, str]],
    *,
    verbose: bool = False,
    sample_limit: int = 20,
    save_json_path: str | Path | None = "mapping_applied_summary.json",
) -> None:
    """Print a compact mapping summary and optionally save full details to JSON.

    Parameters
    ----------
    used_mappings : Dict[int, Dict[str, str]]
        The mapping dictionary returned by create_studies_with_masks()
        where each timepoint maps {raw_name -> canonical_name}.
    verbose : bool, optional
        If True, print up to `sample_limit` non-identity pairs per modality.
    sample_limit : int, optional
        How many pairs to print per modality when verbose=True.
    save_json_path : str | Path | None, optional
        When provided, save the full per-timepoint mapping details to this JSON file.
        Set to None to skip saving.
    """
    per_tp = {}
    for tp, mp in sorted(used_mappings.items()):
        ct, spect, other = _split_modalities(mp)
        per_tp[tp] = {"ct": ct, "spect": spect, "other": other}

    # Compact counts
    for tp, parts in per_tp.items():
        ct_n = sum(1 for k, v in parts["ct"] if k != v)
        sp_n = sum(1 for k, v in parts["spect"] if k != v)
        ot_n = sum(1 for k, v in parts["other"] if k != v)
        print(f"tp{tp}: CT {ct_n} | SPECT {sp_n} | Other {ot_n} non-identity mappings")

        if verbose:

            def _print_pairs(label: str, pairs: Iterable[Tuple[str, str]]) -> None:
                shown = 0
                for k, v in pairs:
                    if k != v:
                        print(f"  {label}: {k} -> {v}")
                        shown += 1
                        if shown >= sample_limit:
                            break

            _print_pairs("CT", parts["ct"])
            _print_pairs("SPECT", parts["spect"])
            if parts["other"]:
                _print_pairs("Other", parts["other"])

    if save_json_path is not None:
        out = {
            int(tp): {
                "ct": [{"from": k, "to": v} for k, v in parts["ct"]],
                "spect": [{"from": k, "to": v} for k, v in parts["spect"]],
                "other": [{"from": k, "to": v} for k, v in parts["other"]],
            }
            for tp, parts in per_tp.items()
        }
        save_path = Path(save_json_path)
        with save_path.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved detailed mapping summary to {save_path}")
