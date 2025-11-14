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
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Split mapping into non-identity and identity pairs.

    Returns (non_identity_pairs, identity_pairs).
    """
    non_identity = [(k, v) for k, v in mapping.items() if k != v]
    identity = [(k, v) for k, v in mapping.items() if k == v]
    return non_identity, identity


def summarize_used_mappings(
    used_mappings: Dict[int, Dict[str, Dict[str, str]]],
    *,
    verbose: bool = False,
    sample_limit: int = 20,
    save_json_path: str | Path | None = "mapping_applied_summary.json",
    include_unmapped: bool = True,
) -> None:
    """Print a compact mapping summary and optionally save full details to JSON.

    Parameters
    ----------
    used_mappings : Dict[int, Dict[str, Dict[str, str]]]
        The mapping dictionary returned by create_studies_with_masks()
        where each timepoint has {"ct": {raw->canonical}, "spect": {raw->canonical}}.
    verbose : bool, optional
        If True, print up to `sample_limit` non-identity pairs per modality.
    sample_limit : int, optional
        How many pairs to print per modality when verbose=True.
    save_json_path : str | Path | None, optional
        When provided, save the full per-timepoint mapping details to this JSON file.
        Set to None to skip saving.
    include_unmapped : bool, optional
        If True, also report identity (unmapped) pairs separately for CT and SPECT.
    """
    per_tp = {}
    for tp, studies in sorted(used_mappings.items()):
        ct_mapping = studies.get("ct", {})
        spect_mapping = studies.get("spect", {})

        ct_mapped, ct_unmapped = _split_modalities(ct_mapping)
        spect_mapped, spect_unmapped = _split_modalities(spect_mapping)

        entry = {
            "ct": ct_mapped,
            "spect": spect_mapped,
        }

        if include_unmapped:
            entry["unmapped_ct"] = sorted([k for k, v in ct_unmapped])
            entry["unmapped_spect"] = sorted([k for k, v in spect_unmapped])

        per_tp[tp] = entry

    # Compact counts
    for tp, parts in per_tp.items():
        ct_n = len(parts["ct"])
        sp_n = len(parts["spect"])
        msg = f"tp{tp}: CT {ct_n} | SPECT {sp_n} non-identity mappings"
        if include_unmapped:
            unmapped_ct_n = len(parts.get("unmapped_ct", []))
            unmapped_sp_n = len(parts.get("unmapped_spect", []))
            msg += f" | Unmapped: CT {unmapped_ct_n}, SPECT {unmapped_sp_n}"
        print(msg)

        if verbose:

            def _print_pairs(label: str, pairs: Iterable[Tuple[str, str]]) -> None:
                shown = 0
                for k, v in pairs:
                    print(f"  {label}: {k} -> {v}")
                    shown += 1
                    if shown >= sample_limit:
                        break

            _print_pairs("CT", parts["ct"])
            _print_pairs("SPECT", parts["spect"])

        # Optionally list a small sample of unmapped names per modality
        if include_unmapped:
            if parts.get("unmapped_ct"):
                sample_ct = parts["unmapped_ct"][:sample_limit]
                print(f"  Unmapped CT (identity): {sample_ct}")
            if parts.get("unmapped_spect"):
                sample_sp = parts["unmapped_spect"][:sample_limit]
                print(f"  Unmapped SPECT (identity): {sample_sp}")

    if save_json_path is not None:
        out = {
            int(tp): {
                "ct": [{"from": k, "to": v} for k, v in parts["ct"]],
                "spect": [{"from": k, "to": v} for k, v in parts["spect"]],
                **(
                    {
                        "unmapped_ct": parts.get("unmapped_ct", []),
                        "unmapped_spect": parts.get("unmapped_spect", []),
                    }
                    if include_unmapped
                    else {}
                ),
            }
            for tp, parts in per_tp.items()
        }
        save_path = Path(save_json_path)
        with save_path.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved detailed mapping summary to {save_path}")
