# Voxel Dosimetry Validation Assets

This directory stores optional assets for the slow voxel dosimetry validation test:

- `rtstructs/`: precomputed RT-STRUCT DICOM files for the SNMMI dosimetry dataset
- `expected_results.csv`: reference subset of `VoxelSDosimetry.results`
- `expected_df_ad.csv`: reference subset of `VoxelSDosimetry.df_ad`
- `voi_mappings_config.json` (optional): custom ROI mapping overrides for the validation run
- `dosimetry_fit_defaults.json` (optional): custom ROI fit defaults for the validation run

The validation test is implemented in `tests/test_voxel_dosimetry_validation.py`.
It will skip automatically unless the RT-STRUCT directory and both CSV files are present.

Recommended contents of the CSV files:
- Include only stable, comparable columns.
- Avoid object/list-valued columns such as raw fit-parameter arrays unless you serialize and compare them intentionally.
- Keep the index as ROI names so it matches the DataFrame produced by the pipeline.
