# Voxel Dosimetry Validation Assets

This directory stores lightweight reference assets for the slow voxel dosimetry
validation test:

- `expected_results.csv`: reference subset of `VoxelSDosimetry.results`
- `expected_df_ad.csv`: reference subset of `VoxelSDosimetry.df_ad`
- `voi_mappings_config.json` (optional): custom ROI mapping overrides for the validation run
- `dosimetry_fit_defaults.json` (optional): custom ROI fit defaults for the validation run

The precomputed RT-STRUCT DICOM files are downloaded on demand from Zenodo
record `21893683`: https://zenodo.org/records/21893683.

The validation test is implemented in `tests/test_voxel_dosimetry_validation.py`.
GitHub Actions may skip this test when the external SNMMI Deep Blue dataset
rejects CI-hosted downloads. The Zenodo RT-STRUCT files are still expected to
download and validate when the test runs. Before opening a pull request,
developers should run it locally:

```bash
pytest tests/test_voxel_dosimetry_validation.py -rs
```

Outside CI, SNMMI data-fetch failures fail the test instead of skipping it.

Recommended contents of the CSV files:
- Include only stable, comparable columns.
- Avoid object/list-valued columns such as raw fit-parameter arrays unless you serialize and compare them intentionally.
- Keep the index as ROI names so it matches the DataFrame produced by the pipeline.
