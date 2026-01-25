"""Project initialization and scaffolding utilities for PyTheranostics.

This module provides tools to create and configure new PyTheranostics projects
with standardized directory structures and configuration templates.
"""

from pathlib import Path
from typing import List, Optional


def _get_template_dir() -> Path:
    """Get the path to configuration templates directory."""
    # Navigate from this file to the data/configuration_templates directory
    module_dir = Path(__file__).parent
    template_dir = module_dir / "data" / "configuration_templates"

    if not template_dir.exists():
        raise FileNotFoundError(
            f"Template directory not found at {template_dir}. "
            "PyTheranostics may not be installed correctly."
        )

    return template_dir


def init_project(
    project_dir: str | Path,
    project_name: Optional[str] = None,
    templates: Optional[List[str]] = None,
    create_subdirs: bool = True,
    overwrite: bool = False,
) -> Path:
    """Initialize a new PyTheranostics project with directory structure and configs.

    Creates a project directory with:
    - Configuration files from templates
    - Standard subdirectories for data, results, segmentations, etc.
    - README with basic project information

    Parameters
    ----------
    project_dir : str | Path
        Path where the project should be created. Will be created if it doesn't exist.
    project_name : str, optional
        Name of the project. If None, uses the directory name.
    templates : List[str], optional
        List of template names to copy. If None, copies all available templates.
        Available: ['total_seg_config.json', 'voi_mappings_config.json']
    create_subdirs : bool, optional
        If True, creates standard subdirectories (data/, results/, etc.), by default True.
    overwrite : bool, optional
        If True, overwrites existing config files. If False, skips existing files,
        by default False.

    Returns
    -------
    Path
        The path to the created project directory.

    Examples
    --------
    >>> from pytheranostics.project import init_project
    >>> init_project("./my_dosimetry_project")
    Created project: /path/to/my_dosimetry_project
    ├── total_seg_config.json
    ├── voi_mappings_config.json
    ├── README.md
    ├── data/
    ├── results/
    ├── segmentations/
    └── rtstructs/

    >>> # Initialize with only specific templates
    >>> init_project("./kidney_study", templates=['voi_mappings_config.json'])

    >>> # Minimal setup without subdirectories
    >>> init_project("./simple_project", create_subdirs=False)
    """
    project_dir = Path(project_dir).resolve()
    project_name = project_name or project_dir.name

    # Create project directory
    project_dir.mkdir(parents=True, exist_ok=True)
    print(f"Initializing PyTheranostics project: {project_dir}")

    # Get template directory
    template_dir = _get_template_dir()

    # Determine which templates to copy
    available_templates = {
        "total_seg_config.json": "TotalSegmentator ROI filtering/renaming/combining",
        "voi_mappings_config.json": "VOI name mappings for CT/SPECT analysis",
        "dosimetry_fit_defaults.json": "Dosimetry fit parameters for organs and lesions",
    }

    if templates is None:
        templates_to_copy = list(available_templates.keys())
    else:
        templates_to_copy = templates
        # Validate template names
        for t in templates_to_copy:
            if t not in available_templates:
                print(
                    f"⚠️  Warning: Unknown template '{t}'. "
                    f"Available: {list(available_templates.keys())}"
                )

    # Copy configuration templates
    copied_configs = []
    skipped_configs = []

    for template_name in templates_to_copy:
        if template_name not in available_templates:
            continue

        dest_path = project_dir / template_name
        if dest_path.exists() and not overwrite:
            skipped_configs.append(template_name)
            continue

        try:
            template_path = template_dir / template_name
            # Read template content and write to destination
            template_content = template_path.read_text()
            dest_path.write_text(template_content)
            copied_configs.append(template_name)
        except Exception as e:
            print(f"⚠️  Could not copy {template_name}: {e}")

    # Create standard subdirectories
    subdirs_created = []
    if create_subdirs:
        standard_dirs = {
            "data": "Raw DICOM data and downloaded datasets",
            "results": "Analysis outputs, plots, and reports",
            "segmentations": "TotalSegmentator outputs (.nii.gz files)",
            "rtstructs": "RT-STRUCT DICOM files",
            "notebooks": "Jupyter notebooks for analysis",
        }

        for dirname, description in standard_dirs.items():
            dir_path = project_dir / dirname
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                subdirs_created.append(dirname)

    # Create README
    readme_path = project_dir / "README.md"
    if not readme_path.exists() or overwrite:
        readme_content = f"""# {project_name}

PyTheranostics project initialized on {Path.cwd()}

## Project Structure

```
{project_name}/
├── total_seg_config.json       # TotalSegmentator configuration
├── voi_mappings_config.json    # VOI name mappings
├── data/                        # Raw DICOM data
├── results/                     # Analysis outputs
├── segmentations/              # TotalSegmentator outputs
├── rtstructs/                  # RT-STRUCT DICOM files
└── notebooks/                  # Jupyter notebooks
```

## Configuration Files

### total_seg_config.json
Configure which anatomical structures to include in RT-STRUCT files:
- Set `include: true/false` to filter organs
- Use `new_name` to rename structures
- Use `combine` section to merge structures (e.g., all ribs → "Ribs")

### voi_mappings_config.json
Map VOI names between different naming conventions:
- `ct_mappings`: Morphology-based names (e.g., "Kidney_L_m")
- `spect_mappings`: Activity-based names (e.g., "Kidney_L_a")

### dosimetry_fit_defaults.json
Configure default fit parameters for dosimetry calculations:
- `organ_defaults`: Parameters applied to all organs
- `organs`: Override specific organ kinetics (e.g., BoneMarrow)
- `lesion_defaults`: Parameters for auto-discovered lesions
- `lesions.pattern`: Regex pattern for lesion ROI names

## Getting Started

```python
from pytheranostics.segmentation import totalseg_segment, convert_masks_to_rtstruct

# Run TotalSegmentator
result = totalseg_segment(
    root_dir="./data/ct_series",
    base_output_dir="./segmentations",
    device="mps"
)

# Convert to RT-STRUCT with your config
convert_masks_to_rtstruct(
    segmentation_base_dir="./segmentations",
    ct_series_paths=result["ct_paths"],
    rtstruct_output_dir="./rtstructs",
    config_path="total_seg_config.json"
)
```

## Documentation

- PyTheranostics: https://github.com/pytheranostics/pytheranostics
- TotalSegmentator: https://doi.org/10.1148/ryai.230024
"""
        readme_path.write_text(readme_content)
        print("✓ Created README.md")

    # Print summary
    print("\n" + "=" * 60)
    print(f"✓ Project initialized: {project_dir}")
    print("=" * 60)

    if copied_configs:
        print("\nConfiguration files:")
        for config in copied_configs:
            desc = available_templates.get(config, "")
            print(f"  ✓ {config}")
            if desc:
                print(f"    └─ {desc}")

    if skipped_configs:
        print("\nSkipped (already exist):")
        for config in skipped_configs:
            print(f"  ⊗ {config} (use overwrite=True to replace)")

    if subdirs_created:
        print("\nDirectories created:")
        for dirname in subdirs_created:
            print(f"  ✓ {dirname}/")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Edit configuration files to match your project needs")
    print("  2. Place DICOM data in data/ directory")
    print("  3. Run segmentation and analysis workflows")
    print("=" * 60 + "\n")

    return project_dir


def list_templates() -> dict:
    """List available project templates.

    Returns
    -------
    dict
        Dictionary mapping template names to descriptions.

    Examples
    --------
    >>> from pytheranostics.project import list_templates
    >>> templates = list_templates()
    >>> for name, desc in templates.items():
    ...     print(f"{name}: {desc}")
    """
    return {
        "total_seg_config.json": "TotalSegmentator ROI filtering/renaming/combining",
        "voi_mappings_config.json": "VOI name mappings for CT/SPECT analysis",
        "dosimetry_fit_defaults.json": "Dosimetry fit parameters for organs and lesions",
    }


def get_template_path(template_name: str) -> Path:
    """Get the path to a specific configuration template.

    Useful for inspecting template contents before initializing a project.

    Parameters
    ----------
    template_name : str
        Name of the template file.

    Returns
    -------
    Path
        Path to the template file within the package.

    Examples
    --------
    >>> from pytheranostics.project import get_template_path
    >>> template = get_template_path("total_seg_config.json")
    >>> import json
    >>> config = json.loads(template.read_text())
    """
    template_dir = _get_template_dir()
    template_path = template_dir / template_name

    if not template_path.exists():
        available = list_templates()
        raise FileNotFoundError(
            f"Template '{template_name}' not found. "
            f"Available templates: {list(available.keys())}"
        )

    return template_path
