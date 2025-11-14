# PyTheranostics Data Layout

All reference assets that ship with the library live under this directory so
they are available to both library code and tests via `importlib.resources`.

```
pytheranostics/data/
├── phantom/
│   ├── human/   # ICRP organ masses, literature tables, supporting workbooks
│   └── mouse/   # Preclinical phantom masses, scaling factors, literature data
├── olinda/
│   └── templates/
│       ├── human/   # Adult male/female OLINDA case templates
│       └── mouse/   # Mouse-specific case templates (e.g., mouse25g)
├── s-values/
│   ├── organ/   # Radionuclide/sex-specific organ S-value tables
│   └── spheres.json
├── monte_carlo/ # Geant4/GATE templates used by voxel dosimetry
└── phantom/     # (additional imaging phantoms, e.g., skeleton masks)
```

When adding new assets, prefer extending these folders (or adding a clearly
named subdirectory) instead of nesting data inside individual subpackages.
