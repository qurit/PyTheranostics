DICOM File Organization
=======================

PyTheranostics provides utilities to organize DICOM files into a structured format suitable for dosimetry workflows. The ``dicom_organizer`` module can process folders of DICOM files and automatically organize them by patient, cycle, and timepoint.

Overview
--------

The organizer handles:

* **Multiple patients** in a single folder
* **Multiple imaging cycles** per patient (e.g., therapy cycles separated by weeks)
* **Multiple timepoints** per cycle (e.g., scans at different times during a cycle)
* **Mixed modalities** (CT, SPECT/NM, PET, RTSTRUCT)
* **Same-day acquisitions** at different times (using datetime-based splitting)

Output structure
----------------

The organizer creates a hierarchical folder structure::

    PatientID/
        Cycle1/
            tp1/
                CT/
                    *.dcm
                SPECT/
                    *.dcm
                CT/
                    RTstruct/
                        *.dcm
            tp2/
                CT/
                SPECT/
        Cycle2/
            tp1/
                ...

Basic Usage
-----------

Organize all DICOM files in a folder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pytheranostics.dicomtools import organize_folder_by_cycles
    
    # Organize all patients in a folder
    result = organize_folder_by_cycles(
        storage_root="/path/to/dicom/files",
        output_base="/path/to/organized/output",
        cycle_gap_days=15,              # New cycle if gap >= 15 days
        timepoint_separation_days=1,    # New timepoint if gap >= 1 day
        move=True                        # Move files (False to copy)
    )
    
    # Result is a dict: {PatientID: {CycleX: {tpY: [Path, ...]}}}
    print(result)

Organize specific patients only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Filter to specific patient IDs
    result = organize_folder_by_cycles(
        storage_root="/path/to/dicom/files",
        output_base="/path/to/organized/output",
        patient_id_filter=["PATIENT001", "PATIENT002"]
    )

Handle same-day acquisitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For protocols with multiple scans on the same day (e.g., morning CT and afternoon SPECT+CT), use fractional days to separate timepoints based on actual acquisition times:

.. code-block:: python

    # Separate timepoints if acquisition times differ by >= 4.8 hours
    result = organize_folder_by_cycles(
        storage_root="/path/to/dicom/files",
        output_base="/path/to/organized/output",
        timepoint_separation_days=0.2  # 0.2 days ≈ 4.8 hours
    )

This uses DICOM acquisition date/time tags (e.g., ``AcquisitionDate``/``AcquisitionTime``, or related series/content/study date/time tags) with file modification time as a fallback to split same-day scans into separate timepoints.

Debugging and Inspection
-------------------------

Use ``summarize_timepoints()`` to inspect detected series before organizing:

.. code-block:: python

    from pytheranostics.dicomtools import summarize_timepoints
    
    # Get summary of all detected series
    summary = summarize_timepoints(
        storage_root="/path/to/dicom/files",
        patient_id_filter=["PATIENT001"]
    )
    
    # Summary shows: study_date, modality, series_number, datetime, and gaps
    for patient_id, entries in summary.items():
        print(f"\n{patient_id}:")
        for entry in entries:
            print(f"  {entry['study_date']} - {entry['modality']} "
                  f"Series{entry['series_number']} at {entry['datetime']} "
                  f"(gap: {entry['delta_hours']:.1f}h)")

Example output::

    PATIENT001:
      20190409 - CT Series2 at 2019-04-09 11:34:57 (gap: None)
      20190409 - NM Series5 at 2019-04-09 16:06:50 (gap: 4.5h)
      20190409 - CT Series2 at 2019-04-09 16:26:59 (gap: 0.3h)
      20190410 - CT Series2 at 2019-04-10 10:15:23 (gap: 17.8h)
      20190413 - NM Series4 at 2019-04-13 14:22:10 (gap: 76.1h)

This helps diagnose issues like:

* Missing timepoints
* Incorrectly merged same-day scans
* Unexpected gaps between acquisitions

Parameters Reference
--------------------

``organize_folder_by_cycles()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:storage_root: Root directory to scan recursively for ``.dcm`` files
:output_base: Base directory for organized output (defaults to ``storage_root``)
:cycle_gap_days: Gap threshold (days) to start a new cycle (default: 15)
:timepoint_separation_days: Gap threshold (days) to start a new timepoint within a cycle (default: 1, can be fractional like 0.1 for 2.4 hours)
:move: If ``True``, move files; if ``False``, copy files (default: ``True``)
:patient_id_filter: List of PatientIDs to process; if ``None``, process all (default: ``None``)

Returns a nested dictionary: ``{PatientID: {"CycleX": {"tpY": [Path, ...]}}}}``

``summarize_timepoints()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:storage_root: Root directory to scan for DICOM files
:patient_id_filter: Optional list of PatientIDs to summarize

Returns: ``{PatientID: [{study_date, modality, series_number, datetime, delta_hours}, ...]}``

Advanced: Integration with DICOM Receiver
------------------------------------------

The organizer can be triggered automatically after receiving DICOM files via C-STORE:

.. code-block:: python

    from pytheranostics.dicomtools.dicom_receiver import DICOMReceiver
    
    receiver = DICOMReceiver(
        ae_title="PYTHERANOSTICS",
        port=11112,
        storage_root="/path/to/storage",
        auto_organize=True,                            # Enable auto-organize
        auto_organize_output_base="/path/to/organized",
        auto_organize_cycle_gap_days=15,
        auto_organize_timepoint_separation_days=0.2,   # 4.8 hours
        auto_organize_debounce_seconds=60              # Wait 60s after last file
    )
    
    receiver.start()

The receiver will automatically call ``organize_folder_by_cycles()`` 60 seconds after the last DICOM file is received for each patient.

See Also
--------

* :doc:`../Data_Ingestion_Examples/Data_Ingestion_Examples` - General data ingestion workflows
* :doc:`../getting_started/project_setup_tutorial` - Initial project setup
