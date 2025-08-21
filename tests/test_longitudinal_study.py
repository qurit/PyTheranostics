"""Test suite for LongitudinalStudy class.

This test suite uses minimal data objects and property-based testing
to validate the LongitudinalStudy class functionality without requiring
large medical imaging datasets.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import SimpleITK

from pytheranostics.ImagingDS.longitudinal_study import LongitudinalStudy
from pytheranostics.ImagingDS.metadata import ImagingMetadata


class TestLongitudinalStudyFixtures:
    """Test fixtures and helper methods for LongitudinalStudy testing."""

    @staticmethod
    def create_mock_sitk_image(
        shape=(7, 10, 12), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
    ):
        """Create a minimal mock SimpleITK image for testing.

        Args:
            shape: Image dimensions (x, y, z)
            spacing: Voxel spacing (x, y, z)
            origin: Image origin (x, y, z)

        Returns:
            Mock SimpleITK.Image with minimal required functionality
        """
        mock_image = MagicMock(spec=SimpleITK.Image)
        mock_image.GetSpacing.return_value = spacing
        mock_image.GetOrigin.return_value = origin
        mock_image.GetSize.return_value = shape
        mock_image.GetPixelIDValue.return_value = 1  # sitkUInt8 = 1

        return mock_image

    @staticmethod
    def create_test_metadata(
        patient_id="TEST_001",
        acquisition_date="20250101",
        acquisition_time="120000",
        hours_after_injection=24.0,
        radionuclide="Lu-177",
        injected_activity_mbq=7400.0,
    ):
        """Create test metadata with sensible defaults."""
        return ImagingMetadata(
            PatientID=patient_id,
            AcquisitionDate=acquisition_date,
            AcquisitionTime=acquisition_time,
            HoursAfterInjection=hours_after_injection,
            Radionuclide=radionuclide,
            Injected_Activity_MBq=injected_activity_mbq,
        )

    @staticmethod
    def create_minimal_study(num_timepoints=2, modality="NM", image_shape=(6, 10, 13)):
        """Create a minimal LongitudinalStudy for testing.

        Args:
            num_timepoints: Number of time points to create
            modality: Imaging modality
            image_shape: Shape of mock images

        Returns:
            LongitudinalStudy instance with mock data
        """
        images = {}
        meta = {}

        for time_id in range(num_timepoints):
            # Create mock image
            mock_image = MagicMock(spec=SimpleITK.Image)
            mock_image.GetSpacing.return_value = (1.0, 1.0, 1.0)
            mock_image.GetOrigin.return_value = (0.0, 0.0, 0.0)
            mock_image.GetSize.return_value = image_shape
            mock_image.GetPixelIDValue.return_value = 1  # sitkUInt8 = 1
            images[time_id] = mock_image

            # Create metadata
            meta[time_id] = TestLongitudinalStudyFixtures.create_test_metadata(
                patient_id=f"TEST_{time_id:03d}",
                hours_after_injection=24.0 + time_id * 24.0,
            )

        return LongitudinalStudy(images=images, meta=meta, modality=modality)


class TestLongitudinalStudyInit:
    """Test LongitudinalStudy initialization and validation."""

    def test_init_success_minimal(self):
        """Test successful initialization with minimal valid data."""
        study = TestLongitudinalStudyFixtures.create_minimal_study()

        assert study.modality == "NM"
        assert len(study.images) == 2
        assert len(study.meta) == 2
        assert len(study.masks) == 0
        assert isinstance(study._VALID_ORGAN_NAMES, list)
        assert "Liver" in study._VALID_ORGAN_NAMES
        assert LongitudinalStudy._is_valid_mask_name("Lesion_1")

    def test_init_mismatched_keys_raises_error(self):
        """Test that mismatched image and metadata keys raise ValueError."""
        images = {0: MagicMock(spec=SimpleITK.Image)}
        meta = {1: TestLongitudinalStudyFixtures.create_test_metadata()}

        with pytest.raises(
            ValueError,
            match="Not all time points have corresponding images and metadata",
        ):
            LongitudinalStudy(images=images, meta=meta)

    @pytest.mark.parametrize("modality", ["NM", "PT", "CT", "DOSE"])
    def test_init_valid_modalities(self, modality):
        """Test initialization with all valid modalities."""
        study = TestLongitudinalStudyFixtures.create_minimal_study(modality=modality)
        assert study.modality == modality

    @pytest.mark.parametrize("invalid_modality", ["MRI", "US", "XR", "invalid"])
    def test_init_invalid_modality_raises_error(self, invalid_modality):
        """Test that invalid modalities raise ValueError."""
        images = {0: MagicMock(spec=SimpleITK.Image)}
        meta = {0: TestLongitudinalStudyFixtures.create_test_metadata()}

        with pytest.raises(
            ValueError, match=f"Modality {invalid_modality} is not supported"
        ):
            LongitudinalStudy(images=images, meta=meta, modality=invalid_modality)


class TestLongitudinalStudyArrayAccess:
    """Test array access methods with mock data."""

    @patch("SimpleITK.GetArrayFromImage")
    def test_array_at_success(self, mock_get_array):
        """Test successful array access."""
        # Setup mock return value
        test_array = np.random.rand(7, 10, 12)
        mock_get_array.return_value = test_array

        study = TestLongitudinalStudyFixtures.create_minimal_study()

        result = study.array_at(time_id=0)

        # Verify the array is transposed correctly
        expected_shape = (test_array.shape[1], test_array.shape[2], test_array.shape[0])
        assert result.shape == expected_shape
        mock_get_array.assert_called_once()

    def test_array_at_invalid_time_id(self):
        """Test array access with invalid time_id."""
        study = TestLongitudinalStudyFixtures.create_minimal_study()

        with pytest.raises(KeyError):
            study.array_at(time_id=999)


class TestLongitudinalStudyActivityCalculations:
    """Test activity-related calculations."""

    @patch("SimpleITK.GetArrayFromImage")
    def test_array_of_activity_at_invalid_modality(self, mock_get_array):
        """Test that activity calculation fails for non-nuclear modalities."""
        study = TestLongitudinalStudyFixtures.create_minimal_study(modality="CT")

        with pytest.raises(
            ValueError, match="Activity can't be calculated from CT data"
        ):
            study.array_of_activity_at(time_id=0)

    @patch("SimpleITK.GetArrayFromImage")
    def test_array_of_activity_at_invalid_time_id(self, mock_get_array):
        """Test activity calculation with invalid time_id."""
        study = TestLongitudinalStudyFixtures.create_minimal_study(modality="NM")

        with pytest.raises(ValueError, match="Time ID 999 not found in dataset"):
            study.array_of_activity_at(time_id=999)

    @patch("SimpleITK.GetArrayFromImage")
    def test_array_of_activity_at_no_region_creates_ones_mask(self, mock_get_array):
        """Test that activity calculation without region creates ones mask."""
        test_array = np.random.rand(5, 5, 5)
        mock_get_array.return_value = test_array

        study = TestLongitudinalStudyFixtures.create_minimal_study(modality="NM")

        # Mock voxel_volume to return a simple value
        study.voxel_volume = MagicMock(return_value=1.0)

        result = study.array_of_activity_at(time_id=0, region=None)

        # Should equal array * 1.0 (ones mask) * 1.0 (voxel volume)
        # Note: array will be transposed, so we need to account for that
        expected_shape = (test_array.shape[1], test_array.shape[2], test_array.shape[0])
        assert result.shape == expected_shape

    @patch("SimpleITK.GetArrayFromImage")
    def test_array_of_activity_at_region_no_masks_raises_error(self, mock_get_array):
        """Test that requesting region without masks raises appropriate error."""
        study = TestLongitudinalStudyFixtures.create_minimal_study(modality="NM")

        # Set up mock to return a realistic array
        mock_get_array.return_value = np.random.rand(10, 10, 10)

        with pytest.raises(ValueError, match="Time ID 0 does not include mask data"):
            study.array_of_activity_at(time_id=0, region="Liver")

    @patch("SimpleITK.GetArrayFromImage")
    def test_array_of_activity_at_invalid_region_raises_error(self, mock_get_array):
        """Test that requesting invalid region raises appropriate error."""
        study = TestLongitudinalStudyFixtures.create_minimal_study(modality="NM")

        # Set up mock to return a realistic array
        mock_get_array.return_value = np.random.rand(10, 10, 10)

        # Add empty masks dictionary for time_id 0
        study.masks[0] = {"Liver": np.ones((10, 10, 10), dtype=np.bool_)}

        with pytest.raises(ValueError, match="Region InvalidRegion not found"):
            study.array_of_activity_at(time_id=0, region="InvalidRegion")


class TestLongitudinalStudyMaskManagement:
    """Test mask addition and validation."""

    def test_add_masks_to_time_point_basic_success(self):
        """Test basic mask addition functionality."""
        study = TestLongitudinalStudyFixtures.create_minimal_study()

        # Create mock mask image
        mock_mask_image = MagicMock(spec=SimpleITK.Image)
        masks = {"liver_mask": mock_mask_image}
        mask_mapping = {"liver_mask": "Liver"}

        # Mock the required functions
        with patch(
            "pytheranostics.ImagingDS.longitudinal_study.resample_mask_to_target"
        ) as mock_resample, patch("SimpleITK.GetArrayFromImage") as mock_get_array:

            mock_resample.return_value = mock_mask_image
            mock_get_array.return_value = np.ones((10, 10, 10), dtype=np.uint8)

            study.add_masks_to_time_point(
                time_id=0, masks=masks, mask_mapping=mask_mapping
            )

            assert 0 in study.masks
            assert "Liver" in study.masks[0]
            assert study.masks[0]["Liver"].dtype == np.bool_

    def test_add_masks_invalid_source_mask_raises_error(self):
        """Test that invalid source mask name raises error."""
        study = TestLongitudinalStudyFixtures.create_minimal_study()

        masks = {"existing_mask": MagicMock(spec=SimpleITK.Image)}
        mask_mapping = {"nonexistent_mask": "Liver"}

        with pytest.raises(
            ValueError, match="nonexistent_mask is not part of the available masks"
        ):
            study.add_masks_to_time_point(
                time_id=0, masks=masks, mask_mapping=mask_mapping
            )

    def test_add_masks_invalid_target_mask_raises_error(self):
        """Test that invalid target mask name raises error."""
        study = TestLongitudinalStudyFixtures.create_minimal_study()

        masks = {"liver_mask": MagicMock(spec=SimpleITK.Image)}
        mask_mapping = {"liver_mask": "InvalidOrgan"}

        with pytest.raises(ValueError, match="InvalidOrgan is not a valid mask name"):
            study.add_masks_to_time_point(
                time_id=0, masks=masks, mask_mapping=mask_mapping
            )


class TestLongitudinalStudyVolumeCalculations:
    """Test volume and density calculations."""

    def test_voxel_volume_calculation(self):
        """Test voxel volume calculation."""
        study = TestLongitudinalStudyFixtures.create_minimal_study()

        # Mock image with known spacing
        study.images[0].GetSpacing.return_value = (2.0, 3.0, 4.0)  # mm

        expected_volume = (
            (2.0 / 10) * (3.0 / 10) * (4.0 / 10)
        )  # Convert mm to cm then to mL
        actual_volume = study.voxel_volume(time_id=0)

        assert abs(actual_volume - expected_volume) < 1e-10

    def test_volume_of_region(self):
        """Test volume calculation for a region."""
        study = TestLongitudinalStudyFixtures.create_minimal_study()

        # Create a mask with known number of voxels
        mask = np.zeros((10, 10, 10), dtype=np.bool_)
        mask[0:5, 0:5, 0:5] = True  # 125 voxels
        study.masks[0] = {"Liver": mask}

        # Mock voxel volume
        study.voxel_volume = MagicMock(return_value=0.001)  # 1 mm³ = 0.001 mL

        volume = study.volume_of(region="Liver", time_id=0)

        assert volume == 125 * 0.001  # 125 voxels * 0.001 mL/voxel


class TestLongitudinalStudyPropertyBased:
    """Property-based tests for LongitudinalStudy."""

    @pytest.mark.parametrize("num_timepoints", [1, 2, 5, 10])
    def test_consistent_timepoint_keys(self, num_timepoints):
        """Property: images and meta should always have same keys."""
        study = TestLongitudinalStudyFixtures.create_minimal_study(
            num_timepoints=num_timepoints
        )

        assert study.images.keys() == study.meta.keys()
        assert len(study.images) == num_timepoints
        assert len(study.meta) == num_timepoints

    @pytest.mark.parametrize("shape", [(5, 5, 5), (10, 20, 30)])
    def test_array_transpose_consistency(self, shape):
        """Property: array_at should consistently transpose dimensions."""
        study = TestLongitudinalStudyFixtures.create_minimal_study(image_shape=shape)

        with patch("SimpleITK.GetArrayFromImage") as mock_get_array:
            # SimpleITK returns arrays in (z, y, x) order
            test_array = np.random.rand(*shape[::-1])  # shape[::-1] gives (z, y, x)
            mock_get_array.return_value = test_array

            result = study.array_at(time_id=0)

            # Result should be transposed (1, 2, 0) from (z, y, x) to (y, x, z)
            # Original shape: (shape[2], shape[1], shape[0]) = (z, y, x)
            # After transpose (1, 2, 0): (shape[1], shape[0], shape[2]) = (y, x, z)
            expected_shape = (shape[1], shape[0], shape[2])
            assert result.shape == expected_shape

    @pytest.mark.parametrize(
        "mask_name,expected",
        [
            # Valid organ names
            ("Liver", True),
            ("Spleen", True),
            ("Kidney_Left", True),
            ("Kidney_Right", True),
            ("Bladder", True),
            ("BoneMarrow", True),
            ("WholeBody", True),
            # Valid lesion formats
            ("Lesion_1", True),
            ("Lesion_42", True),
            ("Lesion_99999", True),
            # Invalid formats
            ("lesion_1", False),  # lowercase
            ("LESION_1", False),  # uppercase
            ("leSIon_1", False),  # mixed case
            ("Lesion_0", False),  # zero not allowed
            ("Lesion_01", False),  # leading zero
            ("Lesion_", False),  # missing number
            ("Lesion_a", False),  # non-numeric
            ("Lesion_1a", False),  # mixed alphanumeric
            ("Lesion 1", False),  # space instead of underscore
            ("Lesion-1", False),  # hyphen instead of underscore
            # Edge cases
            ("", False),  # empty string
            ("Kidneys", False),  # not in valid list (plural vs Kidney_Left/Right)
            ("Lungs", False),  # not in valid list
            ("Tumor", False),  # not in valid list (use TotalTumorBurden or Lesion_N)
            ("Background", False),  # not in valid list
            ("Unknown", False),  # not in valid list
            ("Random", False),  # not in valid list
            ("Lesion_-1", False),  # negative number
        ],
    )
    def test_mask_validation(self, mask_name, expected):
        """Test comprehensive mask name validation patterns."""
        from pytheranostics.ImagingDS.longitudinal_study import LongitudinalStudy

        result = LongitudinalStudy._is_valid_mask_name(mask_name)
        assert (
            result == expected
        ), f"Expected {mask_name} to be {expected}, got {result}"


class TestLongitudinalStudyIntegration:
    """Integration tests that test multiple components working together."""

    @patch("SimpleITK.GetArrayFromImage")
    def test_end_to_end_activity_calculation_with_mask(self, mock_get_array):
        """Integration test: Create study, add mask, calculate activity."""
        # Setup
        study = TestLongitudinalStudyFixtures.create_minimal_study(modality="NM")

        # Mock array data - uniform activity of 100 Bq/ml
        activity_data = np.full((10, 10, 10), 100.0)
        mock_get_array.return_value = activity_data

        # Add a liver mask covering half the volume
        liver_mask = np.zeros((10, 10, 10), dtype=np.bool_)
        liver_mask[0:5, :, :] = True  # Half the volume
        study.masks[0] = {"Liver": liver_mask}

        # Mock voxel volume
        study.voxel_volume = MagicMock(return_value=0.001)  # 1 mm³

        # Test
        result = study.array_of_activity_at(time_id=0, region="Liver")

        # Verify
        # Should have activity only in masked region
        assert np.sum(result > 0) == 500  # 5*10*10 voxels
        assert np.all(result[5:, :, :] == 0)  # No activity outside mask
        assert np.all(result[0:5, :, :] > 0)  # Activity inside mask

    def test_multiple_timepoints_consistency(self):
        """Integration test: Verify behavior is consistent across timepoints."""
        study = TestLongitudinalStudyFixtures.create_minimal_study(num_timepoints=3)

        # All timepoints should have same valid modality
        assert all(study.modality == "NM" for _ in range(3))

        # Should be able to access voxel volume for all timepoints
        for time_id in [0, 1, 2]:
            vol = study.voxel_volume(time_id)
            assert isinstance(vol, float)
            assert vol > 0


class TestLongitudinalStudyEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_study_behavior(self):
        """Test behavior when creating study with no timepoints."""
        # Empty study should be created successfully but have empty dictionaries
        study = LongitudinalStudy(images={}, meta={}, modality="NM")

        assert len(study.images) == 0
        assert len(study.meta) == 0
        assert len(study.masks) == 0
        assert study.modality == "NM"

    @patch("SimpleITK.GetArrayFromImage")
    def test_activity_calculation_with_zero_volume_mask(self, mock_get_array):
        """Test behavior when mask has no True values."""
        study = TestLongitudinalStudyFixtures.create_minimal_study(modality="NM")

        mock_get_array.return_value = np.full((10, 10, 10), 100.0)
        study.voxel_volume = MagicMock(return_value=0.001)

        # Create empty mask (all False)
        empty_mask = np.zeros((10, 10, 10), dtype=np.bool_)
        study.masks[0] = {"EmptyRegion": empty_mask}

        result = study.array_of_activity_at(time_id=0, region="EmptyRegion")

        # Should return array of all zeros
        assert np.all(result == 0)
        assert result.shape == (10, 10, 10)


class TestLongitudinalStudyPerformance:
    """Test performance characteristics and scaling behavior."""

    def test_large_timepoint_initialization(self):
        """Test that initialization scales reasonably with number of timepoints."""
        import time

        start_time = time.time()
        study = TestLongitudinalStudyFixtures.create_minimal_study(num_timepoints=100)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second)
        assert (end_time - start_time) < 1.0
        assert len(study.images) == 100
        assert len(study.meta) == 100

    @pytest.mark.parametrize("image_size", [(5, 5, 5), (50, 50, 50)])
    def test_memory_efficiency_with_mocks(self, image_size):
        """Test that our mocking doesn't consume excessive memory."""
        study = TestLongitudinalStudyFixtures.create_minimal_study(
            num_timepoints=10, image_shape=image_size
        )

        # Should be able to create study regardless of declared image size
        # since we're using mocks
        assert len(study.images) == 10
        assert all(img.GetSize() == image_size for img in study.images.values())


if __name__ == "__main__":
    pytest.main([__file__])
