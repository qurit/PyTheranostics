import SimpleITK
from typing import Tuple

def register_ct_to_spect(ct_image: SimpleITK.Image, spect_image: SimpleITK.Image
                         ) -> Tuple[SimpleITK.Image, SimpleITK.Transform]:
    """
    Registers a CT image to a SPECT image and applies the resulting transformation to a CT-derived mask.

    Parameters:
    - ct_image: sitk.Image
        The moving image (CT) to be registered.
    - spect_image: sitk.Image
        The fixed image (SPECT) to which CT will be registered.

    Returns:
    - registered_ct: sitk.Image
        The CT image resampled in SPECT space.
    - final_transform: sitk.Transform
        The transform mapping CT space to SPECT space.
    """
    # Initialize registration method
    registration_method = SimpleITK.ImageRegistrationMethod()

    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.MetricUseFixedImageGradientFilterOff()
    registration_method.MetricUseMovingImageGradientFilterOff()

    # Optimizer settings
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Interpolator
    registration_method.SetInterpolator(SimpleITK.sitkLinear)

    # Setup initial transform (rigid)
    initial_transform = SimpleITK.CenteredTransformInitializer(
        spect_image,
        ct_image,
        SimpleITK.Euler3DTransform(),
        SimpleITK.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Execute registration
    final_transform = registration_method.Execute(
        SimpleITK.Cast(spect_image, SimpleITK.sitkFloat32),
        SimpleITK.Cast(ct_image, SimpleITK.sitkFloat32)
    )

    # Resample CT into SPECT space
    registered_ct = SimpleITK.Resample(
        ct_image,
        spect_image,
        final_transform,
        SimpleITK.sitkLinear,
        0.0,
        ct_image.GetPixelID()
    )

    return registered_ct,  final_transform

def transform_ct_mask_to_spect(mask: SimpleITK.Image, spect: SimpleITK.Image, transform: SimpleITK.Transform) -> SimpleITK.Image:
    """_summary_

    Parameters
    ----------
    mask : SimpleITK.Image
        _description_
    spect : SimpleITK.Image
        _description_
    transform : SimpleITK.Transform
        _description_

    Returns
    -------
    SimpleITK.Image
        _description_
    """
    return SimpleITK.Resample(
        mask,
        spect,
        transform,
        SimpleITK.sitkNearestNeighbor,
        0,
        mask.GetPixelID()
    )
    
    