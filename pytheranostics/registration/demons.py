"""Multiscale Demons registration helpers."""

import SimpleITK


def smooth_and_resample(image, shrink_factor, smoothing_sigma):
    """Gaussian smooth an image and resample it by the given shrink factor."""
    smoothed_image = SimpleITK.SmoothingRecursiveGaussian(image, smoothing_sigma)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz / float(shrink_factor) + 0.5) for sz in original_size]
    new_spacing = [
        ((original_sz - 1) * original_spc) / (new_sz - 1)
        for original_sz, original_spc, new_sz in zip(
            original_size, original_spacing, new_size
        )
    ]
    return SimpleITK.Resample(
        smoothed_image,
        new_size,
        SimpleITK.Transform(),
        SimpleITK.sitkLinear,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def multiscale_demons(
    registration_algorithm,
    fixed_image,
    moving_image,
    initial_transform=None,
    shrink_factors=None,
    smoothing_sigmas=None,
):
    """Run a multiscale Demons registration on the provided fixed/moving pair."""
    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(
            list(zip(shrink_factors, smoothing_sigmas))
        ):
            fixed_images.append(
                smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma)
            )
            moving_images.append(
                smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma)
            )

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = SimpleITK.TransformToDisplacementField(
            initial_transform,
            SimpleITK.sitkVectorFloat64,
            fixed_images[-1].GetSize(),
            fixed_images[-1].GetOrigin(),
            fixed_images[-1].GetSpacing(),
            fixed_images[-1].GetDirection(),
        )
    else:
        initial_displacement_field = SimpleITK.Image(
            fixed_images[-1].GetWidth(),
            fixed_images[-1].GetHeight(),
            fixed_images[-1].GetDepth(),
            SimpleITK.sitkVectorFloat64,
        )
        initial_displacement_field.CopyInformation(fixed_images[-1])

    # Run the registration.
    initial_displacement_field = registration_algorithm.Execute(
        fixed_images[-1], moving_images[-1], initial_displacement_field
    )
    # Start at the top of the pyramid and work our way down.
    for f_image, m_image in reversed(
        list(zip(fixed_images[0:-1], moving_images[0:-1]))
    ):
        initial_displacement_field = SimpleITK.Resample(
            initial_displacement_field, f_image
        )
        initial_displacement_field = registration_algorithm.Execute(
            f_image, m_image, initial_displacement_field
        )
    return SimpleITK.DisplacementFieldTransform(initial_displacement_field)
