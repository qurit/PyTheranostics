import SimpleITK
import numpy

def apply_elastic_transform(mask: numpy.ndarray, alpha: float = 10, sigma: float = 3) -> numpy.ndarray:
    """Apply an elastic transformation to a 3-dimensional binary mask using SimpleITK.

    Args:
        mask (numpy.ndarray): Original 3D binary mask 
        alpha (float, optional): Scale of transformation. Higher values have more deformation.. Defaults to 10.
        sigma (float, optional): Smoothness of the deformation. Smaller values make the deformation more local. Defaults to 3.

    Returns:
        numpy.ndarray: Deformed 3D binary mask.
                
    """
    # Convert numpy array to SimpleITK image
    sitk_mask = SimpleITK.GetImageFromArray(mask.astype(numpy.uint8))

    # Define the elastic transform
    elastic_transform = SimpleITK.ElasticTransform(sitk_mask.GetDimension())

    # Set parameters for the elastic transform
    elastic_transform.SetSmoothingSigma(sigma)
    elastic_transform.SetAlpha(alpha)

    # Apply the elastic transform to the mask
    transformed_mask = SimpleITK.Resample(sitk_mask, elastic_transform)

    # Convert the transformed image back to numpy array
    transformed_mask_np = SimpleITK.GetArrayFromImage(transformed_mask)

    return transformed_mask_np

