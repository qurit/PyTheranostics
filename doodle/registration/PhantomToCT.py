import SimpleITK
from pathlib import Path
from doodle.dicomtools.dicomtools import sitk_load_dcm_series
from typing import Optional
from SimpleITK.SimpleITK import Transform
from doodle.registration.demons import multiscale_demons

class PhantomToCTBoneReg:
    """
    Register an XCAT Phantom to a patient CT. Following SimpleITK-Notebooks repository.

    Parameters
    ----------
        CT_dir:

        phantom_dir: Path to directory containing Phantom DICOM data. 
            Phantom Data represents the bone+Marrow anatomy of an standard Male XCAT Phantom.

    Limitations:
        For optimal performance, CT and Phantom should roughly cover the same
        FOV, and be oriented along the same direction.
    """
    
    def __init__(self,
                 CT_dcm_dir: Path,
                 phantom_bone_dcm_dir: Path = Path("../data/bone_phantom")) -> None:
        
        self.CT = sitk_load_dcm_series(dcm_dir=CT_dcm_dir)
        self.Phantom = sitk_load_dcm_series(dcm_dir=phantom_bone_dcm_dir)  # **

        # Set Origin for Phantom to that of reference CT.
        self.Phantom.SetOrigin(self.CT.GetOrigin())  # **

        # Threshold phantom and CT to get bone anatomy.
        self.Phantom = SimpleITK.Cast(self.Phantom > 0, SimpleITK.sitkFloat32)  # **
        self.CT = SimpleITK.Cast(self.CT > 100, SimpleITK.sitkFloat32)

        # Instance of Rigid and Elastic Registration Algorithms
        self.InitialTransform: Optional[Transform] = None
        self.RigidTransform: Optional[Transform] = None
        self.ElasticTransform: Optional[Transform] = None

    def initial_alignment(self, 
                          fixed_image: SimpleITK.SimpleITK.Image,
                          moving_image: SimpleITK.SimpleITK.Image) -> None:
        
        self.InitialTransform = SimpleITK.CenteredTransformInitializer(
                                fixed_image,
                                moving_image,
                                SimpleITK.Euler3DTransform(),
                                SimpleITK.CenteredTransformInitializerFilter.GEOMETRY,
                            )
        
        return None
    
    def rigid_alignment(self, 
                        fixed_image: SimpleITK.SimpleITK.Image, 
                        moving_image: SimpleITK.SimpleITK.Image) -> None:
        """Rigid Registration. Please Note currently we only support default parameters.
        TODO: add support for user-defined registration parameters, probably using **kwargs"""

        registration_method = SimpleITK.ImageRegistrationMethod()

        # Similarity metric settings.
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)

        registration_method.SetInterpolator(SimpleITK.sitkLinear)

        # Optimizer settings.
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Don't optimize in-place
        if self.InitialTransform is None:
            raise AssertionError("Initial Transform was not applied.")
        
        registration_method.SetInitialTransform(self.InitialTransform, inPlace=False)


        self.RigidTransform = registration_method.Execute(
            SimpleITK.Cast(fixed_image, SimpleITK.sitkFloat32), SimpleITK.Cast(moving_image, SimpleITK.sitkFloat32)
        )

        return None
    
    def elastic_alignment(self,
                          fixed_image: SimpleITK.SimpleITK.Image, 
                          moving_image: SimpleITK.SimpleITK.Image) -> None:

        # Define a simple callback which allows us to monitor registration progress.
        def iteration_callback(filter):
            print(f"Iteration: {filter.GetElapsedIterations()}, Metric: {filter.GetMetric()}\n")
            
        # Select a Demons filter and configure it.
        demons_filter =  SimpleITK.FastSymmetricForcesDemonsRegistrationFilter()
        demons_filter.SetNumberOfIterations(150)
        
        # Regularization (update field - viscous, total field - elastic).
        demons_filter.SetSmoothDisplacementField(True)
        demons_filter.SetStandardDeviations(2.0)

        # Add our simple callback to the registration filter.
        demons_filter.AddCommand(SimpleITK.sitkIterationEvent, lambda: iteration_callback(demons_filter))

        # Run the registration.
        self.ElasticTransform = multiscale_demons(
            registration_algorithm=demons_filter, 
            fixed_image = fixed_image, 
            moving_image = moving_image, 
            shrink_factors=[4, 2], 
            smoothing_sigmas=[8, 4])
        
        return None

    @staticmethod
    def transform(fixed_image: SimpleITK.SimpleITK.Image, 
                  moving_image: SimpleITK.SimpleITK.Image,
                  transform: Transform) -> SimpleITK.SimpleITK.Image:
        if transform is None:
            raise AssertionError("Transform was not calculated.")
        
        return SimpleITK.Resample(moving_image, fixed_image, transform, SimpleITK.sitkLinear,
                                  0.0, moving_image.GetPixelID())

    def register(self, 
                 fixed_image: SimpleITK.SimpleITK.Image, 
                 moving_image: SimpleITK.SimpleITK.Image
                 ) -> SimpleITK.SimpleITK.Image:
        
        # Initial Geometric Transform: Align images in space.
        self.initial_alignment(fixed_image=fixed_image, moving_image=moving_image)

        moving_image = self.transform(fixed_image=fixed_image,
                                      moving_image=moving_image,
                                      transform=self.InitialTransform)
        
        # Rigid Registration
        if self.RigidTransform is None:
            print("Computing Rigid Registration ...")
            self.rigid_alignment(fixed_image=fixed_image, moving_image=moving_image)

        moving_image = self.transform(fixed_image=fixed_image,
                                      moving_image=moving_image,
                                      transform=self.RigidTransform)
        
        # Elastic Registration
        if self.ElasticTransform is None:
            print("Computing Elastic Registration, This might take several minutes ...")
            self.elastic_alignment(fixed_image=fixed_image, moving_image=moving_image)

        return self.transform(fixed_image=fixed_image,
                              moving_image=moving_image,
                              transform=self.ElasticTransform)
        
    def register_mask(self, 
                      fixed_image: SimpleITK.SimpleITK.Image, 
                      mask_dcm_dir: Path = Path("../data/bone_marrow")
                 ) -> SimpleITK.SimpleITK.Image:
        mask_image = sitk_load_dcm_series(dcm_dir=mask_dcm_dir)
        mask_image.SetOrigin(fixed_image.GetOrigin())
        mask_image = SimpleITK.Cast(mask_image, SimpleITK.sitkFloat32)

        return self.register(fixed_image=fixed_image,
                             moving_image=mask_image)