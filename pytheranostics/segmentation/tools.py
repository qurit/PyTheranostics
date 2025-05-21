from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt

def rtst_to_mask(dicom_series_path,rt_struct_path):
    # Load existing RT Struct. Requires the series path and existing RT Struct path
    rtstruct = RTStructBuilder.create_from(
    dicom_series_path=dicom_series_path, 
    rt_struct_path=rt_struct_path
    )

    # View all of the ROI names from within the image
    print(rtstruct.get_roi_names())
    rois = rtstruct.get_roi_names()

    # Loading the 3D Mask from within the RT Struct
    mask_3d = {}

    for voi in rois:
        mask_3d[voi] = rtstruct.get_roi_mask_by_name(voi)

    return mask_3d
    # # Display one slice of the region
    # first_mask_slice = mask_3d[voi][:, :, 0]
    # plt.imshow(first_mask_slice)
    # plt.show()