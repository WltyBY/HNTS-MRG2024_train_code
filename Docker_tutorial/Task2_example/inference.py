"""
The following is a simple example algorithm for Task 2 (mid-RT segmentation) of the HNTS-MRG 2024 challenge.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-task-2-mid-rt-segmentation | gzip -c > example-algorithm-task-2-mid-rt-segmentation.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

import os
import torch
import SimpleITK as sitk
import numpy as np

from pathlib import Path
from glob import glob
from scipy import ndimage
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.DFUNet.DFUNet import DFUNet, open_yaml

INPUT_PATH = Path("/input")  # these are the paths that Docker will use
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def get_ND_bounding_box(volume, margin=None):
    """
    Get the bounding box of nonzero region in an ND volume.
    :param volume: An ND numpy array.
    :param margin: (list)
        The margin of bounding box along each axis.
    :return bb_min: (list) A list for the minimal value of each axis
            of the bounding box.
    :return bb_max: (list) A list for the maximal value of each axis
            of the bounding box.
    """
    input_shape = volume.shape
    if margin is None:
        margin = [0] * len(input_shape)
    assert len(input_shape) == len(margin)
    indxes = np.nonzero(volume)
    bb_min = []
    bb_max = []
    for i in range(len(input_shape)):
        bb_min.append(int(indxes[i].min()))
        bb_max.append(int(indxes[i].max()) + 1)

    for i in range(len(input_shape)):
        bb_min[i] = max(bb_min[i] - margin[i], 0)
        bb_max[i] = min(bb_max[i] + margin[i], input_shape[i])

    return bb_min, bb_max


def crop_ND_volume_with_bounding_box(volume, bb_min, bb_max):
    """
    Extract a subregion form an ND image.
    :param volume: The input ND array (z, y, x).
    :param bb_min: (list) The lower bound of the bounding box for each axis. z, y, x
    :param bb_max: (list) The upper bound of the bounding box for each axis. z, y, x
    :return: A croped ND image (z, y, x).
    """
    dim = len(volume.shape)
    assert dim >= 2 and dim <= 5
    assert bb_max[0] - bb_min[0] <= volume.shape[0]
    if dim == 2:
        output = volume[bb_min[0] : bb_max[0], bb_min[1] : bb_max[1]]
    elif dim == 3:
        output = volume[
            bb_min[0] : bb_max[0], bb_min[1] : bb_max[1], bb_min[2] : bb_max[2]
        ]
    elif dim == 4:
        output = volume[
            bb_min[0] : bb_max[0],
            bb_min[1] : bb_max[1],
            bb_min[2] : bb_max[2],
            bb_min[3] : bb_max[3],
        ]
    elif dim == 5:
        output = volume[
            bb_min[0] : bb_max[0],
            bb_min[1] : bb_max[1],
            bb_min[2] : bb_max[2],
            bb_min[3] : bb_max[3],
            bb_min[4] : bb_max[4],
        ]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output


def preprocess(image_array_lst, channel_names, origin, spacing):
    # crop an image array to the region of head in axis x and y based on intensity.
    assert isinstance(image_array_lst, list)

    bbmin = []
    bbmax = []
    for img_array, channel_name in zip(image_array_lst, channel_names):
        input_shape = img_array.shape
        if any(
            [True if s in channel_name else False for s in ["mask", "label", "seg"]]
        ):
            mask = np.asarray(img_array > 0)
            bbmin_now, bbmax_now = get_ND_bounding_box(mask, margin=[2, 2, 2])
        else:
            mask = np.asarray(img_array > 60)
            se = np.ones([3, 3, 3])
            mask = ndimage.binary_closing(mask, se, iterations=2)
            bbmin_now, bbmax_now = get_ND_bounding_box(mask, margin=[2, 0, 0])

            mask_findhead = mask[input_shape[0] // 2 :]
            bbmin_head, bbmax_head = get_ND_bounding_box(mask_findhead)
            bbmin_now[1:], bbmax_now[1:] = bbmin_head[1:], bbmax_head[1:]

        if not bbmin:
            bbmin, bbmax = bbmin_now, bbmax_now
        else:
            for i in range(len(bbmin)):
                bbmin[i] = max(min(bbmin[i], bbmin_now[i]), 0)
                bbmax[i] = min(max(bbmax[i], bbmax_now[i]), input_shape[i])

    img_crop = []
    for img in image_array_lst:
        img_crop.append(crop_ND_volume_with_bounding_box(img, bbmin, bbmax)[None])
    new_origin = [origin[i] + bbmin[::-1][i] * spacing[i] for i in range(len(origin))]

    return np.vstack(img_crop), new_origin


def run():
    """
    Main function to read input, process the data, and write output.
    """

    ### 1. Read the input(s) from the correct path(s).
    # Currently reads input as a SimpleITK image. You should only need to read in the inputs that your algorithm uses.
    mid_rt_t2w_head_neck = load_image_file(
        location=INPUT_PATH
        / "images/mid-rt-t2w-head-neck",  # Make sure to read from this path; this is exactly how the Grand Challenge will give the input to your container.
    )
    # pre_rt_t2w_head_neck = load_image_file(
    #     location=INPUT_PATH / "images/pre-rt-t2w-head-neck", # Make sure to read from this path; this is exactly how the Grand Challenge will give the input to your container.
    # )
    # pre_rt_head_neck_segmentation = load_image_file(
    #     location=INPUT_PATH / "images/pre-rt-head-neck-segmentation", # Make sure to read from this path; this is exactly how the Grand Challenge will give the input to your container.
    # )
    registered_pre_rt_head_neck = load_image_file(
        location=INPUT_PATH
        / "images/registered-pre-rt-head-neck",  # Make sure to read from this path; this is exactly how the Grand Challenge will give the input to your container.
    )
    registered_pre_rt_head_neck_segmentation = load_image_file(
        location=INPUT_PATH
        / "images/registered-pre-rt-head-neck-segmentation",  # Make sure to read from this path; this is exactly how the Grand Challenge will give the input to your container.
    )

    ### 2. Process the inputs any way you'd like.
    # This is where you should place your relevant inference code.
    _show_torch_cuda_info()

    with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
        print(f.read())

    # Generate a segmentation cube in the middle of the image
    segmentation = generate_segmentation(
        mid_rt_t2w_head_neck,
        registered_pre_rt_head_neck,
        registered_pre_rt_head_neck_segmentation,
    )  # This just a toy example. Make sure to replace this with your inference code.

    ### 3. Save your output to the correct path.
    # Currently takes generated mask (in SimpleITK image format) and writes to MHA file.
    write_mask_file(
        location=OUTPUT_PATH
        / "images/mri-head-neck-segmentation",  # Make sure to write to this path; this is exactly how the Grand Challenge will expect the output from your container.
        segmentation_image=segmentation,
    )

    return 0


def load_image_file(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.mha"))  # Grand Challenge uses MHA files
    result = sitk.ReadImage(input_files[0])

    # Return the sitk image
    return result


def generate_segmentation(mid_img, regis_pre_img, regis_pre_seg):
    """
    Simple example algorithm that uses avaliable pre-RT segmentation mask as a starting point. Erodes label 1 in a segmentation mask and combines it with the original label 2. In other words, the GTVp shrinks and the GTVn stays the same.

    This function takes a pre-RT head and neck segmentation mask as input, erodes label 1 on each 2D axial slice,
    and combines it with the original label 2. The erosion radius is calculated as 20% of the mask size in each slice.
    """
    # Get image properties
    props = {
        "sitk_stuff": {
            # this saves the sitk geometry information. This part is NOT used by nnU-Net!
            "spacing": mid_img.GetSpacing(),
            "origin": mid_img.GetOrigin(),
            "direction": mid_img.GetDirection(),
        },
        # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. NDarrays
        # are returned z, y, x but spacing is returned x, y, z by sitk. Duh.
        "spacing": list(mid_img.GetSpacing())[::-1],
    }

    # Preprocess, array is in z, y, x but new_origin is in x, y, z
    mid_img_array = sitk.GetArrayFromImage(mid_img)
    regis_pre_img_array = sitk.GetArrayFromImage(regis_pre_img)
    regis_pre_seg_array = sitk.GetArrayFromImage(regis_pre_seg)
    preprocessed_array, new_origin = preprocess(
        [mid_img_array, regis_pre_img_array, regis_pre_seg_array],
        ["T2W_mid", "T2W_pre", "mask_pre"],
        props["sitk_stuff"]["origin"],
        props["sitk_stuff"]["spacing"],
    )

    # Use nnUNetPredictor to predict
    PREDICTOR = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    PREDICTOR.initialize_from_trained_model_folder(
        os.path.join(
            "/opt/app/resources/nnUNet_results",
            "Dataset099_HNTSMRG2024mid/MixUpVNetTrainer__nnUNetPlans__3d_fullres",
        ),
        use_folds=(1, 2, 3, 4),
        checkpoint_name="checkpoint_final.pth",
    )
    path = "/opt/app/resources/nnUNet-master/nnunetv2/training/nnUNetTrainer/DFUNet/NetConfig.yaml"
    Network_setting = open_yaml(path)
    dfunet = DFUNet(Network_setting["DFUNet"])
    PREDICTOR.list_of_networks = [dfunet, PREDICTOR.network]
    checkpoint = torch.load(
        "/opt/app/resources/nnUNet_results/Dataset099_HNTSMRG2024mid/DFUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
        map_location=torch.device("cpu"),
    )
    PREDICTOR.list_of_parameters = [checkpoint['network_weights']] + PREDICTOR.list_of_parameters

    segmentation = PREDICTOR.predict_single_npy_array(
        preprocessed_array, props, None, None, False
    )

    # Transform the cropped prediction to original size
    shape_after_cropped = segmentation.shape
    segmentation_origin_shape = np.zeros_like(mid_img_array)
    index_min = np.array(
        [
            (new_origin[i] - props["sitk_stuff"]["origin"][i])
            / props["sitk_stuff"]["spacing"][i]
            for i in range(len(props["sitk_stuff"]["spacing"]))
        ]
    ).astype(int)
    index_max = (index_min + np.array(shape_after_cropped[::-1])).astype(int)
    segmentation_origin_shape[
        index_min[2] : index_max[2],
        index_min[1] : index_max[1],
        index_min[0] : index_max[0],
    ] = segmentation

    # Convert the numpy array back to a SimpleITK image
    segmentation_image = sitk.GetImageFromArray(segmentation_origin_shape)
    segmentation_image.CopyInformation(mid_img)

    return segmentation_image


def write_mask_file(*, location, segmentation_image):
    location.mkdir(parents=True, exist_ok=True)

    # Cast the segmentation image to 8-bit unsigned int
    segmentation_image = sitk.Cast(segmentation_image, sitk.sitkUInt8)

    # Convert the numpy array back to a SimpleITK image
    suffix = ".mha"
    sitk.WriteImage(
        segmentation_image, location / f"output{suffix}", useCompression=True
    )


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
