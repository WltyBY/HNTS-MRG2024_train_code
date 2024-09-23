"""
The following is a simple example algorithm for Task 1 (pre-RT segmentation) of the HNTS-MRG 2024 challenge.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-task-1-pre-rt-segmentation | gzip -c > example-algorithm-task-1-pre-rt-segmentation.tar.gz

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


INPUT_PATH = Path("/input")  # these are the paths that Docker will use
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
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
        "Dataset098_HNTSMRG2024pre/MixUpVNetTrainer__nnUNetPlans__3d_fullres",
    ),
    use_folds=(0, 1, 2, 3, 4),
    checkpoint_name="checkpoint_final.pth",
)


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


def preprocess(image_array, origin, spacing):
    # crop an image array to the region of head in axis x and y based on intensity.
    assert isinstance(image_array, np.ndarray)

    shape = image_array.shape
    body_mask = np.asarray(image_array > 60)

    se = np.ones([3, 3, 3])
    body_mask = ndimage.binary_closing(body_mask, se, iterations=2)
    bbmin, bbmax = get_ND_bounding_box(body_mask, margin=[2, 0, 0])

    bbmin_head, bbmax_head = get_ND_bounding_box(body_mask[shape[0] // 2 :])
    bbmin[1:], bbmax[1:] = bbmin_head[1:], bbmax_head[1:]

    image_array_crop = crop_ND_volume_with_bounding_box(image_array, bbmin, bbmax)
    new_origin = [origin[i] + bbmin[::-1][i] * spacing[i] for i in range(len(origin))]

    return image_array_crop, new_origin


def run():
    """
    Main function to read input, process the data, and write output.
    """

    ### 1. Read the input from the correct path.
    # Currently reads input as a SimpleITK image.
    pre_rt_t2w_headneck = load_image_file(
        location=INPUT_PATH
        / "images/pre-rt-t2w-head-neck",  # Make sure to read from this path; this is exactly how the Grand Challenge will give the input to your container.
    )

    ### 2. Process the inputs any way you'd like.
    # This is where you should place your relevant inference code.
    _show_torch_cuda_info()

    with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
        print(f.read())

    segmentation = generate_segmentation(
        pre_rt_t2w_headneck
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


def generate_segmentation(image):
    """
    Simple example algorithm. Generates a segmentation with a central cube and a smaller adjacent cube. Outputs the mask in SimpleITK image format.

    This function takes a MRI image as input and creates a segmentation mask.
    The segmentation contains two labeled regions:
      1. A central cube of 50 mm side length centered in the image (corresponds to GTVp).
      2. A smaller cube of 30 mm side length adjacent to the central cube (corresponds to GTVn).
    """

    # Get image properties
    props = {
        "sitk_stuff": {
            # this saves the sitk geometry information. This part is NOT used by nnU-Net!
            "spacing": image.GetSpacing(),
            "origin": image.GetOrigin(),
            "direction": image.GetDirection(),
        },
        # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. NDarrays
        # are returned z, y, x but spacing is returned x, y, z by sitk. Duh.
        "spacing": list(image.GetSpacing())[::-1],
    }

    # Preprocess, array is in z, y, x but new_origin is in x, y, z
    image_array = sitk.GetArrayFromImage(image)
    preprocessed_array, new_origin = preprocess(
        image_array, props["sitk_stuff"]["origin"], props["sitk_stuff"]["spacing"]
    )

    # Use nnUNetPredictor to predict
    segmentation = PREDICTOR.predict_single_npy_array(
        preprocessed_array[None], props, None, None, False
    )

    # Transform the cropped prediction to original size
    shape_after_cropped = segmentation.shape
    segmentation_origin_shape = np.zeros_like(image_array)
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
    segmentation_image.CopyInformation(image)

    return segmentation_image


def write_mask_file(*, location, segmentation_image):
    location.mkdir(parents=True, exist_ok=True)

    # Cast the segmentation image to 8-bit unsigned int
    segmentation_image = sitk.Cast(segmentation_image, sitk.sitkUInt8)

    # Write to a MHA file
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
