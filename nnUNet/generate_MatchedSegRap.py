import os
import re
import shutil
import random
import multiprocessing
import SimpleITK as sitk
import numpy as np

from time import sleep
from tqdm import tqdm
from scipy import ndimage
from skimage.exposure import match_histograms


random_seed = 319
random.seed(random_seed)
np.random.seed(random_seed)
os.environ["PYTHONHASHSEED"] = str(random_seed)

# ____________________for SegRap2023_____________________
def check_all_same(input_list):
    # compare all entries to the first
    for i in input_list[1:]:
        if not len(i) == len(input_list[0]):
            return False
        all_same = all(i[j] == input_list[0][j] for j in range(len(i)))
        if not all_same:
            return False
    return True


def read_sitk_case(file_paths):
    """
    file_paths: pathes of different modalities of the same sample
    """
    images = []
    spacings = []
    directions = []
    origins = []

    for f in file_paths:
        sitk_obj = sitk.ReadImage(f)
        data_array = sitk.GetArrayFromImage(sitk_obj)
        assert len(data_array.shape) == 3, "only 3d images are supported"
        # spacing in x, y, z
        spacings.append(sitk_obj.GetSpacing())
        # images' shape in z, y, x
        images.append(data_array[None])
        origins.append(sitk_obj.GetOrigin())
        directions.append(sitk_obj.GetDirection())

    if not check_all_same([i.shape for i in images]):
        print("ERROR! Not all input images have the same shape!")
        print("Shapes:")
        print([i.shape for i in images])
        print("Image files:")
        print(file_paths)
        raise RuntimeError()
    if not check_all_same(origins):
        print("ERROR! Not all input images have the same origins!")
        print("Origins:")
        print(origins)
        print("Image files:")
        print(file_paths)
        raise RuntimeError()
    if not check_all_same(directions):
        print("ERROR! Not all input images have the same directions!")
        print("Directions:")
        print(directions)
        print("Image files:")
        print(file_paths)
        raise RuntimeError()
    if not check_all_same(spacings):
        print(
            "ERROR! Not all input images have the same spacings! This might be caused by them not "
            "having the same affine"
        )
        print("spacings_for_nnunet:")
        print(spacings)
        print("Image files:")
        print(file_paths)
        raise RuntimeError()

    stacked_images = np.vstack(images)
    dict = {
        "spacing": spacings[0],
        "direction": directions[0],
        "origin": origins[0],
    }

    return stacked_images.astype(np.float32), dict


def get_largest_k_components(image, k=1):
    """
    Get the largest K components from 2D or 3D binary image.
    :param image: The input ND array for binary segmentation.
    :param k: (int) The value of k.
    :return: An output array (k == 1) or a list of ND array (k>1)
        with only the largest K components of the input.
    """
    dim = len(image.shape)
    if image.sum() == 0:
        print("the largest component is null")
        return image
    if dim < 2 or dim > 3:
        raise ValueError("the dimension number should be 2 or 3")
    s = ndimage.generate_binary_structure(dim, 1)
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    sizes_sort = sorted(sizes, reverse=True)
    kmin = min(k, numpatches)
    output = []
    for i in range(kmin):
        labeli = min(np.where(sizes == sizes_sort[i])[0]) + 1
        output_i = np.asarray(labeled_array == labeli, np.uint8)
        output.append(output_i)
    return output[0] if k == 1 else output


def get_human_region_mask_one_channel(img, threshold=-600):
    """
    Get the mask of human region in CT volumes
    """
    dim = len(img.shape)
    if dim == 4:
        img = img[0]
    mask = np.asarray(img > threshold)
    se = np.ones([3, 3, 3])
    mask = ndimage.binary_opening(mask, se, iterations=2)
    mask = get_largest_k_components(mask, 1)
    mask_close = ndimage.binary_closing(mask, se, iterations=2)

    D, H, W = mask.shape
    for d in [1, 2, D - 3, D - 2]:
        mask_close[d] = mask[d]
    for d in range(0, D, 2):
        mask_close[d, 2:-2, 2:-2] = np.ones((H - 4, W - 4))

    # get background component
    bg = np.zeros_like(mask)
    bgs = get_largest_k_components(1 - mask_close, 10)
    for bgi in bgs:
        indices = np.where(bgi)
        if bgi.sum() < 1000:
            break
        if (
            indices[0].min() == 0
            or indices[1].min() == 0
            or indices[2].min() == 0
            or indices[0].max() == D - 1
            or indices[1].max() == H - 1
            or indices[2].max() == W - 1
        ):
            bg = bg + bgi
    fg = 1 - bg

    fg = ndimage.binary_opening(fg, se, iterations=1)
    fg = get_largest_k_components(fg, 1)
    if dim == 4:
        fg = np.expand_dims(fg, 0)
    fg = np.asarray(fg, np.uint8)
    return fg


def create_mask_base_on_human_body(data):
    assert (
        len(data.shape) == 4 or len(data.shape) == 3
    ), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"

    mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        # each channel
        this_mask = get_human_region_mask_one_channel(data[c])
        mask = mask | this_mask
    mask = ndimage.binary_fill_holes(mask)
    return mask


def create_mask_base_on_threshold(data, threshold=None):
    assert (
        len(data.shape) == 4 or len(data.shape) == 3
    ), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"

    if threshold is None:
        threshold = np.min(data)

    mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        # each channel
        this_mask = data[c] != threshold
        mask = mask | this_mask
    mask = ndimage.binary_fill_holes(mask)
    return mask


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


def crop_to_mask(
    data, seg=None, crop_fun_args={}, create_mask=create_mask_base_on_threshold
):
    data_nonzero_mask = create_mask(data, *crop_fun_args)
    data_bbmin, data_bbmax = get_ND_bounding_box(data_nonzero_mask)

    if seg is not None:
        seg_nonzero_mask = create_mask_base_on_threshold(seg, threshold=0)
        if not (np.any(seg_nonzero_mask)):
            # all zero in seg_nonzero_mask
            seg_bbmin, seg_bbmax = data_bbmin, data_bbmax
        else:
            seg_bbmin, seg_bbmax = get_ND_bounding_box(seg_nonzero_mask)

        bbmin, bbmax = [], []
        for i in range(len(data_bbmin)):
            bbmin.append(min(data_bbmin[i], seg_bbmin[i]))
            bbmax.append(max(data_bbmax[i], seg_bbmax[i]))

        data_cropped_lst = []
        for i in range(data.shape[0]):
            data_cropped_lst.append(
                crop_ND_volume_with_bounding_box(data[i], bbmin, bbmax)[None]
            )
        data_cropped = np.vstack(data_cropped_lst)
        seg_cropped = crop_ND_volume_with_bounding_box(seg[0], bbmin, bbmax)

        return data_cropped, seg_cropped[None], [bbmin, bbmax]
    else:
        assert len(data_bbmin) == len(
            data_bbmax
        ), "Length of bbox from data and seg should be the same."

        data_cropped_lst = []
        for i in range(data.shape[0]):
            data_cropped_lst.append(
                crop_ND_volume_with_bounding_box(data[i], data_bbmin, data_bbmax)[None]
            )
        data_cropped = np.vstack(data_cropped_lst)

        return (
            data_cropped,
            np.zeros_like(data_cropped[0])[None],
            [data_bbmin, data_bbmax],
        )


def get_seg(each_seg_path_lst):
    obj = sitk.ReadImage(each_seg_path_lst[0])
    array = sitk.GetArrayFromImage(obj)

    seg_array = np.zeros_like(array)
    for i in range(len(each_seg_path_lst)):
        itk_obj = sitk.ReadImage(each_seg_path_lst[i])
        itk_array = sitk.GetArrayFromImage(itk_obj)
        seg_array[itk_array != 0] = i + 1

    return seg_array[None].astype(np.uint8)


def set_itk_obj_info(itk_obj, info_dict):
    itk_obj.SetDirection(info_dict["direction"])
    itk_obj.SetSpacing(info_dict["spacing"])
    itk_obj.SetOrigin(info_dict["origin"])


def run_case_SegRap(case, ref_path, data_folder, img_save_folder, seg_save_folder):
    case_id = int(case.split("_")[-1])
    data_path = os.path.join(data_folder, case)

    contrast_img_path = os.path.join(data_path, "image_contrast.nii.gz")
    contrast_img_save_path = os.path.join(
        img_save_folder, "MatchedSegRap_{:0>4d}_0000.nii.gz".format(case_id)
    )

    GTVnd_path = os.path.join(data_path, "GTVnd.nii.gz")
    GTVp_path = os.path.join(data_path, "GTVp.nii.gz")
    seg_save_path = os.path.join(
        seg_save_folder, "MatchedSegRap_{:0>4d}.nii.gz".format(case_id)
    )

    ###############___get the class of label you want here___##############
    seg_array = get_seg([GTVp_path, GTVnd_path])

    img_array, info_dict = read_sitk_case([contrast_img_path])

    data_cropped, seg_cropped, bbox = crop_to_mask(
        img_array, seg_array, create_mask=create_mask_base_on_human_body
    )
    info_dict["origin"] = [
        info_dict["origin"][i] + bbox[0][::-1][i] * info_dict["spacing"][i]
        for i in range(len(info_dict["origin"]))
    ]

    # Match histogram
    match_image_lst = []
    for i in range(data_cropped.shape[0]):
        ref_image_array, _ = read_sitk_case([ref_path])
        match_image_lst.append(match_histograms(data_cropped[i][None], ref_image_array))
    matched_image = np.vstack(match_image_lst)

    contrast_img_cropped_obj = sitk.GetImageFromArray(matched_image[0])
    seg_obj = sitk.GetImageFromArray(seg_cropped[0])

    set_itk_obj_info(contrast_img_cropped_obj, info_dict)
    set_itk_obj_info(seg_obj, info_dict)

    sitk.WriteImage(contrast_img_cropped_obj, contrast_img_save_path)
    sitk.WriteImage(seg_obj, seg_save_path)


if __name__ == "__main__":
    num_processes = 16
    HNTSMRG_data_folder = "/media/x/Wlty/nnUNet_workspace/nnUNet_raw/Dataset099_HNTSMRG2024pre/imagesTr"
    SegRap_data_folder = "/media/x/Wlty/LymphNodes/Dataset/SegRap2023_raw/SegRap2023_Training_Set_120cases"
    save_folder = "/media/x/Wlty/nnUNet_workspace/nnUNet_raw/Dataset102_MatchedSegRap"

    img_save_folder = os.path.join(save_folder, "imagesTr")
    seg_save_folder = os.path.join(save_folder, "labelsTr")

    if os.path.isdir(img_save_folder):
        shutil.rmtree(img_save_folder, ignore_errors=True)
    os.makedirs(img_save_folder, exist_ok=True)
    if os.path.isdir(seg_save_folder):
        shutil.rmtree(seg_save_folder, ignore_errors=True)
    os.makedirs(seg_save_folder, exist_ok=True)

    ######################Process SegRap2023########################
    print("Start processing SegRap2023")
    SegRap_case_lst = [i for i in os.listdir(SegRap_data_folder) if "segrap" in i]
    ref_case_lst = os.listdir(HNTSMRG_data_folder)
    ref_case_lst = np.random.choice(ref_case_lst, len(SegRap_case_lst), replace=False)
    r = []
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        for case, ref_path in zip(SegRap_case_lst, ref_case_lst):
            r.append(
                p.starmap_async(
                    run_case_SegRap,
                    (
                        (
                            case,
                            os.path.join(HNTSMRG_data_folder, ref_path),
                            SegRap_data_folder,
                            img_save_folder,
                            seg_save_folder,
                        ),
                    ),
                )
            )
        remaining = list(range(len(SegRap_case_lst)))
        workers = [j for j in p._pool]
        with tqdm(desc=None, total=len(SegRap_case_lst), disable=False) as pbar:
            while len(remaining) > 0:
                all_alive = all([j.is_alive() for j in workers])
                if not all_alive:
                    raise RuntimeError("Some background worker is 6 feet under. Yuck.")
                done = [i for i in remaining if r[i].ready()]
                for _ in done:
                    pbar.update()
                remaining = [i for i in remaining if i not in done]
                sleep(0.1)
