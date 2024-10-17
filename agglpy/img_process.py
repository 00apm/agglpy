# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:04:44 2020

@author: Artur
"""

import os
from pathlib import Path
from typing import List, Tuple, cast
import warnings

import cv2  # type: ignore

import numpy as np
import numpy.typing as npt
import pandas as pd

from agglpy.errors import SettingsStructureError
from agglpy.logger import logger


def crop_img(image: npt.NDArray, ratio: float) -> npt.NDArray:
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError(
            f"Crop ratio of: {ratio} is out of range. It must be in range 0 < ratio < 1. "
        )
    h, _ = image.shape[:2]
    crop_height = int((1 - ratio) * h)
    work_image = image[:crop_height, :]
    return work_image


def preprocess_img(
    image: npt.NDArray,
    median_blur: int | None = None,
    clahe: Tuple[float, float] | None = None,
) -> npt.NDArray:
    work_image: npt.NDArray = image
    if median_blur:
        logger.debug(f"Applying median blur with kernel size: {median_blur}")
        work_image = cv2.medianBlur(image, median_blur)
    else:
        work_image = image
    if clahe:
        logger.debug(
            f"Applying CLAHE- Contrast Limited Adaptive Histogram Equalization"
            f" with parameters: {str(clahe)}"
        )
        clahe_obj = cv2.createCLAHE(clipLimit=clahe[0], tileGridSize=clahe[1])
        work_image = clahe_obj.apply(work_image)
    return work_image


def HCT(
    img: npt.NDArray,  # image loaded by cv2.imread and converted to GrayScale
    d_min: int,
    d_max: int,
    dist2R: float = 0.4,
    param1: float = 200,
    param2: float = 15,
    display_img: bool = False,
    export_img: bool = False,
    export_img_path: os.PathLike | None = None,
    export_csv: bool = False,
    export_csv_path: os.PathLike | None = None,
    export_edges=False,
    export_edges_path=None,
) -> pd.DataFrame:
    """Detect primary particles using opencv Hough Circle Transform

    Detect primary particles in the image numpy NDArray.
    Assumes that np.NDArray is a Grayscale image loaded via cv2.imread

    Args:
        img (npt.NDArray): 8-bit, single-channel, grayscale input opencv image
        d_min (int): minimum particle diameter to look for
        d_max (int): maximum particle diameter to look for
        dist2R (float, optional): minimum distance between centers of detected
            particles. Expressed in distance to d_max ratio. Defaults to 0.4.
        param1 (float, optional): higher threshold of the two passed to
            the Canny edge detector (the lower one is twice smaller).
            Defaults to 200.
        param2 (float, optional): accumulator threshold for the circle centers
            at the detection stage. The smaller it is, the more false circles
            may be detected. Circles, corresponding to the larger accumulator
            values, will be returned first. Defaults to 15.
        display_img (bool, optional): if true resulting image with detected
            particles drawn will be displayed. Defaults to False.
        export_img (bool, optional): if true resulting image with detected
            particles drawn will be exported. Defaults to False.
        export_img_path (os.PathLike | None, optional): path to the particle
            image to export. Defaults to None.
        export_csv (bool, optional): if true result DataFrame will be exported
            in csv format. Defaults to False.
        export_csv_path (os.PathLike | None, optional): path to the exported
            result csv file. Defaults to None.
        export_edges (bool, optional): if true image after Canny edge detection
            step will be exported. Defaults to False.
        export_edges_path (_type_, optional): path to the edge image file.
            Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containig detected primary particles
            coordinates (X,Y) and radius (R)
    """
    input_params = {
        "d_min": d_min,
        "d_max": d_max,
        "dist2R": dist2R, 
        "param1": param1, 
        "param2": param2,
    }
    logger.debug(
        f"Performing HCT with parameters: {input_params} ..."
    ) 
    work_image = img
    circles = cv2.HoughCircles(
        img,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=dist2R * d_max / 2,
        param1=param1,
        param2=param2,
        minRadius=int(d_min / 2),
        maxRadius=int(d_max / 2),
    )
    # Check if any particles were found
    if (circles is None) or ((circles[0].ndim and circles[0].size) == 0):
        wmsg = (
            f"No particles detected. HCT resulted in empty dataframe for "
            f"arguments: d_min={d_min}, d_max={d_max}, dist2R={dist2R}, "
            f"param1={param1}, param2={param2}"
        )
        warnings.warn(wmsg)
        logger.warning(wmsg)
        circlesDF = pd.DataFrame([], columns=["X", "Y", "R"])
    else:
        circlesDF = pd.DataFrame(circles[0], columns=["X", "Y", "R"])
    logger.debug(
        f"HCT detected: {len(circlesDF.index)} primary particles."
    )
    # Displaying and exporting images
    if display_img == True or export_img == True:
        imgC = draw_particles(work_image, circlesDF)
        if export_edges == True:
            imgE: npt.NDArray | None = cv2.Canny(
                work_image, threshold1=param1, threshold2=0.5 * param1
            )
        else:
            imgE = None
        if display_img == True:
            cv2.imshow("circles", imgC)
            if export_edges == True:
                cv2.imshow("edges", imgE)
            cv2.waitKey(0)  # waits until a key is pressed
            cv2.destroyAllWindows()  # destroys the window showing image
        if export_img == True:
            if not export_img_path:
                raise ValueError(
                    f"Trying to export HCT result circle image, but "
                    f"export_img_path was not specified."
                )
            else:
                ex_img_pth = Path(export_img_path)
            cv2.imwrite(str(ex_img_pth), imgC)
            if export_edges == True:
                if not export_edges_path:
                    raise ValueError(
                        f"Trying to export HCT result edges image, but "
                        f"export_edges_path was not specified."
                    )
                else:
                    ex_edges_pth = Path(export_edges_path)
                cv2.imwrite(str(ex_edges_pth), imgE)

    # Exporting dataframe to csv file
    if export_csv:
        circlesDF.index.name = "ID"
        circlesDF.to_csv(export_csv_path)
    return circlesDF


def HCT_multi(
    img: npt.NDArray,  # image loaded by cv2.imread and converted to GrayScale
    d_min: List[int] | int,
    d_max: List[int] | int,
    dist2R: List[float] | float = 0.4,
    param1: List[float | int] | float | int= 200,
    param2: List[float | int] | float | int = 15,
    export_img: bool = False,
    export_csv: bool = False,
    export_edges: bool = False,
    export_namebase: str | None = None,
    export_dir: os.PathLike | None = None,
) -> pd.DataFrame:

    # Convert single number inputs to lists
    if isinstance(d_min, int):
        d_min = [d_min]
    if isinstance(d_max, int):
        d_max = [d_max]
    if isinstance(dist2R, float):
        dist2R = [dist2R]
    if isinstance(param1, (float, int)):
        param1 = [param1]
    if isinstance(param2, (float, int)):
        param2 = [param2]

    d_min_arr = np.array(d_min, dtype=np.int64)
    d_max_arr = np.array(d_max, dtype=np.int64)
    dist2R_arr = np.array(dist2R, dtype=np.float64)
    param1_arr = np.array(param1, dtype=np.float64)
    param2_arr = np.array(param2, dtype=np.float64)

    # Find the maximum length among the arrays
    max_length = max(
        d_min_arr.size,
        d_max_arr.size,
        dist2R_arr.size,
        param1_arr.size,
        param2_arr.size,
    )

    # Broadcast arrays to the max length or raise error if broadcasting is not possible
    d_min_arr = broadcast_to_max_length(d_min_arr, max_length)
    d_max_arr = broadcast_to_max_length(d_max_arr, max_length)
    dist2R_arr = broadcast_to_max_length(dist2R_arr, max_length)
    param1_arr = broadcast_to_max_length(param1_arr, max_length)
    param2_arr = broadcast_to_max_length(param2_arr, max_length)

    exp_dir: Path | None = None
    exp_csv_path: Path | None = None
    export_any: bool = export_csv or export_edges or export_img
    if export_any:
        if (export_namebase is None) or (export_dir is None):
            raise ValueError(
                f"Trying to export HCT results, but export_namebase or "
                f"export_dir was not defined."
            )
        else:
            exp_dir = Path(export_dir)
            exp_csv_path = exp_dir / (export_namebase + "_HCT.csv")

    logger.debug(
        f"Starting multiple HCT particle detection with {max_length} parameter sets"
    )

    dmin: int
    dmax: int
    d2R: float
    p1: float
    p2: float
    circlesDF: pd.DataFrame
    df_list: List[pd.DataFrame] = []  # partial dataframe collector for concat
    for i, (dmin, dmax, d2R, p1, p2) in enumerate(
        zip(
            d_min_arr,
            d_max_arr,
            dist2R_arr,
            param1_arr,
            param2_arr,
        )
    ):

        imname = (
            f"{str(i + 1)}_{export_namebase}_D({str(dmin)}-{str(dmax)})"
            f"_p1({str(p1)})_p2({str(p2)})"
        )
        if export_any:
            # ensure that mypy recognize that exp_dir is a Path at this point
            exp_dir = cast(Path, exp_dir)
            cirname = imname + "_circles.jpg"
            cirpath = exp_dir / cirname
            edgename = imname + "_edges.jpg"
            edgepath = exp_dir / edgename
        else:
            cirpath = None
            edgepath = None

        edgename = imname + "_edges.jpg"
        buffDF = HCT(
            img=img,
            d_min=dmin,
            d_max=dmax,
            dist2R=d2R,
            param1=p1,
            param2=p2,
            display_img=False,
            export_img=export_img,
            export_img_path=cirpath,
            export_csv=False,
            export_csv_path=None,
            export_edges=export_edges,
            export_edges_path=edgepath,
        )
        df_list.append(buffDF)
        
    circlesDF = pd.concat(df_list, ignore_index=True)
    logger.debug(
        f"Multiple HCT in total detected: {len(circlesDF.index)} primary particles."
    )
    if export_csv:
        circlesDF.index.name = "ID"
        circlesDF.to_csv(exp_csv_path)
    return circlesDF


def HCT_from_file(
    img_path: Path,
    **kwargs,
) -> pd.DataFrame:
    """Wrapper function for HCT function

    Load an image from img_path and perform Hough Circle Transform on it.

    Args:
        img_path (os.PathLike): path to an image file
        **kwargs: Additional keyword arguments to be passed to `HCT`.
            See `HCT` for a complete list of arguments.

    Returns:
        pd.DataFrame: DataFrame with primary particles coordinates and radius
    """
    img = cv2.imread(str(img_path))

    # Check if the image was loaded successfully
    if img is None:
        raise ValueError(f"Could not load image from path: {img_path}")

    # Convert BGR to Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Call the original HCT_multi function with the loaded image and unpacked kwargs
    return HCT(img=img, **kwargs)


def HCT_multi_from_file(
    img_path: os.PathLike,
    **kwargs,
) -> pd.DataFrame:
    """Wrapper function for HCT_multi

    Load an image from img_path and perform Hough Circle Transform on it.

    Args:
        img_path (os.PathLike): path to an image file
        **kwargs: Additional keyword arguments to be passed to `HCT_multi`.
            See `HCT_multi` for a complete list of arguments.

    Returns:
        pd.DataFrame: DataFrame with primary particles coordinates and radius
    """
    img = cv2.imread(str(img_path))

    # Check if the image was loaded successfully
    if img is None:
        raise ValueError(f"Could not load image from path: {img_path}")

    # Convert BGR to Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Call the original HCT_multi function with the loaded image and unpacked kwargs
    return HCT_multi(img=img, **kwargs)


def draw_particles(
    image: npt.NDArray,
    particles: pd.DataFrame,
    color: npt.NDArray = np.array([100, 255, 100], dtype=np.uint8),
) -> npt.NDArray:

    alpha: float = 0.2
    work_image = image.copy()
    overlay = image.copy()

    # Validate the color array
    if color.shape != (3,):
        raise ValueError(
            f"Expected color to be an array of shape (3,), but got shape {color.shape}"
        )
    if not np.all((0 <= color) & (color <= 255)):
        raise ValueError(
            f"Color values should be in the range 0-255, but got {color}"
        )

    for i, row in particles.iterrows():
        X = row.X
        Y = row.Y
        R = row.R
        cv2.circle(
            overlay, (int(X), int(Y)), int(R), tuple(color.tolist()), -1
        )
        cv2.circle(work_image, (int(X), int(Y)), int(R), (255, 255, 255), 1)

    cv2.addWeighted(overlay, alpha, work_image, 1 - alpha, 0, work_image)

    return work_image


def broadcast_to_max_length(arr: npt.NDArray, max_length: int) -> npt.NDArray:
    """Broadcasts an array of length 1 to the max_length or returns the array if lengths match."""
    if arr.size == 1:
        return np.full(max_length, arr[0])
    elif arr.size == max_length:
        return arr
    else:
        raise ValueError(
            f"Array of length {len(arr)} cannot be broadcast to length {max_length}."
        )
