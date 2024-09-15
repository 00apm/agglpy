# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:04:44 2020

@author: Artur
"""

import os

import cv2

# import skimage as ski
# import skimage.io
import numpy as np
import pandas as pd

# import skimage.filters as ski.filters

# from skimage import io, filters
# from skimage.feature import canny


# TODO: This function is bloated too many tasks. Separate image preprocessing
#   from image filtering and transforms. No need to export image here.
#   Create export function for images and edges.
def HCTcv(
    img_path,
    Dmin,
    Dmax,
    dist2R=0.4,
    param1=200,
    param2=15,
    display_img=True,
    export_img=True,
    export_csv=True,
    eximgname=None,
    export_edges=True,
    edges_eximgname=None,
):
    """
    Function detects particles via Hough Transform and returns
    pandas.DataFrame with diameter and x, y coordinate for each
    detected particle.

    Parameters
    ----------
    img_path : absolute path to the image for particle detection


    Returns
    -------
    particles : pandas.DataFrame consisting of diameter and x, y coordinates
                of each detected particle

    """

    img_filename = os.path.basename(img_path)
    img_dir = os.path.dirname(img_path)

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]
    crop_ratio = 0.1
    work_image = image[0 : int((1 - crop_ratio) * h), :]  # cropping SEM table
    work_image = cv2.medianBlur(work_image, 3)  # smoothing noise (median blur)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # work_image = clahe.apply(work_image)

    circles = cv2.HoughCircles(
        work_image,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=dist2R * Dmax / 2,
        param1=param1,
        param2=param2,
        minRadius=int(Dmin / 2),
        maxRadius=int(Dmax / 2),
    )
    circlesDF = pd.DataFrame(circles[0], columns=["X", "Y", "R"])

    if display_img == True or export_img == True:
        imgC = draw_particles(work_image, circlesDF)
        if export_edges == True:
            imgE = cv2.Canny(
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
            if eximgname:
                imgCname = eximgname
            else:
                imgCname = os.path.splitext(img_filename)[0] + "_circles.png"
            cv2.imwrite(img_dir + os.sep + imgCname, imgC)
            if export_edges == True:
                if edges_eximgname:
                    imgEname = edges_eximgname
                else:
                    imgEname = os.path.splitext(img_filename)[0] + "_edges.png"
                cv2.imwrite(img_dir + os.sep + imgEname, imgE)
    if export_csv == True:
        circlesDF.rename(
            columns={
                "X": "X (pixels)",
                "Y": "Y (pixels)",
                "R": "Radius (pixels)",
            },
            inplace=True,
        )
        circlesDF.index.name = "ID"
        circlesDF.to_csv(
            img_dir + os.sep + os.path.splitext(img_filename)[0] + "_HCT.csv"
        )
    return circlesDF


def HCTcv_varR(
    img_path,
    Dspace,
    dist2R=0.4,
    param1=200,
    param2=15,
    export_img=True,
    export_csv=True,
):
    assert isinstance(Dspace, np.ndarray), (
        "Dspace must be numpy array," " for example np.linspace()"
    )

    img_filename = os.path.basename(img_path)
    img_dir = os.path.dirname(img_path)

    for i in range(len(Dspace) - 1):

        imname = (
            str(i + 1)
            + "_"
            + img_filename.split(".")[0]
            + "_range("
            + str(int(Dspace[i]))
            + "-"
            + str(int(Dspace[i + 1]))
            + ").png"
        )
        print(imname)
        buffDF = HCTcv(
            img_path,
            Dspace[i],
            Dspace[i + 1],
            dist2R,
            param1,
            param2,
            display_img=False,
            export_csv=False,
            export_img=export_img,
            eximgname=imname,
        )

        if i == 0:
            circlesDF = buffDF
        else:
            circlesDF = circlesDF.append(buffDF)

        print(circlesDF.shape)
    print(circlesDF)
    circlesDF = circlesDF.reset_index()
    print(circlesDF)
    circlesDF = circlesDF.drop_duplicates()
    print(circlesDF)

    if export_csv == True:
        circlesDF.rename(
            columns={
                "X": "X (pixels)",
                "Y": "Y (pixels)",
                "R": "Radius (pixels)",
            },
            inplace=True,
        )
        circlesDF.index.name = "ID"
        circlesDF.to_csv(
            img_dir + os.sep + os.path.splitext(img_filename)[0] + "_HCT.csv"
        )
    return circlesDF


def HCTcv_multi(
    img_path,
    Dspace,
    dist2R=0.4,
    param1=200,
    param2=15,
    export_img=True,
    export_csv=True,
    export_edges=True,
):

    # assert os.path.isfile(img_path), "File: " + img_path + " not found!"
    img_filename = os.path.basename(img_path)
    img_dir = os.path.dirname(img_path)

    Dspace = np.array(Dspace, dtype=np.float64)
    dist2R = np.array(dist2R, dtype=np.float64)
    param1 = np.array(param1, dtype=np.float64)
    param2 = np.array(param2, dtype=np.float64)

    # Convert all parameters to numpy.arrays and fill values if necessary
    # for further iteration

    if dist2R.shape != Dspace.shape:
        if dist2R.ndim == 1:
            if dist2R.shape[0] == Dspace.shape[0] - 1:
                dist2R = np.append(dist2R, dist2R[-1])
        dist2R = np.full_like(Dspace, dist2R)
    if param1.shape != Dspace.shape:
        if param1.ndim == 1:
            if param1.shape[0] == Dspace.shape[0] - 1:
                param1 = np.append(param1, param1[-1])
        param1 = np.full_like(Dspace, param1)
    if param2.shape != Dspace.shape:
        if param2.ndim == 1:
            if param2.shape[0] == Dspace.shape[0] - 1:
                param2 = np.append(param2, param2[-1])
        param2 = np.full_like(Dspace, param2)
    # print(Dspace, Dspace.dtype)
    # print(dist2R, dist2R.dtype)
    # print(param1, param1.dtype)
    # print(param2, param2.dtype)
    for i, (D, d2R, p1, p2) in enumerate(zip(Dspace, dist2R, param1, param2)):
        if i < len(Dspace) - 1:
            imname = (
                str(i + 1)
                + "_"
                + img_filename.split(".")[0]
                + "_D("
                + str(int(Dspace[i]))
                + "-"
                + str(int(Dspace[i + 1]))
                + ")_p1("
                + str(int(p1))
                + ")_p2("
                + str(int(p2))
                + ")"
            )
            cirname = imname + "_circles.png"
            edgename = imname + "_edges.png"
            buffDF = HCTcv(
                img_path,
                Dspace[i],
                Dspace[i + 1],
                d2R,
                param1=p1,
                param2=p2,
                display_img=False,
                export_csv=False,
                export_img=export_img,
                eximgname=cirname,
                export_edges=export_edges,
                edges_eximgname=edgename,
            )

            if i == 0:
                circlesDF = buffDF
            else:
                circlesDF = circlesDF.append(buffDF)
    if export_csv == True:
        circlesDF.rename(
            columns={
                "X": "X (pixels)",
                "Y": "Y (pixels)",
                "R": "Radius (pixels)",
            },
            inplace=True,
        )
        circlesDF.index.name = "ID"
        circlesDF.to_csv(
            img_dir + os.sep + os.path.splitext(img_filename)[0] + "_HCT.csv"
        )
    return circlesDF


def draw_particles(image, particles, color=(100, 255, 100)):

    alpha = 0.2
    work_image = image.copy()
    overlay = image.copy()

    for i, row in particles.iterrows():
        X = row.X
        Y = row.Y
        R = row.R
        cv2.circle(overlay, (int(X), int(Y)), int(R), color, -1)
        cv2.circle(work_image, (int(X), int(Y)), int(R), (255, 255, 255), 1)

    cv2.addWeighted(overlay, alpha, work_image, 1 - alpha, 0, work_image)

    return work_image


if __name__ == "__main__":

    path = (
        "C:\\DATA\\! IMP\\#Projekty Badawcze\\#HYBRYDA+\\wyniki badan\\"
        + "LAZISKA\\SEM\\analiza_aglomeratow\\1_spaliny\\ALL_OFF\\st15\\"
        + "st15-013\\st15-013.tif"
    )

    path2 = (
        "C:\\DATA\\! IMP\\Doktorat\\badania i obliczenia\\"
        + "analiza aglomeratow SEM\\agl_doz & podsysanie\\"
        + "1str\\AGL15kV\\AGL15-002\\AGL15-002.tif"
    )

    circs = HCTcv(
        path2,
        Dmin=3,
        Dmax=400,
        dist2R=0.4,
        param1=250,
        param2=20,
        display_img=False,
        export_edges=False,
        export_img=False,
        export_csv=False,
    )
    # display(circs)


# =============================================================================
#
#     path01 = "C:\\DATA\\! IMP\\#Projekty Badawcze\\#HYBRYDA+\\wyniki badan\\"\
#                     + "LAZISKA\\SEM\\analiza_aglomeratow\\1_spaliny\\ALL_OFF\\st15\\st15-013\\st15-013.tif"
#
#     print(os.path.abspath(path01))
#     # "C://DATA//! IMP//Doktorat//badania i obliczenia//analiza aglomeratow SEM//laziska//1//laz1.tif"
#     image = ski.io.imread(path01, as_gray=True)
#
#
#
#     print(image.shape)
#
#     image = ski.util.crop(image, ((0,160),(0,0)))
#     print(image.shape)
#
#     ski.io.imshow(image)
#
#     image_M = ski.filters.median(image)
#
#     e = []
#
#
#     e.append(image)
#     e.append(image_M)
#     # e.append(filters.sobel(image))
#     # e.append(filters.roberts(image))
#     # e.append(filters.scharr(image))
#     # e.append(filters.prewitt(image))
#     # e.append(canny(image, sigma=2, low_threshold=-.01, high_threshold=.05))
#     # e.append(canny(image, sigma=3, low_threshold=-.01, high_threshold=.05))
#     # e.append(canny(image, sigma=5, low_threshold=-.01, high_threshold=.05))
#     # e.append(ski.feature.canny(image, sigma=2, low_threshold=-.01, high_threshold=.05))
#     e.append(ski.feature.canny(image, sigma=2, low_threshold=-.01, high_threshold=.2))
#     e.append(ski.feature.canny(image_M, sigma=2, low_threshold=-.01, high_threshold=.2))
#     edge = e[1]
#
#     cv_image = ski.img_as_ubyte(image)
#     cv_image_HCT = ski.img_as_ubyte(edge)
#
#     tA = 200
#     tB = 100
#     Rmin = 3
#     Rmax = 40
#     dist2R = 0.4
#
#
#     cv_edge = cv2.Canny(cv_image_HCT, threshold1=tA, threshold2=0.5*tA)
#     cv2.imshow("edges", cv_edge)
#     # cv_edge2 = cv2.Canny(cv_image_HCT, threshold1=tB, threshold2=0.5*tB)
#     # cv2.imshow("edges2", cv_edge2)
#
#     circles = cv2.HoughCircles(cv_image_HCT, cv2.HOUGH_GRADIENT, dp=1, minDist=dist2R*Rmax, param1=tA, param2=12, minRadius=Rmin, maxRadius=Rmax)
#     # print(circles)
#     print(circles.shape)
#
#     circlesDF = pd.DataFrame(circles[0], columns = ["X", "Y", "R"])
#     # print(circlesDF.dtypes)
#
#     # ski.io.imshow(cv_image_HCT)
#
#     imgC = draw_particles(cv_image_HCT, circlesDF)
#     imgCname = "C://DATA/! IMP/Doktorat/badania i obliczenia/analiza aglomeratow SEM/laziska/1/laz1_CIRC2.png"
#     # cv2.imwrite(imgCname, imgC)
#
#     cv2.imshow("circles",imgC)
#     cv2.waitKey(0) # waits until a key is pressed
#     cv2.destroyAllWindows() # destroys the window showing image
#     # ski.io.imshow(imgC)
#
#     # fig, ax = filters.try_all_threshold(edges_sch, figsize=(50, 40), verbose=False)
#     # plt.show()
#
#
#     # =============================================================================
#     # for i, j in enumerate(e):
#     #     efile = "C://DATA/! IMP/Doktorat/badania i obliczenia/analiza aglomeratow SEM/laziska/1/laz1_edges_" + str(i) + ".png"
#     #     ski.io.imsave(efile, ski.img_as_uint(j))
#     # =============================================================================
#
#     # io.imsave("C://DATA/! IMP/Doktorat/badania i obliczenia/analiza aglomeratow SEM/laziska/1/laz1_edges_s.png", edges_s)
#     # io.imsave("C://DATA/! IMP/Doktorat/badania i obliczenia/analiza aglomeratow SEM/laziska/1/laz1_edges_r.png", edges_r)
#     # io.imsave("C://DATA/! IMP/Doktorat/badania i obliczenia/analiza aglomeratow SEM/laziska/1/laz1_edges_sch.png", edges_sch)
#     # io.imsave("C://DATA/! IMP/Doktorat/badania i obliczenia/analiza aglomeratow SEM/laziska/1/laz1_edges_p.png", edges_p)
#
#
# =============================================================================
