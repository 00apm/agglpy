# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:25:40 2020

@author: Artur
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import concurrent
import cv2  # type: ignore

# cv2 .pyi packaging problem as of 2024-10-08
# https://github.com/conda-forge/opencv-feedstock/issues/374
import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import constants  # type: ignore
from scipy import spatial as spsp
from tqdm import tqdm


from agglpy.aggl import Agglomerate, Particle
from agglpy.auxiliary import read_tiff_tags, RGB_convert_to256, RGB_shader
from agglpy.logger import logger
from agglpy.typing import ImageSettingsTypedDict, PPSouceCsvType
from agglpy.defaults import VALID_PARTICLE_CSV_DATA
from agglpy.errors import (
    DirectoryStructureError,
    ImgDataSetBufferError,
    ImgDataSetStructureError,
    ParticleCsvStructureError,
)


class ImgDataSet:
    """A class used for analyzis of primary particles and their agglomerates

    This class is used to obtain data about particles and its agglomerates
    on single image. It allows:
    - detection od primary particles
    - detection of their agglomerates
    - calculating primary particles statistical and morphological parameters
    - calculating agglomerate statistical and morphological parameters

    The ImgDataSet objects are constructed based on directory path (working_dir)
    and settings dict (which is a part of main .yaml settings file).
    The directory must contain an image to be analyzed.

    """

    # Public attributes
    name: str
    mag: float | None
    px_size: float | None

    # Public result attributes
    res_particleDF: pd.DataFrame | None
    res_agglDF: pd.DataFrame | None
    res_PSD: pd.DataFrame | None
    res_summary: pd.DataFrame | None

    # Private attributes
    _path: Path
    _settings: ImageSettingsTypedDict
    _img_filename: str | None
    _img_path: Path | None
    _img: npt.NDArray | None
    _img_meta_dict: dict | None

    _PPsource_filename: str | None
    _PPsource_path: Path | None
    _PPsource_type: PPSouceCsvType | None
    _PPsource_flag: bool  #

    _PPsource_bufferDF: pd.DataFrame | None
    _PP_dict: Dict[int, Particle] | None
    _all_PP_DF: pd.DataFrame | None
    _all_AGL_DF: pd.DataFrame | None
    _KDTree: spsp.KDTree | None

    def __init__(
        self,
        working_dir: os.PathLike,
        settings: ImageSettingsTypedDict,
        auto_load: bool = False,
    ):
        """ImgDataSet constructor

        Args:
            working_dir (os.PathLike): directory containing image and input
                data files for analyzis
            settings (ImageSettingsTypedDict): image settings dict, provided
                in settings.data.images in YAML settings
            auto_load (bool, optional): If true load primary particle data
                from .csv provided in settings. Defaults to False.

        """
        # Initialize attributes directly related to constructor arguments
        self._path = Path(working_dir)  # used for immutable @property path
        self._settings = settings

        # Initialize public attributes
        self.name = self._path.name
        self.mag = self._settings["magnification"]
        self.px_size = self._settings["pixel_size"]

        # Initialize public result DataFrames
        self.res_particleDF = None
        self.res_agglDF = None
        self.res_PSD = None
        self.res_summary = None

        # Initialize private attributes
        self._img_filename = self._settings["img_file"]
        self._img_path = self._path / self._img_filename
        self._img = None
        self._img_meta_dict = None
        self._PPsource_filename = self._settings["HCT_file"]
        self._PPsource_path = None  # HCT_filename may be None at this point
        self._PPsource_type = None
        self._PPsource_flag = False

        # Initialize internal data structure attributes
        self._PPsource_bufferDF = None
        self._PP_dict = None
        self._all_PP_DF = None
        self._all_AGL_DF = pd.DataFrame()
        # TODO: change _all_agl_DF creation. It should be seperated from
        # data structure containing Agglomerate objects
        self._KDTree = None

        # Check if image file exists
        if not self._img_path.exists():
            raise ImgDataSetStructureError(
                f"Image file {self._img_path} not found for ImgDataSet: "
                f"{str(self)}"
            )

        # Loading SEM image file and tags
        self._img = cv2.imread(str(self._img_path))
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB)
        if self._img_path.suffix == ".tif":
            self._img_meta_dict = read_tiff_tags(self._img_path)

            # If magnification and pixel size are not provided explicitly
            # attempt to retrive them from tiff tags
            if not self.mag:
                self.mag = self.retrieve_magnification()
            if not self.px_size:
                self.px_size = self.retrieve_pixel_size()
        else:
            if not self.mag:
                self.mag = 1.0
            if not self.px_size:
                self.px_size = 1.0

        logger.debug(f"{str(self)} Magnification set to: {self.mag:.1f}x.")
        logger.debug(f"{str(self)} Pixel size set to: {self.px_size:.4f}")
        # Define HCT related attributes and check if HCT .csv file exists
        if self._PPsource_filename:
            self._PPsource_path = self._path / self._PPsource_filename
            if self._PPsource_path.exists():
                self._PPsource_flag = True
            else:  # self._HCT_flag = False was initialized
                raise DirectoryStructureError(
                    f"HCT input file {self._PPsource_filename} declared in settings"
                    f" for ImgDataSet: {str(self)} was not found at: {self._PPsource_path}."
                )

        if auto_load:
            if self._PPsource_flag:
                self.load_PPsource_csv()
                self.create_primary_particles(multiprocessing=False)

    @property
    def path(self) -> Path:
        """Path to directory of Date Set

        This is read-only attribute setted at the ImgDataSet object construction

        Returns:
            Path: Working directory path
        """
        return self._path

    # ------------------ API methods --------------------------

    def classify_all_AGL(self, threshold=0):
        for i in self._all_AGL_DF.OBJ:
            i.classify(threshold)
        logger.debug(
            f"{str(self)} Agglomerates classified for threshold= {threshold}"
        )

    def load_PPsource_csv(self) -> None:
        """Load Hough Circle Transform data from input file

        Load and convert the data to the buffer with unified format.
        After loading particle data is scaled and sorted by size. Procedure:
        1. Load HCT data from input .csv file
        2. Recognize particle data type in .csv
        3. Convert DataFrame to unified format of internal buffer DF
        4. Store the DF in buffer internal attribute
        5. Scale HCT data according to pixel size
        6. Sort the DF descending by particle size
        """
        if not self._PPsource_path:
            raise ValueError("HCT csv path is not initialized.")
        self._PPsource_type = recognize_particle_csv(self._PPsource_path)
        if self._PPsource_type == "agglpy":
            self._PPsource_bufferDF = load_agglpy_csv(self._PPsource_path)
        elif self._PPsource_type == "ImageJ":
            self._PPsource_bufferDF = load_imagej_csv(self._PPsource_path)
        elif self._PPsource_type == "agglpy_old":
            self._PPsource_bufferDF = load_agglpy_old_csv(self._PPsource_path)
        else:
            raise ValueError(
                f"Primary particle csv data file type ({self._PPsource_type}) was"
                f" not recognized."
            )
        if not ("D" in self._PPsource_bufferDF.columns):
            self._PPsource_bufferDF.loc[:, "D"] = (
                2 * self._PPsource_bufferDF.loc[:, "R"]
            )
        self._PPsource_bufferDF = self._PPsource_bufferDF.astype(
            {
                "ID": int,
                "X": np.float64,
                "Y": np.float64,
                "D": np.float64,
                "R": np.float64,
            },
        )
        self._scale_HCT_data()

        # sort particles by size (descending) to assure proper order in
        # finding agglomerates
        self._PPsource_bufferDF = self._PPsource_bufferDF.sort_values(
            by=["R"],
            ascending=False,
        )
        logger.debug(
            f"{str(self)} Corrected Primary Particles data loaded from: "
            f"{self._PPsource_path}"
        )

    def create_primary_particles(self, multiprocessing: bool = False) -> None:
        if self._PPsource_bufferDF is None or self._PPsource_bufferDF.empty:
            raise ImgDataSetBufferError(
                f"Primary Parcicle source buffer is not initialized. Check if "
                f".csv data was loaded properly for ImgDataSet: {str(self)}."
            )
        if not self._PP_dict:
            if multiprocessing:
                logger.debug(
                    f"{str(self)} Creating particles with multiprocessing..."
                )
                self._PP_dict = self._create_PPobj_multiprocessing(
                    buffer=self._PPsource_bufferDF
                )
            else:
                logger.debug(
                    f"{str(self)} Creating particles in single thread..."
                )
                self._PP_dict = self._create_PPobj(
                    buffer=self._PPsource_bufferDF
                )
            self._create_PP_DF(PP_dict=self._PP_dict)
            if not (self._all_PP_DF is None or self._all_PP_DF.empty):
                logger.debug(
                    f"{str(self)} create_primary_particles() constructed: "
                    f"{len(self._all_PP_DF.index)} Primary Particles."
                )
        else:
            raise ImgDataSetStructureError(
                f"Primary Particles were already created for {str(self)}."
            )

    def find_agglomerates(self):
        self._find_all_intersecting()
        DF = self._all_PP_DF.copy()
        # TODO: _all_AGL_DF must be created first
        j = 0
        while not DF.empty:
            agl_obj = Agglomerate([])
            particle = DF.iloc[0, DF.columns.get_loc("OBJ")]
            self._all_AGL_DF.loc[j, "OBJ"] = agl_obj
            self._all_AGL_DF.loc[j, "ID"] = agl_obj.ID
            self._all_AGL_DF.loc[j, "name"] = agl_obj.name
            iFamily = self._find_intersecting_family(particle)
            agl_obj.append_particles(self.get_particles(iFamily))
            for i in iFamily:
                DF.drop(DF.loc[DF["ID"] == i].index, inplace=True)
            iFamily.clear()
            agl_obj.calc_member_param()
            agl_obj.calc_agl_param(include_dsom=True)

            # self._all_AGL_DF.loc[j,"members"] = str(self._all_AGL_DF.loc[j,"OBJ"].members)
            # self._all_AGL_DF.loc[j,"member count"] = self._all_AGL_DF.loc[j,"OBJ"].members_count
            j += 1

        # =============================================================================
        #         for i, row in self._all_AGL_DF.iterrows():
        #
        #             self._all_AGL_DF.loc[i,"OBJ"]._update_param()
        #             print(self._all_AGL_DF.loc[i,"OBJ"].members_count)
        # =============================================================================
        logger.debug(
            f"{str(self)} find_agglomerates() resulted in: "
            f"{len(self._all_AGL_DF.index)} Agglomerates detected."
        )
        return self._all_AGL_DF

    def get_particles(self, IDlist=[]):
        """
        Returns list of particle objects provided by ID list

        Parameters
        ----------
        IDlist : list of ints, optional
            Provide list of particle IDs. The default is [].

        Returns
        -------
        selected : list of agglpy.aggl.Particle objects
            List of particle objects

        """

        selected = []
        DF = self._all_PP_DF
        for i in IDlist:
            s = DF[DF["ID"] == i].OBJ.values[0]
            selected.append(s)
        return selected

    def get_largest_particle(self):
        return self._all_PP_DF.OBJ[self._all_PP_DF.D.idxmax()]

    def get_smallest_particle(self):
        return self._all_PP_DF.OBJ[self._all_PP_DF.D.idxmin()]

    # =============================================================================
    #     def get_PSD(self, start = 0, end = 10, periods = 20, log = False, cat=False):
    #
    #
    #         if log == True:
    #             bins_arr = np.logspace(int(np.log10(start)), int(np.log10(end)), periods)
    #         else:
    #             bins_arr = np.linspace(start, end, periods)
    #
    #         self.res_PSD = pd.cut(self._all_P_DF.D, bins_arr)
    #
    #         if cat == True:
    #             return self.res_PSD
    #         else:
    #             return pd.value_counts(self.res_PSD, sort = False)
    # =============================================================================

    def get_results_pTable(self):
        """
        Returns result table of all particles detected in the image

        Parameters
        ----------
        None

        Returns
        -------
        pTable : pd.DataFrame
            Table with information about particles identified in the image

        """
        # pTable = pd.DataFrame()
        # for i in self._all_P_DF.OBJ:
        #     pTable = pTable.append(i.get_properties())
        # pTable.reset_index(inplace = True, drop = True)
        # self.res_particleDF = pTable
        self.res_particleDF = self._all_PP_DF.loc[:, "OBJ"].apply(
            lambda p: pd.Series(p.get_properties())
        )
        self.res_particleDF = self.res_particleDF.astype({"ID": int})
        return self.res_particleDF

    def get_results_aglTable(self):
        # aglTable = pd.DataFrame()
        # for i in self._all_AGL_DF.OBJ:
        #     aglTable = aglTable.append(i.get_properties())
        # aglTable.reset_index(inplace=True, drop=True)
        # self.res_aglDF = aglTable
        self.res_agglDF = self._all_AGL_DF.loc[:, "OBJ"].apply(
            lambda a: pd.Series(a.get_properties())
        )
        return self.res_agglDF

    def get_summary(self):
        # self.res_summary = self.res_summary.assign(pd.Series(len(self._all_P_DF.index), name="particle_count"))
        # print(self.res_summary)
        if len(self.res_agglDF.index) == 0:
            self.get_results_aglTable()
        self.res_summary = pd.DataFrame()
        self.res_summary.loc[0, "N_primary_particle"] = len(
            self._all_PP_DF.index
        )
        self.res_summary.loc[0, "N_aerosol_particle"] = len(
            self._all_AGL_DF.index
        )
        pp1_mask = self.res_agglDF.loc[:, "members_count"] == 1
        self.res_summary.loc[0, "N_pp1"] = pp1_mask.sum()
        self.res_summary.loc[0, "N_ppA"] = (
            self.res_summary.loc[0, "N_primary_particle"]
            - self.res_summary.loc[0, "N_pp1"]
        )
        self.res_summary.loc[0, "N_agl"] = (
            self.res_summary.loc[0, "N_aerosol_particle"]
            - self.res_summary.loc[0, "N_pp1"]
        )

        self.res_summary.loc[0, "N_collector_agl"] = len(
            self.res_agglDF[self.res_agglDF["type"] == "collector"].index
        )
        self.res_summary.loc[0, "N_similar_agl"] = len(
            self.res_agglDF[self.res_agglDF["type"] == "similar"].index
        )
        self.res_summary.loc[0, "N_pp1_separate"] = len(
            self.res_agglDF[self.res_agglDF["type"] == "separate"].index
        )

        self.res_summary.loc[0, "ER"] = 1 - (
            self.res_summary.loc[0, "N_pp1_separate"]
            / self.res_summary.loc[0, "N_primary_particle"]
        )
        self.res_summary.loc[0, "Ra"] = (
            self.res_summary.loc[0, "N_agl"]
            / self.res_summary.loc[0, "N_primary_particle"]
        )
        self.res_summary.loc[0, "sep2agl"] = self.res_summary.loc[
            0, "N_pp1_separate"
        ] / (
            self.res_summary.loc[0, "N_collector_agl"]
            + self.res_summary.loc[0, "N_similar_agl"]
        )
        self.res_summary.loc[0, "n_ppA"] = (
            self.res_summary.loc[0, "N_ppA"] / self.res_summary.loc[0, "N_agl"]
        )
        self.res_summary.loc[0, "n_ppP"] = (
            self.res_summary.loc[0, "N_primary_particle"]
            / self.res_summary.loc[0, "N_aerosol_particle"]
        )
        self.res_summary.loc[0, "particle_Dmean"] = self.res_particleDF[
            "D"
        ].mean()
        self.res_summary.loc[0, "particle_Dstd"] = self.res_particleDF[
            "D"
        ].std()
        self.res_summary.loc[0, "particle_D10"] = self.res_particleDF[
            "D"
        ].quantile(q=0.1)
        self.res_summary.loc[0, "particle_D50"] = self.res_particleDF[
            "D"
        ].quantile(q=0.5)
        self.res_summary.loc[0, "particle_D90"] = self.res_particleDF[
            "D"
        ].quantile(q=0.9)
        # Sauter Mean Diameter
        self.res_summary.loc[0, "particle_SMD"] = (
            self.res_particleDF.loc[:, "D"] ** 3
        ).sum() / (self.res_particleDF.loc[:, "D"] ** 2).sum()
        self.res_summary.loc[0, "agl_member_count_mean"] = self.res_agglDF[
            "members_count"
        ].mean()
        self.res_summary.loc[0, "agl_member_count_std"] = self.res_agglDF[
            "members_count"
        ].std()

        # print(self.res_summary.dtypes)

        return self.res_summary

    def get_img(self):
        return self._img

    def get_img_filename(self):
        return self._img_filename

    def retrieve_magnification(self) -> float:
        if not self._img_meta_dict:
            raise ImgDataSetStructureError(
                f"._img_meta_dict is empty for ImgDataSet {str(self)}. "
                f"Check if tiff tags are loaded properly."
            )
        logger.debug(
            f"{str(self)} Attempting to find magnification in "
            f"{self.get_img_filename()} tags."
        )
        lmag = self._img_meta_dict["CZ_SEM"]["ap_mag"][1].split()
        if "K" in lmag:
            magn = float(lmag[0]) * 1000
        else:
            magn = float(lmag[0])
        return magn

    def retrieve_pixel_size(self) -> float:
        assert (
            self._img_meta_dict
        ), "._img_meta_dict is empty. Check if tiff tags are loaded properly."
        logger.debug(
            f"{str(self)} Attempting to find pixel size in "
            f"{self.get_img_filename()} tags."
        )
        keys = ["ap_image_pixel_size", "ap_pixel_size"]
        px_size = None
        for k in keys:
            try:
                px_size = self._img_meta_dict["CZ_SEM"][k][1]
                px_size_unit = self._img_meta_dict["CZ_SEM"][k][2]
                break
            except KeyError:
                pass
            except:
                raise
        if px_size is None:
            px_size = self._img_meta_dict["CZ_SEM"][""][3]
            px_size_unit = "m"
            warnings.warn(
                "SEM image have old format. Retrieving pixel size from .tiff exif"
                " asuming meters as pixel size unit",
            )
        if px_size_unit == "nm":
            px_size = px_size * 1e-9
        elif px_size_unit == "µm":
            px_size = px_size * 1e-6
        elif px_size_unit == "m":
            pass
        else:
            raise ValueError(
                "Pixel size unit in tif exif SEM metadata not recognized."
            )
        return px_size

    def plot_img(
        self,
        img,
        show=True,
        export=True,
        export_dpi=300,
        dirpath: Optional[Path] = None,
        bar=False,
        bar_data=None,
        bar_discrete=False,
    ):

        fig = plt.figure(figsize=(14, 9), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(img)

        if bar:
            if bar_data is not None:
                prop = bar_data[0]
                prop_key = list(prop.keys())[0]
                prop_label = list(prop.values())[0]
                cmap = bar_data[1]
                norm = bar_data[2]
                ticks = None
                if bar_discrete:
                    vmin = norm.vmin
                    vmax = norm.vmax

                    bounds = np.linspace(
                        vmin, vmax + 1, int(vmax - vmin + 2), endpoint=True
                    )
                    bounds_moved = bounds - 0.5
                    bnorm = mpl.colors.BoundaryNorm(
                        boundaries=bounds_moved, ncolors=256
                    )
                    norm = bnorm
                    ticks = bounds

                cbax = fig.add_axes(
                    [0.9, 0.25, 0.015, 0.6]
                )  # [0.87, 0.25, 0.015, 0.6])
                cb1 = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cbax,
                    extend="max",
                    ticks=ticks,
                )
                cb1.set_label(str(prop_label), fontsize=14)
                cb1.ax.tick_params(labelsize=14)

            else:
                raise ValueError(
                    "please provide bar_data to properly plot"
                    " color bar into image."
                )

        if show == True:
            fig.show()

        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        ax.tick_params(
            axis="y", which="both", right=False, left=False, labelleft=False
        )

        if export == True:
            fname = self.name + "_circles.png"
            if dirpath is not None:
                fpth = dirpath / fname
            else:
                fpth = self._path / fname
            fig.savefig(
                fpth,
                facecolor="w",
                edgecolor="w",
                dpi=export_dpi,
                bbox_inches="tight",
                pad_inches=0,
            )
            print("IMAGE EXPORTED TO: ", fpth)
        return fig

    # =============================================================================
    #     def draw_particle(self, particle, color = (100, 255, 100, 0)):
    #         shape = self._img.shape
    #         im1 = np.zeros((shape[0],shape[1],4), dtype = np.uint8)
    #         im2 = np.zeros((shape[0],shape[1],4), dtype = np.uint8)
    #
    #         factor = 0.4
    #         color_shaded = RGB_shader(color, factor)
    #
    #         X = particle.X / self.scf
    #         Y = particle.Y / self.scf
    #         D = particle.D / self.scf
    #         cv2.circle(im1, (int(X), int(Y)), int(D/2), color, -1)
    #         cv2.circle(im2, (int(X), int(Y)), int(D/2), color_shaded, 1)
    #
    #         return im1, im2
    # =============================================================================

    def draw_particles_transp(
        self,
        particles,
        color=(100, 255, 100, 0),
        transparency=0,
        centers=False,
        shade_factor=0.4,
        thickness=1,
    ):
        shape = self._img.shape
        im1 = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        im2 = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)

        if isinstance(particles, Particle):
            particles = [particles]

        if isinstance(color, tuple):
            color = np.full((len(particles), 4), color)

        assert isinstance(color, np.ndarray), (
            "color array have wrong"
            " data type. Color must be represented as tuple of length 4"
            " or ndarray of shape (len(particles), 4) meaning single RGBA"
            " color for each particle."
        )

        assert color.shape == (len(particles), 4), (
            "color array have wrong"
            " data type. Color must be represented as tuple of length 4"
            " or ndarray of shape (len(particles), 4) meaning single RGBA"
            " color for each particle."
        )
        color[:, 3] = color[:, 3] * (1 - transparency)

        color_shaded = RGB_shader(color, shade_factor)

        cross_size = 6

        for i, p in enumerate(particles):
            X = p.X / self.px_size
            Y = p.Y / self.px_size
            D = p.D / self.px_size
            cv2.circle(
                im1, (int(X), int(Y)), int(D / 2), color[i].tolist(), -1
            )
            cv2.circle(
                im2,
                (int(X), int(Y)),
                int(D / 2),
                color_shaded[i].tolist(),
                thickness,
            )
            if centers:
                # centers vertical lines
                cv2.line(
                    im2,
                    (int(X), int(Y - cross_size / 2)),
                    (int(X), int(Y + cross_size / 2)),
                    color[i].tolist(),
                    thickness,
                )
                # centers horizontal lines
                cv2.line(
                    im2,
                    (int(X - cross_size / 2), int(Y)),
                    (int(X + cross_size / 2), int(Y)),
                    color[i].tolist(),
                    thickness,
                )
        # res = self._img
        # cnd = im1[:, :, 2] > 0
        # res[cnd] = im1[cnd]
        return im1, im2

    def draw_agl(self, AGL, color=(100, 255, 100, 0), labels=False):
        # for i in AGL.get_members():

        memTable = pd.DataFrame()
        for i, P in enumerate(AGL.get_members()):
            memTable.loc[i, "X"] = P.X / self.px_size
            memTable.loc[i, "Y"] = P.Y / self.px_size
        memTable.reset_index(inplace=True, drop=True)
        MAXs = memTable.max()
        MINs = memTable.min()

        im1, im2 = self.draw_particles_transp(AGL.get_members(), color=color)

        if labels == True:

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            fontColor = color
            thickness = 2

            bbox = cv2.getTextSize(
                AGL.name, fontFace=font, fontScale=scale, thickness=thickness
            )

            bboxW = bbox[0][0]
            bboxH = bbox[0][1]

            # locX = int((MINs.X + 1*(MAXs.X-MINs.X)/2 ))
            # locY = int((MINs.Y + 1*(MAXs.Y-MINs.Y)/2 ))
            # locX = int(MINs.X)
            # locY = int(MINs.Y)
            locX = int((AGL.get_members()[0].X) / self.px_size - bboxW / 2)
            locY = int((AGL.get_members()[0].Y) / self.px_size)
            loc = (locX, locY)

            cv2.putText(
                im2,
                AGL.name,
                loc,
                fontFace=font,
                fontScale=scale,
                color=fontColor,
                thickness=thickness,
            )

        return im1, im2

    # def draw_agl_label(self, AGL, color = (100, 255, 100, 0)):

    def draw_all_agl(self, labels=False, transparency=0.1):

        # --- COLORMAP

        colors = {
            "collector": 255
            * mpl.colors.to_rgba_array(mpl.colors.CSS4_COLORS["deepskyblue"])[
                0
            ],
            "similar": 255
            * mpl.colors.to_rgba_array(mpl.colors.CSS4_COLORS["red"])[0],
            "separate": 255
            * mpl.colors.to_rgba_array(mpl.colors.CSS4_COLORS["lime"])[0],
        }
        for i in colors.values():
            i = i.astype(np.uint8)

        img = self._img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        shape = self._img.shape

        res1 = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        res2 = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        for i in self._all_AGL_DF.OBJ:
            im1, im2 = self.draw_agl(i, color=colors[i.type], labels=labels)
            cnd1 = im1[:, :, 3] > 0
            res1[cnd1] = im1[cnd1]
            cnd2 = im2[:, :, 3] > 0
            res2[cnd2] = im2[cnd2]

        # fig, axs = plt.subplots(2,1)
        # axs[0].imshow(res1)
        # axs[1].imshow(res2)
        # plt.show()
        alpha = 1 - transparency
        cv2.addWeighted(res1, alpha, img, 1, 0, img)
        CND = res2[:, :, 3] > 0
        img[CND] = res2[CND]

        return img

    def draw_all_agl_by(
        self,
        prop,
        vmin=None,
        vmax=None,
        labels=False,
        transparency=0.1,
        cmap=mpl.cm.plasma,
        ret_cbar_data=False,
        bg_white=False,
    ):

        # --- COLORMAP
        if isinstance(prop, dict):
            assert len(prop) == 1, (
                "prop dict is too long," " choose single property to draw"
            )
            # label = list(prop.items())[0]
            prop_key = list(prop.keys())[0]
        else:
            assert isinstance(prop, str), (
                "wrong data input- only string and"
                " dict of strings are accepted."
            )
            prop_key = "ID"
        if (vmin is None) or (vmax is None):
            if self.res_agglDF.empty:
                self.get_results_aglTable()
            vmin = self.res_agglDF.loc[:, prop_key].min()
            vmax = self.res_agglDF.loc[:, prop_key].max()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        img = self._img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        shape = self._img.shape

        res1 = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        res2 = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        for i in self._all_AGL_DF.OBJ:
            val = i.get_properties()[prop_key]
            color = RGB_convert_to256(cmap(norm(val)))
            im1, im2 = self.draw_agl(i, color=color, labels=labels)
            cnd1 = im1[:, :, 3] > 0
            res1[cnd1] = im1[cnd1]
            cnd2 = im2[:, :, 3] > 0
            res2[cnd2] = im2[cnd2]
        mask1 = res1[:, :, 3] > 0
        # fig, axs = plt.subplots(2,1)
        # axs[0].imshow(res1)
        # axs[1].imshow(res2)
        # plt.show()
        alpha = 1 - transparency
        overlay = cv2.addWeighted(res1, alpha, img, 1 - alpha, 0)
        img[mask1] = overlay[mask1]
        mask2 = res2[:, :, 3] > 0
        img[mask2] = res2[mask2]
        if ret_cbar_data:
            bar_data = (prop, cmap, norm)
            if bg_white:
                return (res1, bar_data)
            else:
                return (img, bar_data)
        else:
            if bg_white:
                return img
            else:
                return res1

    def draw_all_particles_by(
        self,
        prop=None,
        vmin=None,
        vmax=None,
        labels=False,
        transparency=0.1,
        thickness=1,
        centers=True,
        cmap=mpl.cm.plasma,
        ret_cbar_data=False,
        bg_white=True,
    ):
        # --- COLORMAP
        if isinstance(prop, dict):
            assert len(prop) == 1, (
                "prop dict is too long," " choose single property to draw"
            )
            # label = list(prop.items())[0]
            prop_key = list(prop.keys())[0]
        else:
            assert isinstance(prop, str), (
                "wrong data input- only string and"
                " dict of strings are accepted."
            )
        if (vmin is None) or (vmax is None):
            if self.res_particleDF.empty:
                self.get_results_pTable()
            vmin = self.res_particleDF.loc[:, prop_key].min()
            vmax = self.res_particleDF.loc[:, prop_key].max()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        img = self._img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        shape = self._img.shape

        res1 = np.full(
            (shape[0], shape[1], 4),
            np.array([255, 255, 255, 255]),
            dtype=np.uint8,
        )

        colors = np.zeros([len(self.res_particleDF), 4])
        for i, p in enumerate(self._all_PP_DF.OBJ):
            val = p.get_properties()[prop_key]
            colors[i] = np.array(RGB_convert_to256(cmap(norm(val))))

        im1, im2 = self.draw_particles_transp(
            self._all_PP_DF.OBJ,
            color=colors,
            transparency=0,
            centers=True,
            shade_factor=0,
            thickness=thickness,
        )

        mask1 = im2[:, :, 3] > 0
        img[mask1] = im2[mask1]
        res1[mask1] = im2[mask1]
        res1[:, :, 3] = 255
        # fig, axs = plt.subplots(2,1)
        # axs[0].imshow(im1)
        # axs[1].imshow(im2)
        # plt.show()

        # alpha = 1-transparency
        # overlay = cv2.addWeighted(im2, alpha, img, 1-alpha, 0)
        # img[mask1] = overlay[mask1]
        # mask2 = res2[:, :, 3] > 0
        # img[mask2] = res2[mask2]
        if ret_cbar_data:
            bar_data = (prop, cmap, norm)
            if bg_white:
                return (res1, bar_data)
            else:
                return (img, bar_data)
        else:
            if bg_white:
                return res1
            else:
                return img

    def _scale_HCT_data(self) -> None:
        # applying scale to R (converting to µm)
        if self._PPsource_bufferDF is None or self._PPsource_bufferDF.empty:
            raise ImgDataSetBufferError(
                f"_HCT_bufferDF is not initialized or is empty. Probably .csv "
                f"data was not loaded properly"
            )
        if not self.px_size:
            raise ImgDataSetStructureError(
                f"Pixel size is None. Check settings of image file for "
                f"DataSet {str(self)}"
            )
        self._PPsource_bufferDF.loc[:, ["X", "Y", "R", "D"]] = (
            self._PPsource_bufferDF.loc[:, ["X", "Y", "R", "D"]] * self.px_size
        )

    def _create_PPobj_from_row(
        self,
        row: pd.Series,
    ) -> Tuple[int, Particle]:
        """Create Particle object from row of pd.DataFrame
        Args:
            row (pd.Series): single row from DataFrame containing
                primary particle coordinate and diameter data
        Returns:
            Tuple[int, Particle]: Tuple containg Particle ID nad Particle object
        """
        particle = Particle(row["ID"], row["X"], row["Y"], row["D"])
        return row["ID"], particle

    def _create_PPobj(
        self,
        buffer: pd.DataFrame,
    ) -> Dict[int, "Particle"]:
        if buffer.empty:
            raise ImgDataSetBufferError(
                f"Primary Parcicle source buffer is empty. Check if .csv data "
                f"was loaded properly for ImgDataSet: {str(self)}."
            )
        particles_dict: Dict[int, "Particle"] = {}
        for _, row in buffer.iterrows():
            ID, particle = self._create_PPobj_from_row(row=row)
            particles_dict[ID] = particle
        return particles_dict

    def _create_PPobj_multiprocessing(
        self,
        buffer: pd.DataFrame,
    ) -> Dict[int, Particle]:
        if buffer.empty:
            raise ImgDataSetBufferError(
                f"Primary Parcicle source buffer is empty. Check if .csv data "
                f"was loaded properly for ImgDataSet: {str(self)}."
            )
        particles_dict: Dict[int, "Particle"] = {}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self._create_PPobj_from_row, row): row["ID"]
                for _, row in buffer.iterrows()
            }

            for future in concurrent.futures.as_completed(futures):
                id, particle = future.result()
                particles_dict[id] = particle
        return particles_dict

    def _create_PP_DF(self, PP_dict: Dict[int, Particle]) -> None:
        data = {
            "ID": list(PP_dict.keys()),
            "X": [particle.X for particle in PP_dict.values()],
            "Y": [particle.Y for particle in PP_dict.values()],
            "D": [particle.D for particle in PP_dict.values()],
            # TODO: Deprecate OBJ column when find agglomerate functions are based on PP_dict
            "OBJ": list(PP_dict.values()),
        }
        self._all_PP_DF = pd.DataFrame(data)
        self._all_PP_DF = self._all_PP_DF.astype(
            {
                "ID": int,
                "X": np.float64,
                "Y": np.float64,
                "D": np.float64,
            },
        )

    def _construct_KDTree(self) -> None:
        if self._all_PP_DF is None or self._all_PP_DF.empty:
            raise ImgDataSetStructureError(
                f"Primary Particle DataFrame was not properly created after "
                f" Particle objects creation for ImgDataSet: {str(self)}"
            )
        self._KDTree = spsp.KDTree(self._all_PP_DF.loc[:, ["X", "Y"]])
        return self._KDTree

    def _find_all_intersecting(self) -> None:
        if not self._KDTree:
            logger.debug(
                f"KDTree structure was not created for {str(self)}. "
                f"Initializing KDTree ..."
            )
            self._construct_KDTree()
        if self._all_PP_DF is None or self._all_PP_DF.empty:
            raise ImgDataSetStructureError(
                f"Primary Particle DataFrame was not properly created after "
                f" Particle objects creation for ImgDataSet: {str(self)}"
            )
        KD: spsp.KDTree = self._KDTree
        P_workDF = self._all_PP_DF.loc[:, "OBJ"]
        D_MAX = self.get_largest_particle().D

        for P in P_workDF:
            nbrs = KD.query_ball_point([P.X, P.Y], D_MAX)

            intersecting = []
            for i in nbrs:
                P_i: Particle = P_workDF[i]
                if P.ID == P_i.ID:
                    continue
                # dist = 0
                # contact = 0
                R_p = P.D / 2
                R_pi = P_i.D / 2
                dist = ((P_i.X - P.X) ** 2 + (P_i.Y - P.Y) ** 2) ** 0.5
                if dist <= (R_p + R_pi):
                    intersecting.append(P_i.ID)
                    if dist <= (R_p - R_pi):
                        P_i.set_idj(True)
            P.set_interIDs(intersecting)

    def _find_intersecting(self, particle, d):
        try:
            self._KDTree
        except AttributeError:
            print(
                "Program hasn't found KDTree structure. Initializing KDTree ..."
            )
            self._construct_KDTree()
        KD = self._KDTree
        nbrs = KD.query_ball_point([particle.X, particle.Y], d)
        intersecting = []
        for i in nbrs:
            dist = (
                (self._all_PP_DF.OBJ[i].X - particle.X) ** 2
                + (self._all_PP_DF.OBJ[i].Y - particle.Y) ** 2
            ) ** 0.5
            contact = particle.D / 2 + self._all_PP_DF.OBJ[i].D / 2

            if dist <= contact:
                intersecting.append(self._all_PP_DF.OBJ[i].ID)
        return intersecting

    def _find_intersecting_family(self, particle, list_=[]):

        if particle.ID not in list_:
            list_.append(particle.ID)
            children = particle.interIDs
            for i in children:
                if i not in list_:
                    self._find_intersecting_family(
                        self.get_particles([i])[0], list_
                    )

        return list_

    # -------------- dunder methods ---------------------------
    def __repr__(self):
        class_name = type(self).__name__
        repr_str = (
            f"{class_name}(working_dir={self._path!r}, "
            f"settings= {self._settings!r})"
        )
        return repr_str

    def __str__(self):
        return f"<DS({self.name})>"


def recognize_particle_csv(
    filepath: os.PathLike, full: bool = False
) -> PPSouceCsvType:
    fpath: Path = Path(filepath)
    if not full:
        nrows = 5
    recognized_types: List[PPSouceCsvType] = []
    df: pd.DataFrame = pd.read_csv(fpath, nrows=nrows)
    for csvtype, cols in VALID_PARTICLE_CSV_DATA.items():
        if set(cols).issubset(df.columns):
            recognized_types.append(csvtype)
    if not recognized_types:
        raise ParticleCsvStructureError(
            f"Data structure in primary particle data .csv file: {fpath.name} "
            f"not recognized."
        )
    elif len(recognized_types) > 1:
        # Use default agglpy csv type if more then one structure is recognized
        if "agglpy" in recognized_types:
            return "agglpy"
        else:
            raise ParticleCsvStructureError(
                f"Multiple data structures recognized in primary particle data "
                f".csv file: {fpath.name}, and none of it is of 'agglpy' "
                f"default type."
            )
    else:
        return recognized_types[0]


def load_agglpy_csv(path: os.PathLike) -> pd.DataFrame:
    """Loads .csv primary particle data in agglpy format to the DataFrame

    Args:
        path (os.PathLike): Path to .csv primary particle data file

    Returns:
        pd.DataFrame: DataFrame containing primary particle coordinates and radius
    """
    fpath: Path = Path(path)
    df: pd.DataFrame = pd.read_csv(path, sep=",", encoding="ansi")
    valid_columns: Tuple[str, ...] = VALID_PARTICLE_CSV_DATA["agglpy"]
    assert set(valid_columns).issubset(df.columns), (
        f"Some of columns valid for agglpy default csv ({str(valid_columns)}) "
        f"were not found in the csv file: {fpath}"
    )
    return df


def load_agglpy_old_csv(path: os.PathLike) -> pd.DataFrame:
    """Loads .csv primary particle data in agglpy format to the DataFrame

    Args:
        path (os.PathLike): Path to .csv primary particle data file

    Returns:
        pd.DataFrame: DataFrame containing primary particle coordinates and radius
    """
    fpath: Path = Path(path)
    df: pd.DataFrame = pd.read_csv(path, sep=",", encoding="ansi")
    valid_columns: Tuple[str, ...] = VALID_PARTICLE_CSV_DATA["agglpy_old"]
    assert set(valid_columns).issubset(df.columns), (
        f"Some of columns valid for agglpy old format csv ({str(valid_columns)}) "
        f"were not found in the csv file: {fpath}"
    )
    cols_rename = {
        "X (pixels)": "X",
        "Y (pixels)": "Y",
        "Radius (pixels)": "R",
    }
    df.rename(columns=cols_rename, inplace=True)
    return df


def load_imagej_csv(path: os.PathLike) -> pd.DataFrame:
    """Loads .csv primary particle data in ImageJ format to the DataFrame

    Args:
        path (os.PathLike): Path to ImageJ type .csv primary particle data file

    Returns:
        pd.DataFrame: DataFrame containing primary particle coordinates and radius
    """
    fpath: Path = Path(path)
    df: pd.DataFrame = pd.read_csv(path, sep=",", encoding="ansi")
    valid_columns: Tuple[str, ...] = VALID_PARTICLE_CSV_DATA["ImageJ"]
    valid_agglpy_columns: Tuple[str, ...] = VALID_PARTICLE_CSV_DATA["agglpy"]
    assert set(valid_columns).issubset(df.columns), (
        f"Some of columns valid for ImageJ csv ({str(valid_columns)}) were not "
        f"found in the csv file: {fpath}"
    )

    # Select all particles with ImageJ "Oval" type
    map_oval = df.loc[:, "Type"] == "Oval"
    # Select all particles with equal width and height (eliminate ellipses)
    map_wh_eq = df.loc[:, "Width"] == df.loc[:, "Height"]
    sum_non_spherical = ~map_oval.sum() + ~map_wh_eq.sum()

    if sum_non_spherical > 0:
        dropped_IDs: list = df.loc[~(map_oval & map_wh_eq), "Index"].to_list()
        logger.warning(
            f"Non spherical particles were detected in ImageJ csv file: {fpath} "
            f"Dropped {sum_non_spherical} particles, "
            f"their ROI IDs were: {dropped_IDs}"
        )

    # Drop non spherical particles
    df = df.loc[map_oval & map_wh_eq, :]

    # Drop unnecessary columns
    df = df.loc[:, list(valid_columns)]
    df.drop(["Type", "Height"], axis="columns", inplace=True)
    df.rename(columns={"Index": "ID", "Width": "D"}, inplace=True)
    df.loc[:, "R"] = df.loc[:, "D"] / 2

    # correcting center coordinates (anchor point of oval in ImageJ is defined
    # in upper-left corner)
    df.loc[:, "X"] = df.loc[:, "X"] + df.loc[:, "R"]
    df.loc[:, "Y"] = df.loc[:, "Y"] + df.loc[:, "R"]

    return df
