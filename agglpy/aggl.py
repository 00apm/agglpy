# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:25:40 2020

@author: Artur
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fluids import particle_size_distribution as fPSD
from scipy import constants
from scipy import spatial as spsp
from tqdm import tqdm

from agglpy.auxiliary import RGB_convert_to256, RGB_shader


class ImgAgl:
    """
    This class allows analyzis of Particle Matter (PM) data from SEM micrographs.
    Allows measurement of basic properties of particles (its count on single
    micrograph, its coordinates, diameter). Enables agglomerates analysis-
    identifies agglomerates based on the properties of individual particles.
    Provides summarized data exportable data.

    Parameters
    ----------
    working_folder : string
        path to the folder containing SEM photo and .csv particle data file.



    """

    def __init__(
        self,
        working_folder,
        settings,
    ):
        self._path: Path = working_folder  # os.path.abspath(file)
        self.name = os.path.basename(self._path)

        self._settings = settings
        self._img_filename = self._settings["img_file"]
        self._img_path = self._path / self._img_filename
        # TO DO: change _fitRes_filename to corrected
        self._fitRes_filename = self._settings["correction_file"]
        self._corrected_path = self._path / self._fitRes_filename

        # Checking if files declared in settings exists
        assert (
            self._img_path.exists()
        ), f"Image file {self._img_path} not found"
        assert (
            self._corrected_path.exists()
        ), f"Corrected fitting file {self._corrected_path} not found"

        # Loading SEM image file
        self._img = cv2.imread(str(self._img_path))
        self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB)

        # Loading circle fitting file
        self._fitRes = pd.read_csv(
            self._corrected_path,
            sep=",",
            encoding="ansi",
        )

        self.mag = self.set_magnification()

        # Setting scale factor (pixel size in um)
        width_const = 300000  # image width (in um) for magnification 1X
        # and constant image pixel width

        if self.mag != None:
            self.scf = width_const / (self.mag * self._img.shape[1])
        else:
            self.scf = 1

        # Initializing internal attributes

        self._initDF = self._init_p_coor_data()
        self._all_P_DF = pd.DataFrame()
        self._all_AGL_DF = pd.DataFrame()
        self._KDTree = None

        # Initializing results attributes
        self.res_particleDF = pd.DataFrame()
        self.res_aglDF = pd.DataFrame()
        self.res_PSD = pd.DataFrame()
        self.res_summary = pd.DataFrame()

        # Running constructor methods
        self.find_particles()

    class Agglomerate:
        """
        Agglomerate class creates agglomerate objects and defines its properties

        """

        AGL_ID = 1

        def __init__(self, member_list=[]):
            self.ID = ImgAgl.Agglomerate.AGL_ID
            self.members = member_list
            self.member_IDs = []
            self.members_count = 0
            self.name = "AGL" + str(ImgAgl.Agglomerate.AGL_ID)
            self._members_DF = pd.DataFrame()
            self.type = "not_defined"
            self.Xc = None
            self.Yc = None

            ImgAgl.Agglomerate.AGL_ID = ImgAgl.Agglomerate.AGL_ID + 1

        def append_particles(self, particle_list=[]):
            """
            ImgAgl.Agglomerate method for appending particle objects
            to agglomerate members container

            Parameters
            ----------
            particle_list : list of ImgAgl.Particle objects, optional
                Provide list of particle objects (ImgAgl.Particle instances)
                to append to Agglomerate members container. The default is [].

            Returns
            -------
            None.

            """
            for i in particle_list:
                # print(i, type(i))
                if i not in self.members:
                    self.members.append(i)

        def get_members(self):
            return self.members

        def get_members_IDs(self):
            IDs = []
            for p in self.members:
                IDs.append(p.ID)
            return IDs

        def get_members_Ds(self):
            Ds = []
            for p in self.members:
                Ds.append(p.D)
            return Ds

        def get_member_byID(self, ID):
            IDs = self.get_members_IDs()
            assert ID in IDs, "Given ID is not a member of this Agglomerate"
            ix = IDs.index(ID)
            return self.members[ix]

        def get_properties(self, form="dict"):
            dict_prop = {
                "ID": self.ID,
                "name": self.name,
                "volume": self.volume,
                "D": self.D,
                "type": self.type,
                "members_count": self.members_count,
                "members_Dmean": self.members_Dmean,
                "members_Dstdev": self.members_Dstdev,
            }
            if form == "dict":
                return dict_prop
            elif form == "series":
                return pd.Series(dict_prop)
            elif form == "df":
                return pd.DataFrame.from_dict(
                    dict_prop,
                    orient="index",
                ).T
            else:
                raise ValueError("Unknown form parameter: " + str(form))

        def clasify(self, threshold=0):
            # self._calc_member_param()

            type_sw_AGL = {0: "collector", 1: "separate", 2: "similar"}

            type_sw_P = {
                0: "collector",
                1: "attached2coll",
                2: "separate",
                3: "similar",
            }

            for i in self.members:
                i.set_affiliation(self)

            if self.members_count > 1:
                if (
                    self._members_DF.iloc[1].D / self._members_DF.iloc[0].D
                    <= threshold
                ):
                    self.type = type_sw_AGL.get(0)
                    for i in range(len(self.members)):
                        if i == 0:
                            self._members_DF.iloc[i].membersOBJ.type = (
                                type_sw_P.get(0)
                            )
                        else:
                            self._members_DF.iloc[i].membersOBJ.type = (
                                type_sw_P.get(1)
                            )
                else:
                    self.type = type_sw_AGL.get(2)
                    for i in range(len(self.members)):
                        self._members_DF.iloc[i].membersOBJ.type = (
                            type_sw_P.get(3)
                        )

            elif self.members_count == 1:
                self.type = type_sw_AGL.get(1)
                self._members_DF.iloc[0].membersOBJ.type = type_sw_P.get(2)

        def _calc_agl_param(self):
            # print(self.members)
            for i in self.members:
                self.member_IDs.append(i.ID)
            self._members_DF["membersOBJ"] = self.get_members()
            self.members_count = len(self.members)
            Ds = self._members_DF.loc[
                :, "D"
            ]  # pd.Series(self.get_members_Ds())
            # Vols = 4/3*constants.pi*(Ds/2)**3
            self.members_Dmean = Ds.mean()
            self.members_Dstdev = Ds.std()
            self.volume = self._members_DF["Vol"].sum()  # Vols.sum()
            # self.D = (self.volume)**(1/3) / (constants.pi/8)
            self.D = (6 * self.volume / constants.pi) ** (1 / 3)

            #

        def _calc_member_param(self):
            self._members_DF["membersOBJ"] = self.get_members()
            self._members_DF["ID"] = self.get_members_IDs()
            self._members_DF["D"] = self.get_members_Ds()
            self._members_DF["Vol"] = (
                4 / 3 * constants.pi * (self._members_DF["D"] / 2) ** 3
            )
            self._members_DF["D-ratios"] = (
                self._members_DF["D"] / self._members_DF["D"].max()
            )
            self._members_DF.sort_values(
                by=["D"], ascending=False, inplace=True
            )
            # print(self._members_DF)

        def __repr__(self):
            return self.name

    class Particle:

        def __init__(self, ID, X, Y, D):

            self.ID = ID
            self.name = "P" + str(int(self.ID))
            self.X = X
            self.Y = Y
            self.D = D
            self.Vol = 4 / 3 * constants.pi * ((self.D) / 2) ** 3
            self.interIDs = None
            self.type = "not_defined"
            self.affil = "not_defined"

        def __repr__(self):
            return self.name

        def get_coord(self):
            return (self.X, self.Y)

        def get_D(self):
            return self.D

        def get_type(self):
            return self.type

        def get_affiliation(self):
            return self.affil

        def get_interIDs(self):
            """


            Returns
            -------



            """
            return self.interIDs

        def get_properties(self, form="dict"):
            dict_prop = {
                "ID": self.ID,
                "name": self.name,
                "D": self.D,
                "X": self.X,
                "Y": self.Y,
                "Vol": self.Vol,
                "type": self.type,
                "affiliation": self.affil,
            }
            if form == "dict":
                return dict_prop
            elif form == "series":
                return pd.Series(dict_prop)
            elif form == "df":
                return pd.DataFrame.from_dict(
                    dict_prop,
                    orient="index",
                ).T
            else:
                raise ValueError("Unknown form parameter: " + str(form))

        def set_type(self, type_str):
            self.type = type_str

        def set_affiliation(self, affil):
            assert isinstance(affil, ImgAgl.Agglomerate), (
                "given object "
                + str(affil)
                + "of type "
                + str(type(affil))
                + "is not a member of ImgAgl.Agglomerate class"
            )
            self.affil = affil.name

        def set_interIDs(self, list_=[]):
            self.interIDs = list_

    # ------------------ API methods --------------------------

    def clasify_all_AGL(self, threshold=0):
        for i in self._all_AGL_DF.OBJ:
            i.clasify(threshold)

    def find_particles(self):

        # self._init_p_coor_data()

        for i, row in self._initDF.iterrows():
            o = ImgAgl.Particle(
                row["ID imageJ"].astype("int64"), row.X, row.Y, row.D
            )

            self._all_P_DF.loc[i, "P ID"] = o.ID
            self._all_P_DF.loc[i, "name"] = o.name
            self._all_P_DF.loc[i, "D"] = o.D
            self._all_P_DF.loc[i, "OBJ"] = o
        return self._all_P_DF

    def find_agglomerates(self):
        self._find_all_intersecting()
        DF = self._all_P_DF.copy()

        j = 0
        while not DF.empty:
            particle = DF.iloc[0, DF.columns.get_loc("OBJ")]
            self._all_AGL_DF.loc[j, "OBJ"] = self.Agglomerate([])
            self._all_AGL_DF.loc[j, "ID"] = self._all_AGL_DF.loc[j, "OBJ"].ID
            self._all_AGL_DF.loc[j, "name"] = self._all_AGL_DF.loc[
                j, "OBJ"
            ].name
            iFamily = self._find_intersecting_family(particle)
            self._all_AGL_DF.loc[j, "OBJ"].append_particles(
                self.get_particles(iFamily)
            )
            for i in iFamily:
                DF.drop(DF.loc[DF["P ID"] == i].index, inplace=True)
            iFamily.clear()
            self._all_AGL_DF.loc[j, "OBJ"]._calc_member_param()
            self._all_AGL_DF.loc[j, "OBJ"]._calc_agl_param()

            # self._all_AGL_DF.loc[j,"members"] = str(self._all_AGL_DF.loc[j,"OBJ"].members)
            # self._all_AGL_DF.loc[j,"member count"] = self._all_AGL_DF.loc[j,"OBJ"].members_count
            j += 1

        # =============================================================================
        #         for i, row in self._all_AGL_DF.iterrows():
        #
        #             self._all_AGL_DF.loc[i,"OBJ"]._update_param()
        #             print(self._all_AGL_DF.loc[i,"OBJ"].members_count)
        # =============================================================================
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
        selected : list of ImgAgl.Particle objects
            List of particle objects

        """

        selected = []
        DF = self._all_P_DF
        for i in IDlist:
            s = DF[DF["P ID"] == i].OBJ.values[0]
            selected.append(s)
        return selected

    def get_largest_particle(self):
        return self._all_P_DF.OBJ[self._all_P_DF.D.idxmax()]

    def get_smallest_particle(self):
        return self._all_P_DF.OBJ[self._all_P_DF.D.idxmin()]

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
        self.res_particleDF = self._all_P_DF.loc[:, "OBJ"].apply(
            lambda p: p.get_properties(form="series")
        )
        return self.res_particleDF

    def get_results_aglTable(self):
        # aglTable = pd.DataFrame()
        # for i in self._all_AGL_DF.OBJ:
        #     aglTable = aglTable.append(i.get_properties())
        # aglTable.reset_index(inplace=True, drop=True)
        # self.res_aglDF = aglTable
        self.res_aglDF = self._all_AGL_DF.loc[:, "OBJ"].apply(
            lambda a: a.get_properties(form="series")
        )
        return self.res_aglDF

    def get_summary(self):
        # self.res_summary = self.res_summary.assign(pd.Series(len(self._all_P_DF.index), name="particle_count"))
        # print(self.res_summary)
        if len(self.res_aglDF.index) == 0:
            self.get_results_aglTable()
        self.res_summary.loc[0, "particle_count"] = len(self._all_P_DF.index)
        self.res_summary.loc[0, "agl_count"] = len(self._all_AGL_DF.index)

        self.res_summary.loc[0, "collector_agl_count"] = len(
            self.res_aglDF[self.res_aglDF["type"] == "collector"].index
        )
        self.res_summary.loc[0, "similar_agl_count"] = len(
            self.res_aglDF[self.res_aglDF["type"] == "similar"].index
        )
        self.res_summary.loc[0, "separate_count"] = len(
            self.res_aglDF[self.res_aglDF["type"] == "separate"].index
        )
        self.res_summary.loc[0, "ER"] = 1 - (
            self.res_summary.loc[0, "separate_count"]
            / self.res_summary.loc[0, "particle_count"]
        )
        self.res_summary.loc[0, "Ra"] = (
            self.res_summary.loc[0, "agl_count"]
            / self.res_summary.loc[0, "particle_count"]
        )
        self.res_summary.loc[0, "sep2agl"] = self.res_summary.loc[
            0, "separate_count"
        ] / (
            self.res_summary.loc[0, "collector_agl_count"]
            + self.res_summary.loc[0, "similar_agl_count"]
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
        self.res_summary.loc[0, "agl_member_count_mean"] = self.res_aglDF[
            "members_count"
        ].mean()
        self.res_summary.loc[0, "agl_member_count_std"] = self.res_aglDF[
            "members_count"
        ].std()

        # print(self.res_summary.dtypes)

        return self.res_summary

    def get_img(self):
        return self._img

    def get_img_filename(self):
        return self._img_filename

    def set_magnification(self, mag: Optional[float] = None) -> None:
        if not mag:
            self._img_meta_dict = self._read_meta_tif(self._img_path)
            lmag = self._img_meta_dict["CZ_SEM"]["ap_mag"][1].split()
            if "K" in lmag:
                self.mag = pd.to_numeric(lmag[0]) * 1000
            else:
                self.mag = pd.to_numeric(lmag[0])
        else:
            self.mag = float(mag)

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

                cbax = fig.add_axes([0.87, 0.25, 0.015, 0.6])
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

            if dirpath is not None:
                fname = dirpath / self.name.join("_circles.png")
            else:
                fname = self._path / self.name.join("_circles.png")
            fig.savefig(
                fname,
                facecolor="w",
                edgecolor="w",
                dpi=export_dpi,
                bbox_inches="tight",
                pad_inches=0,
            )
            print("IMAGE EXPORTED TO: ", fname)
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

        if isinstance(particles, ImgAgl.Particle):
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
            X = p.X / self.scf
            Y = p.Y / self.scf
            D = p.D / self.scf
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
            memTable.loc[i, "X"] = P.X / self.scf
            memTable.loc[i, "Y"] = P.Y / self.scf
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
            locX = int((AGL.get_members()[0].X) / self.scf - bboxW / 2)
            locY = int((AGL.get_members()[0].Y) / self.scf)
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
            if self.res_aglDF.empty:
                self.get_results_aglTable()
            vmin = self.res_aglDF.loc[:, prop_key].min()
            vmax = self.res_aglDF.loc[:, prop_key].max()
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
                "wron data input- only string and"
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
        for i, p in enumerate(self._all_P_DF.OBJ):
            val = p.get_properties()[prop_key]
            colors[i] = np.array(RGB_convert_to256(cmap(norm(val))))

        im1, im2 = self.draw_particles_transp(
            self._all_P_DF.OBJ,
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

    # -------------- internal methods ---------------------------

    def _read_meta_tif(self, file):
        """
        Method for reading .tif exif based metadata. Produces dictionary of
        exif tags from SEM tif file.

        Parameters
        ----------
        file : string
            Absolute path to SEM image tif file.

        Returns
        -------
        SEM_tags : dict
            Returns dictionary of exif tags from SEM tif file.
            SEM specific tags are included in CZ_SEM key (SEM_tags["CZ_SEM"])


        """

        import tifffile as tf

        with tf.TiffFile(file) as tif:
            tif_tags = {}
            for tag in tif.pages[0].tags.values():
                name, value = tag.name, tag.value
                tif_tags[name] = value
        return tif_tags

    # def _load_fitting_data(self, path=None):
    #     """

    #     UNFINISHED METHOD -
    #     to do:
    #         remake file loading system (settings, names.csv, image, fitting data)

    #     Parameters
    #     ----------
    #     path : TYPE, optional
    #         DESCRIPTION. The default is None.

    #     Returns
    #     -------
    #     None.

    #     """
    #     if path == None:
    #         path = self._path + os.sep

    #     if self._settings_DF["fitting_file_name"] == "default":
    #         self._fitRes_filename = "fitting_Results.csv"
    #     else:
    #         self._fitRes_filename = self._settings_DF["fitting_file_name"]
    #     self._fitRes = pd.read_csv(
    #         self._path + os.sep + self._fitRes_filename,
    #         sep=",",
    #         encoding="ansi",
    #     )

    def _init_p_coor_data(self):
        """
        This method prepares fitting results DataFrame for creating
        Particles, creates DKTree for analysis

        """

        try:
            self._fitRes
        except NameError:
            print("Particle fitting data sheet wasn't properly imported")
        for i in ["Index", "Type", "X", "Y", "Width", "Height"]:
            if i not in self._fitRes.columns:
                raise KeyError(
                    "Structure of particle fitting data sheet was not recognized "
                )

        res = self._fitRes.loc[
            :, ["Index", "Type", "X", "Y", "Width", "Height"]
        ]
        for i, row in res.iterrows():

            if row["Type"] == "Oval":
                if row["Width"] != row["Height"]:
                    print(
                        self._img_filename,
                        " (ROI ID:",
                        row["Index"],
                        ") --> dropping Oval (non circular) particle",
                    )
                    res = res.drop(i)
            else:
                print(
                    self._img_filename,
                    " (ROI ID:",
                    row["Index"],
                    ") --> dropping irregular particle",
                )
                res = res.drop(i)
        res = res.drop(["Type", "Height"], axis=1)
        res.Index = res.Index + 1
        res.rename(columns={"Index": "ID imageJ", "Width": "D"}, inplace=True)

        # applying scale to D (converting to µm)
        res.D = res.D * self.scf

        # applying scale to X and Y (converting to µm) and moving point to the
        # middle of the circle (oval in ImageJ is defined in upper-left corner)
        res.X = res.X * self.scf + res.D / 2
        res.Y = res.Y * self.scf + res.D / 2

        # sorting particles by size (descending) to assure proper order in
        # finding agglomerates
        res = res.sort_values(by=["D"], ascending=False)

        res.reset_index(drop=True, inplace=True)

        return res

    def _construct_KDTree(self):
        self._KDTree = spsp.KDTree(self._initDF[["X", "Y"]])
        return self._KDTree

    def _find_all_intersecting(self):
        if self._KDTree == None:
            # print("Program hasn\'t found KDTree structure. Initializing KDTree ...")
            self._construct_KDTree()
        KD = self._KDTree
        P_workDF = self._all_P_DF.OBJ
        D_MAX = self.get_largest_particle().D

        for P in P_workDF:
            nbrs = KD.query_ball_point([P.X, P.Y], D_MAX)
            # P.intersecting = []
            intersecting = []
            for i in nbrs:
                dist = 0
                contact = 0
                if P.ID != P_workDF[i].ID:
                    dist = (
                        (P_workDF[i].X - P.X) ** 2 + (P_workDF[i].Y - P.Y) ** 2
                    ) ** 0.5
                    contact = P.D / 2 + P_workDF[i].D / 2
                    if dist <= contact:
                        intersecting.append(P_workDF[i].ID)
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
                (self._all_P_DF.OBJ[i].X - particle.X) ** 2
                + (self._all_P_DF.OBJ[i].Y - particle.Y) ** 2
            ) ** 0.5
            contact = particle.D / 2 + self._all_P_DF.OBJ[i].D / 2

            if dist <= contact:
                intersecting.append(self._all_P_DF.OBJ[i].ID)
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


# =============================================================================
#     def _find_DF_IDs(self, key, value):
#
#
#         return self._all_P_DF[self._all_P_DF[key] == value]["P ID"].values[0]
#
#         # s = DF[DF["P ID"] == i].OBJ.values[0]
# =============================================================================


class nlcmap:
    def __init__(self, cmap, levels):
        self.name = cmap.name
        self.cmap = cmap
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype="float64")
        self._x = self.levels
        self.levmax = self.levels.max()
        self.transformed_levels = np.linspace(
            0.0, self.levmax, len(self.levels)
        )

    def __call__(self, xi, alpha=1.0, **kw):
        yi = np.interp(xi, self._x, self.transformed_levels)
        return self.cmap(yi / self.levmax, alpha)
