import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from agglpy.auxiliary import (
    get_ceil,
    get_floor,
)
from agglpy.cfg import load_manager_settings
from agglpy.defaults import SUPPORTED_IMG_FORMATS
from agglpy.dir_structure import find_datasets_paths, validate_mgr_dirstruct
from agglpy.errors import DirectoryStructureError, MultipleFilesFoundError
from agglpy.img_ds import ImgDataSet
from agglpy.logger import logger
from agglpy.typing import YamlSettingsTypedDict


class Manager:
    """
    description



    """

    # Public attributes
    name: str
    collector_threshold: float | None

    img_info: pd.DataFrame | None
    batch_res_pDF: pd.DataFrame | None
    batch_res_aglDF: pd.DataFrame | None
    batch_res_PSD: pd.DataFrame | None
    batch_res_DSsummary: pd.DataFrame | None
    batch_res_summary: pd.DataFrame | None

    # Private attributes
    _ID: uuid.UUID
    _shortID: str
    _workdir: Path
    _settings_path: Path
    _settings: YamlSettingsTypedDict
    _DS_paths: List[Path]
    _DS: List[ImgDataSet]  # Used to store ImgDataSet opjects
    _PSD_space: npt.NDArray | None

    def __init__(
        self,
        working_dir: os.PathLike,
        name: str | None = None,
        settings_filepath: Path = Path("settings.yml"),
        init_data_sets: bool = True,
    ):
        # Initialize attributes directly related to constructor arguments
        self._workdir = Path(
            working_dir
        )  # used for immutable @property working_dir
        if not name:
            self.name = self._workdir.name
        else:
            self.name = name
        self._ID = uuid.uuid4()
        self._shortID = str(self._ID).split("-")[0]
        if not settings_filepath.is_absolute():  # Settings path
            self._settings_path = self._workdir / settings_filepath
        else:
            self._settings_path = settings_filepath

        # Initialize public attributes
        self.collector_threshold = None

        # Initialize public result DataFrames
        self.img_info = None
        self.batch_res_pDF = None
        self.batch_res_aglDF = None
        self.batch_res_PSD = None
        self.batch_res_DSsummary = None
        self.batch_res_summary = None

        # Initialize private attributes
        self._DS_paths = []
        self._DS = []
        self._PSD_space = None

        logger.info(f"Creating Manager object {str(self)} at: {working_dir}.")

        try:
            validate_mgr_dirstruct(self._workdir)
        except DirectoryStructureError as err:
            logger.error(msg=err.args[0])
            raise err

        # Load settings file
        self._settings = load_manager_settings(path=self._settings_path)
        self.collector_threshold = self._settings["analysis"][
            "collector_threshold"
        ]

        if init_data_sets:
            self.create_image_data_sets()
        logger.info(
            f"{str(self)} Manager object was successfully created at {str(self.working_dir)}"
        )

    # ----------- Manager Properties
    @property
    def working_dir(self) -> Path:
        """Working directory of analysis manager

        This is read-only attribute setted at the Manager object construction

        Returns:
            Path: Working directory path
        """
        return self._workdir

    def batch_detect_primary_particles(self, force: bool = False):
        if force:
            # process all ImgDataSets and overwrite the results
            ds_process = self._DS
        else:
            # process only ImgDataSets without available particle csv data
            ds_process = [ds for ds in self._DS if not ds._PPsource_flag]
        if len(ds_process) == 0:
            wmsg = (
                f"All of the ImgDataSets already have loaded primary particles "
                f"source file. Detection is not needed. Check the 'HCT_file' "
                f"key in settings.data.images if some of the data sets still "
                f"need detection. Otherwise you may want to force "
                f"the detection using force=True, but this will overwrite "
                f"the existing results."
            )
            warnings.warn(wmsg)
            logger.warning(wmsg)
        else:
            for ds in ds_process:
                ds.detect_primary_particles(
                    export_img=True,
                    export_csv=True,
                    export_edges=True,
                )
            logger.info(
                f"Primary particles detected for ImgDataSets: "
                f"{[str(ds) for ds in ds_process]}. CSV file and images were "
                f"exported to respective ImgDataSet directory"
            )

    def batch_detect_agglomerates(self, export_img=True):
        """
        Method for batch analysis of SEM photo and circle fitting data
        from folder structure


        Returns
        -------
        None.

        """
        logger.debug(f"{str(self)} Beginning analysis for all DataSets")
        total1 = len(self._DS)
        pbar1 = tqdm(self._DS, total=total1, position=0, leave=True)

        for i in pbar1:
            desc0 = "Processing DataSet|" + i.name + "| "
            pbar1.update(0)
            pbar1.set_description(desc0 + "- finding agglomerates...")
            i.detect_agglomerates()
            pbar1.update(0)
            pbar1.set_description(desc0 + "- classifying agglomerates...")
            i.classify_all_AGL(self.collector_threshold)
            pbar1.update(0)
            if export_img == True:
                pbar1.set_description(desc0 + "- drawing agglomerates...")
                draw_labels = self._settings["export"]["draw_particles"][
                    "labels"
                ]
                img = i.draw_all_agl_by(prop="ID", labels=draw_labels)
                i.plot_img(img, show=True, export=True)
                pbar1.update(0)
        pbar1.close()
        logger.info(f"{str(self)} Analysis for all DataSets finished.")

    def create_image_data_sets(self, auto_load: bool = True) -> None:
        self._DS_paths = find_datasets_paths(
            path=self._workdir,
            settings=self._settings,
            ignore=True,
        )

        # Constructing DataSets (ImgAgl objects) for analysis
        # and filling IMG_INFO table
        logger.debug(f"{str(self)} Creating ImgDataSet objects.")
        info_dict: Dict[str, List[Any]] = {
            "img_name": [],
            "magnification": [],
            "pixel_size": [],
            "subdir": [],
        }
        for i, path in enumerate(self._DS_paths):
            self._DS.append(
                ImgDataSet(
                    path.parent,
                    settings=self._settings["data"]["images"][
                        path.parent.stem
                    ],
                    auto_load=auto_load,
                )
            )
            info_dict["img_name"].append(self._DS[i].get_img_filename())
            info_dict["magnification"].append(self._DS[i].mag)
            info_dict["pixel_size"].append(self._DS[i].px_size)
            info_dict["subdir"].append(self._DS[i].path)
        self.img_info = pd.DataFrame.from_dict(
            data=info_dict, orient="columns"
        )
        logger.info(
            f"{str(self)} Succesfully appended DataSets: "
            f"{str([str(i) for i in self._DS])}"
        )

    def generate_pTable(self):
        self.batch_res_pDF = pd.DataFrame()
        for i, ds in enumerate(self._DS):
            temp = ds.get_results_pTable()
            # print(temp)
            temp.drop(["X", "Y"], axis=1, inplace=True)
            temp["DS ID"] = ds.name
            cols = temp.columns.tolist()
            cols.insert(0, cols.pop(cols.index("DS ID")))
            temp = temp.reindex(columns=cols)
            self.batch_res_pDF = pd.concat(
                [self.batch_res_pDF, temp], axis=0, ignore_index=True
            )
        logger.info(
            f"{str(self)} Particle table for analyzed DataSets created. "
            f"It contains: {len(self.batch_res_pDF.index)} primary particles."
        )
        return self.batch_res_pDF

    def generate_aglTable(self):
        self.batch_res_aglDF = pd.DataFrame()
        for i, ds in enumerate(self._DS):
            temp = ds.get_results_aglTable()
            temp["DS ID"] = ds.name
            cols = temp.columns.tolist()
            cols.insert(0, cols.pop(cols.index("DS ID")))
            temp = temp.reindex(columns=cols)
            self.batch_res_aglDF = pd.concat(
                [self.batch_res_aglDF, temp], axis=0, ignore_index=True
            )
        logger.info(
            f"{str(self)} Agglomerate table for analyzed DataSets created. "
            f"It contains: {len(self.batch_res_aglDF.index)} agglomerates."
        )
        return self.batch_res_aglDF

    def generate_PSD(self, PSD_space=None, plot=True):
        if self.batch_res_pDF.empty:
            self.generate_pTable()
        if PSD_space is None:
            if self._PSD_space is None:
                self.set_PSD_space()
            PSD_space = self._PSD_space
        # print(self.batch_res_pDF.D)
        # print(self._PSD_space)
        cut = pd.cut(self.batch_res_pDF.D, bins=PSD_space, include_lowest=True)
        self.batch_res_PSD = pd.value_counts(cut, sort=False)
        lefts = []
        mids = []
        rights = []
        widths = []
        for i in self.batch_res_PSD.index:
            lefts.append(i.left)
            mids.append(i.mid)
            rights.append(i.right)
            widths.append(i.length)
        lefts = np.array(lefts)
        mids = np.array(mids)
        rights = np.array(rights)
        widths = np.array(widths)
        self.batch_res_PSD.name = "counts"
        self.batch_res_PSD = pd.DataFrame(self.batch_res_PSD)
        self.batch_res_PSD["left"] = lefts
        self.batch_res_PSD["mid"] = mids
        self.batch_res_PSD["right"] = rights
        self.batch_res_PSD["width"] = widths
        cols = self.batch_res_PSD.columns.tolist()
        cols.insert(0, cols.pop(cols.index("left")))
        cols.insert(1, cols.pop(cols.index("mid")))
        cols.insert(2, cols.pop(cols.index("right")))
        cols.insert(3, cols.pop(cols.index("width")))
        csum = self.batch_res_PSD["counts"].sum()
        self.batch_res_PSD = self.batch_res_PSD.reindex(columns=cols)
        self.batch_res_PSD["cummulative"] = self.batch_res_PSD[
            "counts"
        ].cumsum()
        self.batch_res_PSD["counts_norm"] = self.batch_res_PSD["counts"] / csum
        self.batch_res_PSD["cummulative_norm"] = self.batch_res_PSD[
            "counts_norm"
        ].cumsum()
        self.batch_res_PSD["volume"] = (
            self.batch_res_PSD["counts"]
            * 1
            / 6
            * np.pi
            * (self.batch_res_PSD["mid"]) ** 3
        )
        totalvol = self.batch_res_PSD["volume"].sum()
        self.batch_res_PSD["volume_cummulative"] = self.batch_res_PSD[
            "volume"
        ].cumsum()
        self.batch_res_PSD["volume_norm"] = (
            self.batch_res_PSD["volume"] / totalvol
        )
        self.batch_res_PSD["volume_cummulative_norm"] = self.batch_res_PSD[
            "volume_norm"
        ].cumsum()

        if plot == True:
            self.plot_PSD(norm=False, cummul=True, export=True)
        PSD_space_str = np.array2string(
            PSD_space,
            precision=2,
            threshold=50,  # maximum of 50 elements fully printed
            edgeitems=5,  # number of edge items printed if array is too long
            max_line_width=np.inf,  # no limit on line width
        )
        logger.info(
            f"{str(self)} Primary Particle size distribution table created. "
            f"PSD_space used: {PSD_space_str}"
        )
        # return self.batch_res_PSD

    def generate_aglPSD(
        self,
        PSD_space=None,
        plot=True,
        include_dsom=False,
    ):
        if self.batch_res_aglDF.empty:
            self.generate_aglTable()
        if PSD_space is None:
            if self._PSD_space.size == 0:
                self.set_PSD_space()
            PSD_space = self._PSD_space
        if include_dsom:
            diameters = self.batch_res_aglDF.loc[:, "D_dsom"]
        else:
            diameters = self.batch_res_aglDF.loc[:, "D"]
        cut = pd.cut(diameters, bins=PSD_space, include_lowest=True)
        self.batch_res_aglPSD = pd.value_counts(cut, sort=False)

        lefts = []
        mids = []
        rights = []
        widths = []
        for i in self.batch_res_aglPSD.index:
            lefts.append(i.left)
            mids.append(i.mid)
            rights.append(i.right)
            widths.append(i.length)
        lefts = np.array(lefts)
        mids = np.array(mids)
        rights = np.array(rights)
        widths = np.array(widths)
        self.batch_res_aglPSD.name = "counts"
        self.batch_res_aglPSD = pd.DataFrame(self.batch_res_aglPSD)
        self.batch_res_aglPSD["left"] = lefts
        self.batch_res_aglPSD["mid"] = mids
        self.batch_res_aglPSD["right"] = rights
        self.batch_res_aglPSD["width"] = widths
        cols = self.batch_res_aglPSD.columns.tolist()
        cols.insert(0, cols.pop(cols.index("left")))
        cols.insert(1, cols.pop(cols.index("mid")))
        cols.insert(2, cols.pop(cols.index("right")))
        cols.insert(3, cols.pop(cols.index("width")))
        csum = self.batch_res_aglPSD["counts"].sum()
        self.batch_res_aglPSD = self.batch_res_aglPSD.reindex(columns=cols)
        self.batch_res_aglPSD["cummulative"] = self.batch_res_aglPSD[
            "counts"
        ].cumsum()
        self.batch_res_aglPSD["counts_norm"] = (
            self.batch_res_aglPSD["counts"] / csum
        )
        self.batch_res_aglPSD["cummulative_norm"] = self.batch_res_aglPSD[
            "counts_norm"
        ].cumsum()
        self.batch_res_aglPSD["volume"] = (
            self.batch_res_aglPSD["counts"]
            * 1
            / 6
            * np.pi
            * (self.batch_res_aglPSD["mid"]) ** 3
        )
        totalvol = self.batch_res_aglPSD["volume"].sum()
        self.batch_res_aglPSD["volume_cummulative"] = self.batch_res_aglPSD[
            "volume"
        ].cumsum()
        self.batch_res_aglPSD["volume_norm"] = (
            self.batch_res_aglPSD["volume"] / totalvol
        )
        self.batch_res_aglPSD["volume_cummulative_norm"] = (
            self.batch_res_aglPSD["volume_norm"].cumsum()
        )

        if plot == True:
            self.plot_aglPSD(norm=False, cummul=True, export=True)

        PSD_space_str = np.array2string(
            PSD_space,
            precision=2,
            threshold=50,  # maximum of 50 elements fully printed
            edgeitems=5,  # number of edge items printed if array is too long
            max_line_width=np.inf,  # no limit on line width
        )
        logger.info(
            f"{str(self)} Agglomerate size distribution table created. "
            f"PSD_space used: {PSD_space_str}"
        )

    def generate_aglPCD(
        self,
        PCD_space,
        plot=False,
        include_dsom=False,
    ):
        if self.batch_res_aglDF.empty:
            self.generate_aglTable()
        if include_dsom:
            counts = self.batch_res_aglDF.loc[:, "members_count_dsom"]
        else:
            counts = self.batch_res_aglDF.loc[:, "members_count"]
        cut = pd.cut(counts, bins=PCD_space, include_lowest=True)
        self.batch_res_aglPCD = pd.value_counts(cut, sort=False)

        lefts = []
        mids = []
        rights = []
        widths = []
        for i in self.batch_res_aglPCD.index:
            lefts.append(i.left)
            mids.append(i.mid)
            rights.append(i.right)
            widths.append(i.length)
        lefts = np.array(lefts)
        mids = np.array(mids)
        rights = np.array(rights)
        widths = np.array(widths)
        self.batch_res_aglPCD.name = "counts"
        self.batch_res_aglPCD = pd.DataFrame(self.batch_res_aglPCD)
        self.batch_res_aglPCD["left"] = lefts
        self.batch_res_aglPCD["mid"] = mids
        self.batch_res_aglPCD["right"] = rights
        self.batch_res_aglPCD["width"] = widths
        cols = self.batch_res_aglPCD.columns.tolist()
        cols.insert(0, cols.pop(cols.index("left")))
        cols.insert(1, cols.pop(cols.index("mid")))
        cols.insert(2, cols.pop(cols.index("right")))
        cols.insert(3, cols.pop(cols.index("width")))
        csum = self.batch_res_aglPCD["counts"].sum()
        self.batch_res_aglPCD = self.batch_res_aglPCD.reindex(columns=cols)
        self.batch_res_aglPCD["cummulative"] = self.batch_res_aglPCD[
            "counts"
        ].cumsum()
        self.batch_res_aglPCD["counts_norm"] = (
            self.batch_res_aglPCD["counts"] / csum
        )
        self.batch_res_aglPCD["cummulative_norm"] = self.batch_res_aglPCD[
            "counts_norm"
        ].cumsum()

        if plot == True:
            self.plot_aglPCD(norm=False, cummul=True, export=True)

        PCD_space_str = np.array2string(
            PCD_space,
            precision=2,
            threshold=50,  # maximum of 50 elements fully printed
            edgeitems=5,  # number of edge items printed if array is too long
            max_line_width=np.inf,  # no limit on line width
        )
        logger.info(
            f"{str(self)} Primary Particle count distribution table created. "
            f"PCD_space used: {PCD_space_str}"
        )

    def generate_DSsummary(self):
        self.batch_res_DSsummary = pd.DataFrame()
        for i, ds in enumerate(self._DS):
            temp = ds.get_summary()
            temp["DS ID"] = ds.name
            cols = temp.columns.tolist()
            cols.insert(0, cols.pop(cols.index("DS ID")))
            temp = temp.reindex(columns=cols)
            self.batch_res_DSsummary = pd.concat(
                [self.batch_res_DSsummary, temp], axis=0, ignore_index=True
            )
        logger.info(f"{str(self)} DataSets summary table created.")

    def generate_summary(self):
        # TODO: batch_res_<specifier> logic needs to be reordered / redesigned
        # self.batch_res_DSsummary may be None at this point
        DSsumm = self.batch_res_DSsummary
        summ = pd.DataFrame()  # self.batch_res_summary
        if len(DSsumm.index) == 0:
            self.generate_DSsummary()
        summ = summ.reindex_like(DSsumm)
        summ.drop("DS ID", axis=1, inplace=True)
        summ = summ.head(1)
        summ["N_primary_particle"] = DSsumm["N_primary_particle"].sum()
        summ["N_aerosol_particle"] = DSsumm["N_aerosol_particle"].sum()
        summ["N_pp1"] = DSsumm["N_pp1"].sum()
        summ["N_ppA"] = DSsumm["N_ppA"].sum()
        summ["N_agl"] = DSsumm["N_agl"].sum()
        summ["N_collector_agl"] = DSsumm["N_collector_agl"].sum()
        summ["N_similar_agl"] = DSsumm["N_similar_agl"].sum()
        summ["N_pp1_separate"] = DSsumm["N_pp1_separate"].sum()
        summ["ER"] = 1 - (summ["N_pp1_separate"] / summ["N_primary_particle"])
        summ["Ra"] = summ["N_agl"] / summ["N_primary_particle"]
        summ["sep2agl"] = summ["N_pp1_separate"] / summ["N_agl"]
        summ["n_ppA"] = summ["N_ppA"] / summ["N_agl"]
        summ["n_ppP"] = summ["N_primary_particle"] / summ["N_aerosol_particle"]
        summ["particle_Dmean"] = self.batch_res_pDF["D"].mean()
        summ["particle_Dstd"] = self.batch_res_pDF["D"].std()
        summ["particle_D10"] = self.batch_res_pDF["D"].quantile(q=0.1)
        summ["particle_D50"] = self.batch_res_pDF["D"].quantile(q=0.5)
        summ["particle_D90"] = self.batch_res_pDF["D"].quantile(q=0.9)
        summ["particle_SMD"] = (self.batch_res_pDF.loc[:, "D"] ** 3).sum() / (
            self.batch_res_pDF.loc[:, "D"] ** 2
        ).sum()
        summ["agl_Dmean"] = self.batch_res_aglDF["D"].mean()
        summ["agl_Dstd"] = self.batch_res_aglDF["D"].std()
        summ["agl_D10"] = self.batch_res_aglDF["D"].quantile(q=0.1)
        summ["agl_D50"] = self.batch_res_aglDF["D"].quantile(q=0.5)
        summ["agl_D90"] = self.batch_res_aglDF["D"].quantile(q=0.9)
        summ["agl_member_count_mean"] = self.batch_res_aglDF[
            "members_count"
        ].mean()
        summ["agl_member_count_std"] = self.batch_res_aglDF[
            "members_count"
        ].std()
        summ["agl_member_count_q10"] = self.batch_res_aglDF[
            "members_count"
        ].quantile(q=0.1)
        summ["agl_member_count_q50"] = self.batch_res_aglDF[
            "members_count"
        ].quantile(q=0.5)
        summ["agl_member_count_q90"] = self.batch_res_aglDF[
            "members_count"
        ].quantile(q=0.9)
        self.batch_res_summary = summ.T

        logger.info(
            f"{str(self)} Results summary table created. "
            f"Mean Aerosol Particle Primary Particle Count: "
            f"{self.batch_res_summary.loc['n_ppP', 0]:.3f}"
        )

    def get_path(self):
        return self._workdir

    def get_pTable(self):
        return self.batch_res_pDF

    def get_aglTable(self):
        return self.batch_res_aglDF

    def get_summary(self):
        return self.batch_res_summary

    def get_min_pD(self):
        Dlist = []
        for i, ds in enumerate(self._DS):
            Dlist.append(ds.get_smallest_particle().D)
        return min(Dlist)

    def get_max_pD(self):
        Dlist = []
        for i, ds in enumerate(self._DS):
            Dlist.append(ds.get_largest_particle().D)
        return max(Dlist)

    def get_PSD(
        self,
        PSD_space=None,
        plot=False,
        orientation="left",
        norm=False,
        cpsd=False,
    ):
        if self.batch_res_PSD.empty or (PSD_space is not None):
            self.generate_PSD(PSD_space=PSD_space, plot=plot)
        else:
            self.set_PSD_space()
            self.generate_PSD(PSD_space=None, plot=plot)
        selection = [None, None]
        if orientation == "left":
            selection[0] = "left"
        elif orientation in ["mid", "center"]:
            selection[0] = "mid"
        elif orientation == "right":
            selection[0] = "right"
        if not norm:
            if not cpsd:
                selection[1] = "counts"
            else:
                selection[1] = "cummulative"
        else:
            if not cpsd:
                selection[1] = "counts_norm"
            else:
                selection[1] = "cummulative_norm"

        return self.batch_res_PSD.loc[:, selection]

    def get_distribution(
        self,
        typ="PSD",
        PSD_space=None,
        plot=False,
        orientation="left",
        norm=False,
        cpsd=False,
    ):
        assert typ in ["PSD", "aglPSD", "aglPCD"], (
            "typ must be one of the strings:" " 'PSD', 'aglPSD', 'aglPCD'"
        )
        wDF = pd.DataFrame()
        if typ == "PSD":
            self.generate_PSD(PSD_space=PSD_space, plot=plot)
            wDF = self.batch_res_PSD
        elif typ == "aglPSD":
            self.generate_aglPSD(PSD_space=PSD_space, plot=plot)
            wDF = self.batch_res_aglPSD
        elif typ == "aglPCD":
            self.generate_aglPCD(PCD_space=PSD_space, plot=plot)
            wDF = self.batch_res_aglPCD

        # print(wDF)
        # if PSD_space is not None:
        #     gen_func(PSD_space=PSD_space, plot=plot)
        # else:
        #     self.set_PSD_space()
        #     gen_func(PSD_space=None, plot=plot)

        selection = [None, None]
        if orientation == "left":
            selection[0] = "left"
        elif orientation in ["mid", "center"]:
            selection[0] = "mid"
        elif orientation == "right":
            selection[0] = "right"
        if not norm:
            if not cpsd:
                selection[1] = "counts"
            else:
                selection[1] = "cummulative"
        else:
            if not cpsd:
                selection[1] = "counts_norm"
            else:
                selection[1] = "cummulative_norm"

        return wDF.loc[:, selection]

    def get_all_aglOBJ(self):
        pass

    def plot_PSD(self, norm=False, cummul=True, export=False, lines=True):
        fig, ax1 = plt.subplots()
        if norm == True:
            h = 100 * self.batch_res_PSD["counts_norm"]
            ax1.set_ylabel("dN/N [%]")
        else:
            h = self.batch_res_PSD["counts"]
            ax1.set_ylabel("dN [#]")

        if lines == False:
            ax1.step(
                x=self.batch_res_PSD["right"], y=h, color="black", linewidth=1
            )
            ax1.bar(
                self.batch_res_PSD["left"],
                height=h,
                width=self.batch_res_PSD["width"],
                align="edge",
                linewidth=0,
                color="xkcd:azure",
            )
        else:
            ax1.bar(
                self.batch_res_PSD["mid"],
                height=h,
                width=self.batch_res_PSD["width"],
                align="center",
                edgecolor="black",
                linewidth=0.5,
                color="xkcd:azure",
            )
        if cummul == True:
            ax2 = ax1.twinx()
            ax2.plot(
                self.batch_res_PSD["mid"],
                100 * self.batch_res_PSD["cummulative_norm"],
                "r-",
            )
            ax2.set_ylabel("cumulative dN/N [%]")

        ax1.set_xlabel("Diameter [\u03BCm]")
        fig.show()
        if export == True:
            if norm == True:
                PSDimg = (
                    self._workdir
                    / os.path.basename(self._workdir)
                    / "_particle_normPSD.png"
                )
            else:
                PSDimg = (
                    self._workdir
                    / os.path.basename(self._workdir)
                    / "_particle_PSD.png"
                )
            plt.savefig(PSDimg, dpi=300)

    def plot_aglPSD(self, norm=False, cummul=True, export=False, lines=True):
        fig, ax1 = plt.subplots()
        if norm == True:
            h = 100 * self.batch_res_aglPSD["counts_norm"]
            ax1.set_ylabel("dN/N [%]")
        else:
            h = self.batch_res_aglPSD["counts"]
            ax1.set_ylabel("dN [#]")

        if lines == False:
            ax1.step(
                x=self.batch_res_aglPSD["right"],
                y=h,
                color="black",
                linewidth=1,
            )
            ax1.bar(
                self.batch_res_aglPSD["left"],
                height=h,
                width=self.batch_res_aglPSD["width"],
                align="edge",
                linewidth=0,
                color="lightgrey",
            )
        else:
            ax1.bar(
                self.batch_res_aglPSD["mid"],
                height=h,
                width=self.batch_res_aglPSD["width"],
                align="center",
                edgecolor="black",
                linewidth=0.5,
                color="lightgrey",
            )
        if cummul == True:
            ax2 = ax1.twinx()
            ax2.plot(
                self.batch_res_aglPSD["mid"],
                100 * self.batch_res_aglPSD["cummulative_norm"],
                "r-",
            )
            ax2.set_ylabel("cumulative dN/N [%]")

        ax1.set_xlabel("Agglomerate equivalent diameter [\u03BCm]")
        fig.show()
        if export == True:
            if norm == True:
                PSDimg = (
                    self._workdir
                    / os.path.basename(self._workdir)
                    / "_agl_normPSD.png"
                )
            else:
                PSDimg = (
                    self._workdir
                    / os.path.basename(self._workdir)
                    / "_agl_PSD.png"
                )
            plt.savefig(PSDimg, dpi=300)

    def export_all_results(self):
        xls_file = self._workdir / (self._workdir.name + "_agl_analysis.xlsx")
        with pd.ExcelWriter(xls_file) as writer:
            exp_conditions = pd.DataFrame.from_dict(
                self._settings["metadata"]["conditions"]
            )
            exp_conditions.to_excel(writer, sheet_name="conditions")
            self.img_info.to_excel(writer, sheet_name="img_info")
            self.batch_res_summary.to_excel(writer, sheet_name="summary")
            self.batch_res_DSsummary.to_excel(
                writer, sheet_name="DataSets_summary"
            )
            self.batch_res_PSD.to_excel(writer, sheet_name="PSD")
            self.batch_res_aglPSD.to_excel(writer, sheet_name="aglPSD")
            self.batch_res_aglPCD.to_excel(writer, sheet_name="aglPCD")
            self.batch_res_pDF.to_excel(writer, sheet_name="particle_data")
            self.batch_res_aglDF.to_excel(writer, sheet_name="agl_data")
        logger.info(f"{str(self)} Results exported to excel file: {xls_file}.")

    def set_PSD_space(self):
        s = self._settings["analysis"]["PSD_space"]

        space = []

        if s is None:
            # set PSD space automatically
            dmin = self.get_min_pD()
            dmax = self.get_max_pD()
            start = get_floor(dmin)
            end = get_ceil(dmax)
            space = PSD_space(
                start=start,
                end=end,
                periods=20,
                log=False,
                step=False,
            )
        else:
            space = PSD_space(**s)
        return space

    def set_PSD_space_old(self):
        s = self._settings["analysis"]["PSD_space"]
        log = self._settings["analysis"]["PSD_space_log"]

        space = []
        if ("[" == s[0]) and ("]" == s[-1]):
            space = s.replace("[", "")
            space = space.replace("]", "")
            space = space.split(",")
            space = [float(i) for i in space]
        elif "," in s:
            param = s.split(",")
            # print(param)
            assert len(param) in [3, 4], (
                "Wrong structure of PSD_space "
                "variable in settings.csv. If input is list of interval bounds- "
                "ensure that this parameter starts and ends with [ and ]"
            )

            if "step" in param:
                param.remove("step")
                step_bool = True
            else:
                step_bool = False

            sp_start = float(param[0])
            sp_end = float(param[1])

            if step_bool:
                try:
                    sp_periods = float(param[2])
                except ValueError:
                    sp_periods = max(self.img_info["pixel size [um]"])
            else:
                sp_periods = int(param[2])

            space = PSD_space(
                sp_start, sp_end, sp_periods, log=log, step=step_bool
            )

        else:
            try:
                if float(s).is_integer():
                    sp_start = self.get_min_pD()
                    sp_end = self.get_max_pD()
                    sp_periods = int(s)
                    space = PSD_space(sp_start, sp_end, sp_periods, log=log)
            except ValueError:
                sp_start = self.get_min_pD()
                sp_end = self.get_max_pD()
                sp_periods = max(self.img_info["pixel size [um]"])
                space = PSD_space(
                    sp_start, sp_end, sp_periods, log=False, step=True
                )

        self._PSD_space = space
        return space

    # ----------- dunder methods
    def __getitem__(self, key):
        return self._DS[key]

    def __iter__(self):
        yield from self._DS

    def __repr__(self):
        class_name = type(self).__name__
        repr_str = (
            f"{class_name}(working_dir={self.working_dir!r}, "
            f"name= {self.name!r}, settings_filepath= {self._settings_path})"
        )
        return repr_str

    def __str__(self):
        return f"<M({self.name})#{self._shortID}>"


def PSD_space(
    start: float = 0.0,
    end: float = 10e-6,
    periods: int | float = 20,
    log: bool = False,
    step: bool = False,
):
    if log:
        # Ensure that start and end are positive for logarithmic space
        if start <= 0 or end <= 0:
            raise ValueError(
                "Start and end must be positive for logarithmic scaling."
            )

        # Convert start and end to logarithmic space
        log_start = np.log10(start)
        log_end = np.log10(end)

        if step:
            # Generate bins in logarithmic space using the given step
            bins_arr = 10 ** np.arange(log_start, log_end + periods, periods)
        else:
            # Generate bins with equal number of divisions in logarithmic space
            periods = int(periods)
            bins_arr = np.logspace(log_start, log_end, num=periods + 1)
    else:
        if step:
            # Divide the diameter space by defined step size
            bins_arr = np.arange(start, end + periods, periods, dtype=float)
        else:
            # Divide the diameter space by the number of steps, infer step size
            periods = int(periods)
            bins_arr = np.linspace(
                start=start, stop=end, num=periods + 1, dtype=float
            )
    return bins_arr
