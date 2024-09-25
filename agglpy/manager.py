import os
from pathlib import Path
from typing import (
    List,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from agglpy.img_ds import ImgDataSet
from agglpy.cfg import (
    load_manager_settings,
    SUPPORTED_IMG_FORMATS,
)
from agglpy.dir_structure import validate_mgr_dirstruct, find_datasets_paths
from agglpy.errors import MultipleFilesFoundError, DirectoryStructureError
from agglpy.img_process import (
    HCTcv,
    HCTcv_multi,
)


class Manager:
    """
    description

    Parameters
    ----------
    param1 :
        desc
    param2 : int, optional

    """

    def __init__(
        self,
        working_dir: Path,
        settings_filepath: Path = Path("settings.yml"),
        initialize: bool = True,
    ):

        self._workdir: Path = Path(working_dir)  # Working directory

        if not settings_filepath.is_absolute():  # Settings path
            self._settings_path: Path = self._workdir / settings_filepath
        else:
            self._settings_path: Path = settings_filepath
        validate_mgr_dirstruct(self._workdir)
        # Loading settings file
        self._settings = load_manager_settings(path=self._settings_path)
        self.collector_threshold: float = self._settings["analysis"][
            "collector_threshold"
        ]

        if initialize:

            # Defining directories to work with
            # TO DO: first check directory structure
            # and then iterate over datsets
            self._DS_paths = find_datasets_paths(
                path=self._workdir,
                settings=self._settings,
                ignore=True,
            )

            # Constructing DataSets (ImgAgl objects) for analysis
            # and creating IMG_INFO table
            self._DS = []
            self.img_info = pd.DataFrame()
            for i, path in enumerate(self._DS_paths):
                self._DS.append(
                    ImgDataSet(
                        path.parent,
                        settings=self._settings["data"]["images"][path.stem],
                    )
                )

                self.img_info.loc[i, "img_name"] = self._DS[
                    i
                ].get_img_filename()
                self.img_info.loc[i, "magnification"] = self._DS[i].mag
                self.img_info.loc[i, "pixel size [um]"] = self._DS[i].scf
                self.img_info.loc[i, "subdir"] = (
                    os.path.basename(self._workdir)
                    + os.sep
                    + os.path.basename(path)
                )

            self.batch_res_pDF = pd.DataFrame()
            self.batch_res_aglDF = pd.DataFrame()
            self.batch_res_PSD = pd.DataFrame()
            self.batch_res_DSsummary = pd.DataFrame()
            self.batch_res_summary = pd.DataFrame()
            self._PSD_space = np.array([])

    # ----------- Manager Properties
    @property
    def working_dir(self) -> Path:
        """Working directory of analysis manager

        This is read-only attribute setted at the Manager object construction

        Returns:
            Path: Working directory path
        """
        return self._workdir

    def batch_detect_circles(self, export_img):

        pass

    def batch_analysis(self, export_img=True):
        """
        Method for batch analysis of SEM photo and circle fitting data
        from folder structure


        Returns
        -------
        None.

        """
        total1 = len(self._DS)
        pbar1 = tqdm(self._DS, total=total1, position=0, leave=True)

        for i in pbar1:
            desc0 = "Processing DataSet|" + i.name + "| "
            pbar1.update(0)
            pbar1.set_description(desc0 + "- finding agglomerates...")
            i.find_agglomerates()
            pbar1.update(0)
            pbar1.set_description(desc0 + "- classifying agglomerates...")
            i.clasify_all_AGL(self.collector_threshold)
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

        return self.batch_res_aglDF

    def generate_PSD(self, PSD_space=None, plot=True):
        if self.batch_res_pDF.empty:
            self.generate_pTable()
        if PSD_space is None:
            if self._PSD_space.size == 0:
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

    # def generate_fluidsPSD(self):
    #     self.batch_res_fpPSD = fPSD.ParticleSizeDistribution(
    #         ds=self._PSD_space, fractions=self.batch_res_PSD["counts"], order=0
    #     )
    #     self.batch_res_faglPSD = fPSD.ParticleSizeDistribution(
    #         ds=self._PSD_space,
    #         fractions=self.batch_res_aglPSD["counts"],
    #         order=0,
    #     )

    def generate_DSsummary(self):
        for i, ds in enumerate(self._DS):
            temp = ds.get_summary()
            temp["DS ID"] = ds.name
            cols = temp.columns.tolist()
            cols.insert(0, cols.pop(cols.index("DS ID")))
            temp = temp.reindex(columns=cols)
            self.batch_res_DSsummary = pd.concat(
                [self.batch_res_DSsummary, temp], axis=0, ignore_index=True
            )

    def generate_summary(self):
        DSsumm = self.batch_res_DSsummary
        summ = self.batch_res_summary
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
        summ["particle_SMD"] = (
            (self.batch_res_pDF.loc[:, "D"]**3).sum() 
            / (self.batch_res_pDF.loc[:,"D"]**2).sum()
        )
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

    def set_PSD_space(self):
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

    # ----------- Manager internal methods

    def _find_datasets_paths(self, ignore: bool = True):
        """Detect Data Sets directory structure

        Detect Data Sets in working directory. Data Set is an image that:
        1)  is included in settings yaml at `.data.images`
        2)  is placed in a separate directory at
            working_directory/images/<image_name>

        Args:
            ignore (bool, optional): If True images listed in settings
                at `.data.exclude_images`. Defaults to True.
        """
        DSpaths = []
        ignore_list = []
        if ignore:
            ignore_list = self._settings["data"]["exclude_images"]

        DS_main_dir = self._workdir / "images"

        images = self._settings["data"]["images"]
        for i in images:
            if i not in ignore_list:
                img_dir = DS_main_dir / Path(images[i]["img_file"]).stem
                if img_dir.exists():
                    img_file = img_dir / Path(images[i]["img_file"])

                    # If img_file in settings is without extension
                    # find images with supported extensions
                    if img_file.suffix == "":
                        img_list = []
                        # Iterate over the supported extensions
                        for ext in SUPPORTED_IMG_FORMATS:
                            # Search for files with the specified name and extension
                            img_list.extend(
                                img_dir.glob(f"{img_file.name}.{ext}")
                            )
                        if len(img_list) > 1:
                            raise MultipleFilesFoundError(
                                filename=img_file.name,
                                matches=img_list,
                            )
                        elif len(img_list) == 0:
                            raise FileNotFoundError(
                                f"File {img_file} not ",
                                f"found in {img_dir}",
                            )
                        else:
                            img_file = img_list[0]
                            # update img_file in settings
                            self._settings["data"]["images"][i][
                                "img_file"
                            ] = img_file.name

                    DSpaths.append(img_file)

                else:
                    raise FileNotFoundError(f"Directory {img_dir} not found")
        return DSpaths


def PSD_space(start=0, end=10, periods=20, log=False, step=False):
    if log == True:
        bins_arr = np.logspace(
            int(np.floor(np.log10(start))),
            int(np.ceil(np.log10(end))),
            periods + 1,
        )
    elif step == True:
        bins_arr = np.arange(start, end + periods, periods)
    else:
        bins_arr = np.linspace(start, end, periods + 1)
    return bins_arr


def find_suported_images(img_name: str, supported_ext: List[str], dir: Path):
    matching_files = []

    # Iterate over the supported extensions
    for ext in SUPPORTED_IMG_FORMATS:
        # Use glob to search for files with the specified name and extension
        matching_files.extend(dir.glob(f"{img_name}{ext}"))
    return matching_files
