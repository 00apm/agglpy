from typing import List, Tuple

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from scipy import constants


class Agglomerate:
    """
    Agglomerate class creates agglomerate objects and defines its properties

    """

    AGL_ID = 1

    def __init__(self, member_list=[]):
        self.ID = Agglomerate.AGL_ID
        self.members = member_list
        self.member_IDs = []

        self.name = "AGL" + str(Agglomerate.AGL_ID)
        self._members_DF = pd.DataFrame()
        self.type = "not_defined"

        # Calculated attributes
        self.volume: float = 0
        self.volume_dsom: float = 0
        self.D: float = 0
        self.D_dsom: float = 0
        self.members_count: float = 0
        self.idj_members_count: float = 0
        self.members_count_dsom: float = 0
        self.members_Dmean: float = 0
        self.members_Dstdev: float = 0
        self.X_com: float = 0
        self.Y_com: float = 0
        self.Rg: float = 0
        self.Dg: float = 0

        Agglomerate.AGL_ID = Agglomerate.AGL_ID + 1

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
            "volume_dsom": self.volume_dsom,
            "D": self.D,
            "D_dsom": self.D_dsom,
            "X_com": self.X_com,
            "Y_com": self.Y_com,
            "Rg": self.Rg,
            "Dg": self.Dg,
            "type": self.type,
            "members_count": self.members_count,
            "idj_members_count": self.idj_members_count,
            "members_count_dsom": self.members_count_dsom,
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
                    self._members_DF.iloc[i].membersOBJ.type = type_sw_P.get(3)

        elif self.members_count == 1:
            self.type = type_sw_AGL.get(1)
            self._members_DF.iloc[0].membersOBJ.type = type_sw_P.get(2)

    def center_of_mass(
        self,
    ) -> Tuple[int]:
        """Calculate coordinates for center of mass of the agglomerate

        Takes particle members size and coordinates data and calculates
        the center of mass of the agglomerate.

        Returns:
            (int, int): tuple with agglomerate CoM coordinates (x_com, y_com)
                in pixels
        """
        DF = self._members_DF.loc[:, ["area", "X", "Y"]]
        X_num = (DF.loc[:, "area"] * DF.loc[:, "X"]).sum()
        area_sum = DF.loc[:, "area"].sum()
        Y_num = (DF.loc[:, "area"] * DF.loc[:, "Y"]).sum()
        return (X_num / area_sum, Y_num / area_sum)

    def radius_of_gyration(self) -> float:
        """Calculate radius of gyration

        Calculate radius of gyration based on agglomerate center of mass
        and member particle coordinates

        Returns:
            (float): radius of gyration for the agglomerate

        """
        DF = self._members_DF.loc[:, ["X", "Y", "area"]]

        # Calculate distance between each member particle center
        # and center of mass
        r_x_diff = (DF.loc[:, "X"] - self.X_com) ** 2
        r_y_diff = (DF.loc[:, "Y"] - self.Y_com) ** 2
        distance_squared = r_x_diff + r_y_diff

        # Calculate weighted sum of distances: sum(m_i * r_i^2)
        # mass is calculated from area sinve this is 2D approach
        wght_distance_sum = (DF.loc[:, "area"] * distance_squared).sum()

        total_area = DF.loc[:, "area"].sum()

        Rg = np.sqrt(wght_distance_sum / total_area)
        return Rg

    def count_idj_members(self):
        """Counts internally disjoint primary particles in this agglomerate"""
        return self._members_DF.loc[:, "idj"].sum()

    def calc_agl_param(self, include_dsom: bool = False):
        # print(self.members)
        for i in self.members:
            self.member_IDs.append(i.ID)
        self._members_DF["membersOBJ"] = self.get_members()
        self.members_count = len(self.members)
        Ds = self._members_DF.loc[:, "D"]  # pd.Series(self.get_members_Ds())
        self.members_Dmean = Ds.mean()
        self.members_Dstdev = Ds.std()
        self.volume = self._members_DF["volume"].sum()  
        self.D = (6 * self.volume / constants.pi) ** (1 / 3)
        self.X_com, self.Y_com = self.center_of_mass()
        self.Rg = self.radius_of_gyration()
        self.Dg = 2 * self.Rg
        if include_dsom:
            # Include 'dark side of the moon (dsom)' primary particles
            idj_map = self._members_DF.loc[:, "idj"] == True
            self.volume_dsom = (
                self.volume + self._members_DF.loc[idj_map, "volume"].sum()
            )
            self.D_dsom = (6 * self.volume_dsom / constants.pi) ** (1 / 3)
            self.idj_members_count = self.count_idj_members()
            self.members_count_dsom = (
                self.members_count + self.idj_members_count
            )

    def calc_member_param(self):
        m_rows = []
        for m in self.members:
            m_rows.append(m.get_properties(form="dict"))
        self._members_DF = pd.DataFrame.from_dict(m_rows, orient="columns")
        self._members_DF["membersOBJ"] = self.get_members()
        self._members_DF["D-ratios"] = (
            self._members_DF["D"] / self._members_DF["D"].max()
        )
        self._members_DF.sort_values(by=["D"], ascending=False, inplace=True)
        # print(self._members_DF)

    def __repr__(self):
        return self.name


class Particle:

    def __init__(self, ID, X, Y, D):

        self.ID: int = ID
        self.name: str = "P" + str(int(self.ID))
        self.X: float = X
        self.Y: float = Y
        self.D: float = D
        self.area: float = constants.pi * ((self.D) / 2) ** 2
        self.volume: float = 4 / 3 * constants.pi * ((self.D) / 2) ** 3
        self.interIDs: List[str] = None
        self.type: str = "not_defined"
        self.affil: str = "not_defined"
        self.idj: bool = False  # bool

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
        return self.interIDs

    def get_properties(self, form="dict"):
        dict_prop = {
            "ID": self.ID,
            "name": self.name,
            "D": self.D,
            "X": self.X,
            "Y": self.Y,
            "area": self.area,
            "volume": self.volume,
            "type": self.type,
            "affiliation": self.affil,
            "idj": self.idj,
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
        assert isinstance(affil, Agglomerate), (
            "given object "
            + str(affil)
            + "of type "
            + str(type(affil))
            + "is not a member of ImgAgl.Agglomerate class"
        )
        self.affil = affil.name

    def set_interIDs(self, ID_list=[]):
        self.interIDs = ID_list

    def set_idj(self, idj_bool: bool):
        self.idj = idj_bool
