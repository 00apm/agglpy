from __future__ import annotations

from typing import Any, List, Literal, Tuple, Optional, ClassVar

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import constants # type: ignore

from agglpy.errors import AgglomerateStructureError 

# Particle and agglomerate types- variables used for typechecking
ParticleType = Literal["collector", "attached2coll", "separate", "similar"]
AgglomerateType = Literal["collector", "separate", "similar"]


class Agglomerate:
    """
    Agglomerate class creates agglomerate objects and defines its properties

    """

    AGL_ID: ClassVar[int] = 1

    # Attribute declarations for type checking
    # Public
    ID: int
    name: str
    members: List[Particle]
    member_IDs: List[int]
    type: AgglomerateType | None

    # Public attributes calculated during analysis
    volume: float
    volume_dsom: float
    D: float
    D_dsom: float
    members_count: float
    idj_members_count: float
    members_count_dsom: float
    members_Dmean: float
    members_Dstdev: float
    X_com: float
    Y_com: float
    Rg: float
    Dg: float

    # private
    _members_DF: pd.DataFrame | None

    def __init__(self, member_list: List[Particle] | None = None) -> None:
        self.ID = Agglomerate.AGL_ID
        self.name = "AGL" + str(Agglomerate.AGL_ID)
        if member_list is None:
            self.members = []
        else:
            self.members = member_list
        self.member_IDs = []

        self._members_DF = None
        self.type = None

        # Calculated attributes initialization
        self.volume = 0.0
        self.volume_dsom = 0.0
        self.D = 0.0
        self.D_dsom = 0.0
        self.members_count = 0.0
        self.idj_members_count = 0.0
        self.members_count_dsom = 0.0
        self.members_Dmean = 0.0
        self.members_Dstdev = 0.0
        self.X_com = 0.0
        self.Y_com = 0.0
        self.Rg = 0.0
        self.Dg = 0.0

        Agglomerate.AGL_ID = Agglomerate.AGL_ID + 1

    def append_particles(self, particle_list: List[Particle] = []) -> None:
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

    def calc_member_param(self) -> None:
        m_rows: List[Any] = []
        for m in self.members:
            m_rows.append(m.get_properties())
        self._members_DF = pd.DataFrame(data=m_rows)
        self._members_DF.loc[:, "OBJ"] = self.members
        self._members_DF["D-ratios"] = (
            self._members_DF.loc[:, "D"] / self._members_DF.loc[:, "D"].max()
        )
        self._members_DF.sort_values(by=["D"], ascending=False, inplace=True)

    def calc_agl_param(self, include_dsom: bool = False) -> None:

        if self._members_DF is None or self._members_DF.empty:
            raise AgglomerateStructureError(
                f"Calculation of Agglomerate {str(self)} is not possible. "
                "Agglomerate member DataFrame is not initialized or is empty"
            )
        self.members_count = len(self.members)
        self.read_members_IDs()
        # self._members_DF["membersOBJ"] = self.get_members()

        Ds = self._members_DF.loc[
            :, "D"
        ]  # pd.Series(self.get_members_Ds())
        self.members_Dmean = Ds.mean()
        self.members_Dstdev = Ds.std()
        self.volume = self._members_DF.loc[:, "volume"].sum()
        self.D = (6.0 * self.volume / constants.pi) ** (1 / 3)
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

    def classify(self, threshold: float = 0) -> None:
        if self._members_DF is None or self._members_DF.empty:
            raise AgglomerateStructureError(
                f"Classification of Agglomerate {str(self)} and its member "
                f"particles is not possible. Agglomerate member DataFrame is "
                f"not initialized or is empty."
            )
        self.set_affiliation_in_members()

        if self.members_count > 1:
            classif_condition = (
                self._members_DF.iloc[1, self._members_DF.columns.get_loc("D")] 
                / self._members_DF.iloc[0, self._members_DF.columns.get_loc("D")]
                <= threshold
            )
            sorted_members: List[Particle] = self._members_DF.loc[
                :, "OBJ"
            ].to_list()
            if classif_condition:
                # Collector type Agglomerate
                self.set_type("collector")  # set Agglomerate type
                for i, p in enumerate(sorted_members):
                    if i == 0:
                        p.set_type("collector")  # set largest Particle type
                    else:
                        p.set_type("attached2coll")  # set other Particle types
            else:
                # Similar type Agglomerate
                self.set_type("similar")
                for p in sorted_members:
                    p.set_type("similar")

        elif self.members_count == 1:
            # Agglomerate = Separate Primary Particle
            self.set_type("separate")
            p0: Particle = self._members_DF.loc[
                self._members_DF.index[0],
                "OBJ",
            ]
            p0.set_type("separate")

    def center_of_mass(self) -> Tuple[float, float]:
        """Calculate coordinates for center of mass of the agglomerate

        Takes particle members size and coordinates data and calculates
        the center of mass of the agglomerate.

        Returns:
            (int, int): tuple with agglomerate CoM coordinates (x_com, y_com)
                in pixels
        """
        if self._members_DF is None or self._members_DF.empty:
            raise AgglomerateStructureError(
                f"Calculation of Agglomerate {str(self)} center of mass is not "
                f"possible. Agglomerate member DataFrame is not initialized or "
                f"is empty"
            )
        DF: pd.DataFrame = self._members_DF.loc[:, ["area", "X", "Y"]]
        X_num: float = (DF.loc[:, "area"] * DF.loc[:, "X"]).sum()
        area_sum: float = DF.loc[:, "area"].sum()
        Y_num: float = (DF.loc[:, "area"] * DF.loc[:, "Y"]).sum()
        return (X_num / area_sum, Y_num / area_sum)

    def radius_of_gyration(self) -> float:
        """Calculate radius of gyration

        Calculate radius of gyration based on agglomerate center of mass
        and member particle coordinates

        Returns:
            (float): radius of gyration for the agglomerate

        """
        if self._members_DF is None or self._members_DF.empty:
            raise AgglomerateStructureError(
                f"Calculation of Agglomerate {str(self)} radius of gyration is "
                f"not possible. Agglomerate member DataFrame is not "
                f"initialized or is empty"
            )
        DF: pd.DataFrame = self._members_DF.loc[:, ["X", "Y", "area"]]

        # Calculate distance between each member particle center
        # and center of mass
        r_x_diff: pd.Series = (DF.loc[:, "X"] - self.X_com) ** 2
        r_y_diff: pd.Series = (DF.loc[:, "Y"] - self.Y_com) ** 2
        distance_squared: pd.Series = r_x_diff + r_y_diff

        # Calculate weighted sum of distances: sum(m_i * r_i^2)
        # mass is calculated from area sinve this is 2D approach
        wght_distance_sum: float = (DF.loc[:, "area"] * distance_squared).sum()

        total_area: float = DF.loc[:, "area"].sum()

        Rg: float = np.sqrt(wght_distance_sum / total_area)
        return Rg

    def count_idj_members(self) -> int:
        """Counts internally disjoint primary particles in this agglomerate"""
        if self._members_DF is None or self._members_DF.empty:
            raise AgglomerateStructureError(
                f"Counting the Agglomerates {str(self)} internally disjoint "
                f"members is not possible. Agglomerate member DataFrame is not "
                f"initialized or is empty"
            )
        return self._members_DF.loc[:, "idj"].sum()

    def read_members_IDs(self) -> None:
        m_IDs: List[int] = []
        for m in self.members:
            m_IDs.append(m.ID)
        self.member_IDs = m_IDs

    def get_members(self) -> List[Particle]:
        return self.members

    def get_members_IDs(self) -> List[int]:
        return self.member_IDs

    def get_members_Ds(self) -> List[float]:
        Ds: List[float] = []
        for p in self.members:
            Ds.append(p.D)
        return Ds

    def get_member_byID(self, ID: int) -> Particle:
        IDs: List[int] = self.get_members_IDs()
        assert ID in IDs, "Given ID is not a member of this Agglomerate"
        ix: int = IDs.index(ID)
        return self.members[ix]

    def get_properties(self) -> dict:
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
        return dict_prop

    def set_type(self, type_str: AgglomerateType) -> None:
        self.type = type_str

    def set_affiliation_in_members(self):
        for particle in self.members:
            particle.set_affiliation(self)

    def __repr__(self) -> str:
        class_name = type(self).__name__
        repr_str = (
            f"{class_name}(member_list= {[str(m) for m in self.members]})"
        )
        return repr_str

    def __str__(self) -> str:
        return self.name


class Particle:

    ID: int
    name: str
    X: float
    Y: float
    D: float
    area: float
    volume: float
    interIDs: List[int]  # List of inter-connected Particle IDs
    type: ParticleType | None  # Particle type
    affiliation: str | None  # Affiliated Agglomerate name
    idj: bool  # (i)nternally (d)is(j)oined bool

    def __init__(self, ID: int, X: float, Y: float, D: float) -> None:

        self.ID = ID
        self.name = "P" + str(int(self.ID))
        self.X = X
        self.Y = Y
        self.D = D
        self.area = constants.pi * ((self.D) / 2) ** 2
        self.volume = 4 / 3 * constants.pi * ((self.D) / 2) ** 3
        self.interIDs = []
        self.type = None
        self.affiliation = None
        self.idj = False

    def get_coord(self) -> Tuple[float, float]:
        return (self.X, self.Y)

    def get_D(self) -> float:
        return self.D

    def get_type(self) -> str | None:
        return self.type

    def get_affiliation(self) -> str | None:
        return self.affiliation

    def get_interIDs(self) -> List[int]:
        return self.interIDs

    def get_properties(self) -> dict:
        dict_prop = {
            "ID": self.ID,
            "name": self.name,
            "D": self.D,
            "X": self.X,
            "Y": self.Y,
            "area": self.area,
            "volume": self.volume,
            "type": self.type,
            "affiliation": self.affiliation,
            "idj": self.idj,
        }
        return dict_prop

    def set_type(self, type_str: ParticleType) -> None:
        self.type = type_str

    def set_affiliation(self, dst_aggl: Agglomerate) -> None:
        self.affiliation = dst_aggl.name

    def set_interIDs(self, ID_list: List[int] = []):
        self.interIDs = ID_list

    def set_idj(self, idj_bool: bool) -> None:
        self.idj = idj_bool

    def __repr__(self) -> str:
        class_name = type(self).__name__
        repr_str = (
            f"{class_name}(ID={self.ID}, X= {self.X}, Y= {self.Y}, "
            f"D= {self.D})"
        )
        return repr_str

    def __str__(self) -> str:
        return self.name
