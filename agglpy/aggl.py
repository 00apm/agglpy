import pandas as pd
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
        self.members_count = 0
        self.name = "AGL" + str(Agglomerate.AGL_ID)
        self._members_DF = pd.DataFrame()
        self.type = "not_defined"
        self.Xc = None
        self.Yc = None

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
                    self._members_DF.iloc[i].membersOBJ.type = type_sw_P.get(3)

        elif self.members_count == 1:
            self.type = type_sw_AGL.get(1)
            self._members_DF.iloc[0].membersOBJ.type = type_sw_P.get(2)

    def _calc_agl_param(self):
        # print(self.members)
        for i in self.members:
            self.member_IDs.append(i.ID)
        self._members_DF["membersOBJ"] = self.get_members()
        self.members_count = len(self.members)
        Ds = self._members_DF.loc[:, "D"]  # pd.Series(self.get_members_Ds())
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
        self._members_DF.sort_values(by=["D"], ascending=False, inplace=True)
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
        assert isinstance(affil, Agglomerate), (
            "given object "
            + str(affil)
            + "of type "
            + str(type(affil))
            + "is not a member of ImgAgl.Agglomerate class"
        )
        self.affil = affil.name

    def set_interIDs(self, list_=[]):
        self.interIDs = list_
