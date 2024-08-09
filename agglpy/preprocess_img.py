# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 00:07:19 2020

@author: Artur
"""

import subprocess
import os
import pandas as pd
from tqdm import tqdm
import warnings

cfg_file = "names.csv"

def create_settings_file(imgpath):
    """
    Function creates setting files for individual image to analyze.
    Setting files consists of:
        - image name,
        - name of file containing circle fitting data,
        - magnification,
        - Hough transform parameters
        - additional info.

    Parameters
    ----------
    imgpath : str
        Absolute path of image for analysis setting file creation.

    Returns
    -------
    None.

    """
    imgname = os.path.basename(imgpath)
    sett_dict = {"img_name": [imgname],
                 "fitting_file_name": [imgname.split('.')[0]+"_fitting.csv"],
                 "magnification": ["auto"],
                 "Dmin": [3],
                 "Dmax": [100],
                 "dist2R": [0.4],
                 "param1": [250],
                 "param2": [15],
                 "additional_info": ["-"]
                 }
    sett_DF = pd.DataFrame(sett_dict)
    dpath = os.path.dirname(imgpath)
    cfg_file = generate_filename(dpath, "_names.csv")
    sett_DF.T.to_csv(os.path.dirname(imgpath) + os.sep + cfg_file,
                     sep=";",
                     header=False)



def prepare_imgset_struc(path):
    """
    Function converts folder containing several SEM image files to
    Image Set structure for Particle detection and agglomerates analysis.
    It creates folder for each SEM image and creates necessary setting file
    for each image. 
    When SEM images are already in seperate folders named exactly like 
    SEM image file names, function only creates settings files for each image.

    Parameters
    ----------
    path : str (path-like)
        Absolute path to Image Set considered.

    Raises
    ------
    FileNotFoundError
        Error raises when path directory already have separate folder for
        each SEM image, but images are not found in those folders.

    Returns
    -------
    None.

    """
    if not imgset_struc_iscorrect(path):
        imgnames = []
        for dirpath, dirs, files in os.walk(path):
            for name in files:
                if ".tif" in name.lower():
                    imgnames.append(name.split(".")[0])
        for n in imgnames:
            newimgdir = path + os.sep + n
            newimgpath = newimgdir + os.sep + n + ".tif"
            if not os.path.exists(newimgdir):
                os.makedirs(newimgdir)
                os.rename(newimgdir + ".tif", newimgpath)
                create_settings_file(newimgpath)
            else:
                if os.path.exists(newimgpath):
                    create_settings_file(newimgpath)
                else:
                    raise FileNotFoundError
        ignoreDF = pd.DataFrame(list())
        ignoreDF.to_csv(path + os.sep + "ignore.csv")
        print("Operation complete.")
    else:
        print("IMGSET folder structure is already prepared.")


def reset_imgset_struct(path):
    if imgset_struc_iscorrect(path):

        msg = "Do you want to revert SEM IMAGE SET structure?"\
            " Following action may lead to irreversible loss of data."\
            "\nDo you want to continue [Y/N]? "
        if confirm(msg):
            imgpaths = []
            imgnames = []
            for dirpath, dirs, files in os.walk(path):
                for name in files:
                    name_woext = os.path.splitext(
                        os.path.basename(dirpath+os.sep+name)
                        )[0]
                    if name_woext == os.path.basename(dirpath):
                        print(name," file moved.")
                        os.rename(dirpath + os.sep + name, path + os.sep + name)
                    else:
                        os.remove(dirpath + os.sep + name)
                        print(name," file removed.")
                if not dirpath == path:
                    os.rmdir(dirpath)
                    print(os.path.basename(dirpath)," directory removed.")

    else:
        raise Exception("Unrecognized folder structure.")


def imgset_struc_iscorrect(path):
    assert os.path.exists(path), "Given path does not exist."
    for dirpath, dirs, files in os.walk(path):

        for name in files:
            if ".tif" in name.lower():
                valid_dirname = name.split(".")[0]
                valid_path = path + os.sep + valid_dirname + os.sep + name

                if not os.path.isfile(valid_path):
                    return False
    return True


def find_DataSets(path, ignore=True):
    DSpaths = []
    
    if ignore:
        ignorepath = path + os.sep + "ignore.csv"
        ig = pd.read_csv(ignorepath,
                         sep=";",
                         usecols=[0],
                         header=None,
                         squeeze=True,
                         ).astype(str)
    else:
        ig = pd.Series([";"],dtype="object")

    for dirpath, dirs, files in os.walk(path):
        counter = 0
        for name in files:
            if dirpath != path:
                cfg_file = generate_filename(dirpath, "_names.csv")
                isin_ig = ig.str.contains(name.split(".")[0]).any()
                if name == cfg_file:
                    counter += 1
                if name.split(".")[0] == os.path.basename(dirpath):
                    counter += 1
                    if not isin_ig:
                        counter += 1
        if counter == 3:
            DSpaths.append(dirpath)
    return DSpaths


def load_sett(cfg):
    sett = pd.read_csv(cfg,
                       sep=";",
                       encoding="ansi",
                       index_col=0,
                       header=None,
                       squeeze=True
                       )
    return (sett["img_name"],
            sett["magnification"],
            sett["Dmin"],
            sett["Dmax"])


def activate_images(path, imgnames=None):
    ignorepath = path + os.sep + "ignore.csv"
    try:
        ignore = pd.read_csv(ignorepath,
                             sep=";",
                             usecols=[0],
                             header=None,
                             )
        ignore = ignore.squeeze('columns')
        
    except FileNotFoundError:
        ignore = pd.DataFrame(list(";"))
        ignore.to_csv(ignorepath, header=False, index=False)
    
    # read available image sets
    DSpaths = find_DataSets(path, ignore=False)
    DS = []
    for DSp in DSpaths:
        DS.append(os.path.basename(DSp))
    DS = pd.Series(DS, dtype=object)
    
    if imgnames:
        ignore = ignore.loc[~ignore.isin(imgnames)]
    else:
        ignore = pd.DataFrame(list(";"))
    
    # inform user which data sets are active
    activated = DS.loc[~DS.isin(ignore)]
    print("\nCurrently activated data sets: \n"
          + str(activated.to_string(index=False, header=False).split('\n'))
          + "\n")
    
    # create ignore.csv file
    if ignore.size != 0:
        ignore.to_csv(ignorepath, header=False, index=False)
    else:
        with open(ignorepath, "w") as csv:
            csv.write(";")


def deactivate_images(path, imgnames=None):
    ignorepath = path + os.sep + "ignore.csv"

    # read current state of ignore.csv file
    try:
        ignore = pd.read_csv(ignorepath,
                             sep=";",
                             usecols=[0],
                             header=None,
                             )
        ignore = ignore.squeeze('columns')
    except FileNotFoundError:
        ignore = pd.DataFrame(list(";"))
        ignore.to_csv(ignorepath, header=False, index=False)
    
    # read available image sets
    DSpaths = find_DataSets(path, ignore=False)
    DS = []
    for DSp in DSpaths:
        DS.append(os.path.basename(DSp))
    DS = pd.Series(DS, dtype=object)
    
    # append imgnames that are available in folder structure
    if imgnames:
        imgnames = pd.Series(imgnames, dtype=object)
        imgappend = imgnames.loc[imgnames.isin(DS)&~imgnames.isin(ignore)]
        ignore = ignore.append(imgappend)
        ignore = ignore.dropna()
        
        # warn user that some imgnames were not present in folder structure
        notinDS = imgnames.loc[~imgnames.isin(DS)]
        notinDS = notinDS.dropna()

        if not notinDS.empty:
            warnings.warn("Following data sets are not present in SEM IMG folder"
                          " structure: "
                          + str(notinDS.to_string(index=False).split('\n')))
    else:
        ignore = pd.DataFrame(DS)

    # inform user which data sets are active
    activated = DS.loc[DS.isin(ignore)].dropna()
    if not activated.empty:
        print("\nCurrently activated data sets: \n"
              + str(activated.to_string(index=False, header=False).split('\n'))
              + "\n")
    else:
        print("\nAll data sets are currently deactivated\n")
    ignore.to_csv(ignorepath, header=False, index=False)


def active_list(path):
    ignorepath = path + os.sep + "ignore.csv"

    # read current state of ignore.csv file
    ignore = pd.read_csv(ignorepath,
                         sep=";",
                         usecols=[0],
                         header=None,
                         squeeze=True,
                         )
    DSpaths = find_DataSets(path, ignore=False)
    DS = []
    for DSp in DSpaths:
        DS.append(os.path.basename(DSp))
    DS = pd.Series(DS, dtype=object)
    
    
    activated = DS.loc[~DS.isin(ignore)].dropna()
    if not activated.empty:
        print("\nCurrently activated data sets: \n"
              + str(activated.to_string(index=False, header=False).split('\n'))
              + "\n")
    else:
        print("\nAll data sets are currently deactivated\n")
    return activated

def generate_filename(path, suffix):
    return os.path.basename(path) + suffix

def confirm(msg):
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(msg).lower()
    return answer == "y"

#
def prepare_imgs(IJpath, DSpaths):
    pbar = tqdm(DSpaths, total = len(DSpaths), position=0, leave=True)
    for path in pbar:
        # print(path)
        s = load_sett(path + os.sep + cfg_file)
        imgpath = path + os.sep + s[0]

        command =   IJpath + """ --ij2 --headless --console -Dpython.console.encoding=UTF-8 --run "C:\\DATA\\! IMP\\Oprogramowanie i sprzet PC\\ImageJ\\macros\\im_prep.py" "path='""" + imgpath + "', magn=" + s[1] + """" """
# command = "D:\SOFTWARE\Program Files\fiji-win64\Fiji.app\ImageJ-win64.exe --ij2 --headless --console --run ""C:\DATA\! IMP\Oprogramowanie i sprzet PC\ImageJ\macros\im_prep.py"" ""path='C:\\DATA\! IMP\Doktorat\\badania i obliczenia\\analiza aglomeratow SEM\\laziska\4\charger-02.tif', magn=5000"""
        print(command)
        subprocess.check_output(command)

def runHCT(IJpath, DSpaths):

    pbar = tqdm(DSpaths, total = len(DSpaths), position=0, leave=True)
    for path in pbar:
        s = load_sett(path + os.sep + cfg_file)
        original_imgpath = path + os.sep + s[0]
        name = os.path.basename(original_imgpath).split(".")[0]
        imgpath = os.path.dirname(original_imgpath) + os.sep + name + "_edge.png"
        # print(imgpath)

        defaults = {"minR":3,
                    "maxR":150,
                    "maxCir":750,
                    "res_":50,
                    "threshold_":0.45,
                    "ratio_":0.98,
                    }
        #@ String (label='Image path') path
        #@ Integer (label='min Radius in px',value=10.0) minR
        #@ Integer (label='max Radius in px',value=150.0) maxR
        #@ Integer (label='max circle number',value=10) maxCir
        #@ Integer (label='hough transform resolution',value=500) res_
        #@ Float (label='score threshold',value=0.5) threshold_
        #@ Float (label='neighbour search ratio',value=0.98) ratio_
        command = IJpath + """ --console -Dpython.console.encoding=UTF-8 --run "C:\\DATA\\! IMP\\Oprogramowanie i sprzet PC\\ImageJ\\macros\\auto_HCT.py" "path='""" + imgpath \
                    + "', minR=" + str(defaults["minR"])\
                    + ", maxR=" + str(defaults["maxR"])\
                    + ", maxCir=" + str(defaults["maxCir"])\
                    + ", res_=" + str(defaults["res_"])\
                    + ", threshold_=" + str(defaults["threshold_"])\
                    + ", ratio_=" + str(defaults["ratio_"])\
                    + """" """
        # print(command)
        subprocess.check_output(command)




if __name__ == "__main__":
    IJpath = "D:\\SOFTWARE\\Program Files\\fiji-win64\\Fiji.app\\ImageJ-win64.exe"
    dirpath = "C:\\DATA\\! IMP\\#Projekty Badawcze\\#HYBRYDA+\\wyniki badan\\"\
        "LAZISKA\\SEM\\analiza_aglomeratów\\1_spaliny\\ALL_OFF\\st15"
    setpath = "C:\\DATA\\! IMP\\#Projekty Badawcze\\#HYBRYDA+\\wyniki badan\\"\
        "LAZISKA\\SEM\\analiza_aglomeratów\\1_spaliny\\ALL_OFF\\"

    wdir2 = "C:\\DATA\\! IMP\\Doktorat\\badania i obliczenia\\analiza aglomeratow SEM\\agl_doz & podsysanie\\1str\\AGL15kV\\"
    wdir3 = "C:\\DATA\\! IMP\\Doktorat\\badania i obliczenia\\analiza aglomeratow SEM\\agl_doz & podsysanie\\1str\\ALLoff\\"
    # print(dirpath)
    # prepare_folder_struc(dirpath)
    # create_settings_file('C:\\DATA\\! IMP\\Doktorat\\badania i obliczenia\\analiza aglomeratow SEM\\ELPI 1min\\stage06\\stage06-15\\stage06-15.tif')
    # prepare_imgs(IJpath, find_DataSets(dirpath))
    # runHCT(IJpath, find_DataSets(dirpath))

    # print(IJpath)
