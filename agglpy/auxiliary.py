from typing import Any
import os

import numpy as np



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


def RGB_convert_to256(color):
    c256 = ()
    for c in color:
        c256 = c256 + (int(c * 255),)
    return c256


def RGB_convert_to01(color):
    c01 = ()
    for c in color:
        c01 = c01 + (c / 255,)
    return c01


# =============================================================================
# def RGB_shader(color, factor):
#     c2 = ()
#
#
#     for i, c in enumerate(color):
#         if i<3:
#             c2 = c2 + (int(c * (1 - factor)),)
#         else:
#             c2 = c2 + (c,)
#     return c2
# =============================================================================


def RGB_shader(color, factor):
    if isinstance(color, tuple):
        wcol = np.full([1, 4], color)
        ret_tuple = True
    else:
        wcol = np.copy(color)
        ret_tuple = False
    wcol[:, :3] = wcol[:, :3] * (1 - factor)
    wcol[wcol > 255] = 255
    wcol[wcol < 0] = 0
    if ret_tuple:
        return tuple(wcol[0])
    else:
        return wcol


def isdefault(input: Any) -> bool:
    c1 = input is None
    c2 = txt_isdefault(input)
    c3 = txt_isnone(input)
    c4 = txt_isempty(input)
    return c1 or c2 or c3 or c4

def txt_istrue(txt):
    accepted_strings = ["true", "1", "t", "y", "yes", "yeah", "yup"]
    return txt in accepted_strings


def txt_isdefault(txt):
    accepted_strings = ["default", "def", "auto", "a", "d"]
    return txt in accepted_strings


def txt_isnone(txt):
    accepted_strings = ["none", "null", "nan"]
    return txt in accepted_strings

def txt_isempty(txt):
    return txt == ""

def txt_isnumber(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def data_switcher(list_of_paths, ON=True):
    for i in list_of_paths:
        # print(i)
        if ON == True:
            os.replace(i + os.sep + "OFF_names.csv", i + os.sep + "names.csv")
        else:
            os.replace(i + os.sep + "names.csv", i + os.sep + "OFF_names.csv")
