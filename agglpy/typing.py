from typing import (
    Literal,
    Mapping,
    TypeVar,
    TypedDict,
    Union,
    Dict,
    List,
    Any,
    Optional,
)


class ImageSettingsTypedDict(TypedDict, total=False):
    img_file: str
    HCT_file: Optional[str]
    magnification: Optional[Union[int, float]]
    pixel_size: Optional[Union[int, float]]
    crop_ratio: float
    median_blur: Optional[int]
    d_min: Union[List[int], int]
    d_max: Union[List[int], int]
    dist2R: Union[List[float], float]
    param1: Union[List[float | int], float, int]
    param2: Union[List[float | int], float, int]
    additional_info: Optional[str]


# Typed dict for Raw Image settings after YAML file loading
# possibly containing `<<:*default_img` YAML entries and none-like strings
ImageRawSettingsTypedDict = TypedDict(
    "ImageRawSettingsTypedDict",
    {
        "<<": str,  # Special key for YAML merging
        "img_file": Optional[str],
        "HCT_file": Optional[str],
        "magnification": Union[int, float, str],
        "pixel_size": Union[int, float, str],
        "crop_ratio": float,
        "median_blur": Optional[int],
        "d_min": Union[List[int], int],
        "d_max": Union[List[int], int],
        "dist2R": Union[List[float], float],
        "param1": Union[List[float | int], float, int],
        "param2": Union[List[float | int], float, int],
        "additional_info": Optional[str],
    },
    total = False
)


class DataSettingsTypedDict(TypedDict, total=False):
    default: ImageSettingsTypedDict
    images: Dict[str, ImageSettingsTypedDict]
    exclude_images: List[str]

class DataRawSettingsTypedDict(TypedDict, total=False):
    default: ImageRawSettingsTypedDict
    images: Dict[str, ImageRawSettingsTypedDict]
    exclude_images: List[str]


class GeneralSettingsTypedDict(TypedDict):
    working_dir: str


class MetadataSettingsTypedDict(TypedDict):
    conditions: Dict[str, Union[int, float]]

class PsdSpaceSettingsTypedDict(TypedDict):
    start: float
    end: float
    periods: int | float
    log: bool
    step: bool

class AnalysisSettingsTypedDict(TypedDict):
    PSD_space: PsdSpaceSettingsTypedDict | None
    collector_threshold: float


class ExportSettingsTypedDict(TypedDict):
    draw_particles: Dict[str, Union[bool, float]]


class YamlSettingsTypedDict(TypedDict):
    general: GeneralSettingsTypedDict
    metadata: MetadataSettingsTypedDict
    data: DataSettingsTypedDict
    analysis: AnalysisSettingsTypedDict
    export: ExportSettingsTypedDict

class YamlRawSettingsTypedDict(TypedDict):
    general: GeneralSettingsTypedDict
    metadata: MetadataSettingsTypedDict
    data: DataRawSettingsTypedDict
    analysis: AnalysisSettingsTypedDict
    export: ExportSettingsTypedDict

PPSouceCsvType = Literal["agglpy", "ImageJ", "agglpy_old"]
PreprocessFunction = Literal["median_blur",]
HctParameter = Literal["d_min", "d_max", "dist2R", "param1", "param2"]