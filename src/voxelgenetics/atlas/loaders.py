"""
voxelgenetics.atlas.loaders
---------------------------

Unified loaders for brain atlases (JHU, vbm, Tian).

This module provides:
1. A low-level `load_atlas(kind, nii_path, label_path)` that parses explicit files.
2. A high-level `load_atlas_by_name(name, data_dir=None)` that automatically finds
   standard atlas files in a predefined directory (defaults to ~/.voxelgenetics_data
   or the path specified by the VOXELGENETICS_ATLAS_DIR environment variable).
"""

import os
import nibabel as nib
import pandas as pd
import xmltodict


# Read the default data directory from environment variable, fallback to user home
DEFAULT_ATLAS_DIR = os.environ.get(
    "VOXELGENETICS_ATLAS_DIR", 
    os.path.join(os.path.expanduser("~"), ".voxelgenetics_data")
)


def load_atlas(kind, nii_path, label_path):
    """
    Load brain atlas data and its region dictionary by explicit file paths.

    Parameters
    ----------
    kind : str
        Type of the atlas. Must be one of {'jhu', 'vbm', 'tian'}.
    nii_path : str
        Path to the atlas .nii.gz file.
    label_path : str
        Path to the label metadata file (.xml or .txt depending on kind).

    Returns
    -------
    atlas_data : ndarray
        3D atlas label map.
    region_dict : dict
        Mapping from integer label to region name.
    """
    kind = kind.lower()
    if kind == "jhu":
        return _load_jhu(nii_path, label_path)
    elif kind == "vbm":
        return _load_vbm(nii_path, label_path)
    elif kind == "tian":
        return _load_tian(nii_path, label_path)
    else:
        raise ValueError(f"Unknown atlas kind: '{kind}'. Expected 'jhu', 'vbm', or 'tian'.")


def load_atlas_by_name(name, data_dir=None):
    """
    Load a standard atlas by its name, automatically locating files in `data_dir`.

    Parameters
    ----------
    name : str
        Name of the atlas: 'jhu', 'vbm', or 'tian'.
    data_dir : str, optional
        Directory containing the atlas files. If None, uses the environment
        variable VOXELGENETICS_ATLAS_DIR or falls back to ~/.voxelgenetics_data.

    Returns
    -------
    atlas_data : ndarray
        3D atlas label map.
    region_dict : dict
        Mapping from integer label to region name.
    """
    if data_dir is None:
        data_dir = DEFAULT_ATLAS_DIR

    name = name.lower()
    if name == "jhu":
        nii_path = os.path.join(data_dir, "JHU-ICBM-labels-2mm.nii.gz")
        label_path = os.path.join(data_dir, "JHU-labels.xml")
    elif name == "vbm":
        nii_path = os.path.join(data_dir, "UKB_T1_FAST_combined_atlas.nii.gz")
        label_path = os.path.join(data_dir, "UKB_T1_FAST_combined_atlas_labels.txt")
    elif name == "tian":
        nii_path = os.path.join(data_dir, "Tian_Subcortex_S1_3T.nii.gz")
        label_path = os.path.join(data_dir, "Tian_Subcortex_S1_3T_label.txt")
    else:
        raise ValueError(f"Unknown atlas name: '{name}'. Expected 'jhu', 'vbm', or 'tian'.")

    if not os.path.exists(nii_path):
        raise FileNotFoundError(
            f"Atlas file not found: {nii_path}\n"
            f"Please ensure the file exists, or set the VOXELGENETICS_ATLAS_DIR "
            f"environment variable to the directory containing your atlas files."
        )
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Atlas label file not found: {label_path}")

    return load_atlas(name, nii_path, label_path)


# ---------------------------------------------------------------------------
# Private loaders for specific formats
# ---------------------------------------------------------------------------

def _load_jhu(nii_path, xml_path):
    """Load JHU atlas (XML format)."""
    img = nib.load(nii_path)
    data = img.get_fdata()

    with open(xml_path, "r") as f:
        parsed = xmltodict.parse(f.read())

    region_dict = {}
    labels = parsed["atlas"]["data"]["label"]

    # Handle cases where there is only one label (xmltodict returns dict instead of list)
    if isinstance(labels, dict):
        labels = [labels]

    for label in labels:
        idx = int(label.get("@index") or label.get("index"))
        name = label.get("name") or label.get("#text")
        region_dict[idx] = name

    return data, region_dict


def _load_vbm(nii_path, txt_path):
    """Load vbm atlas (TXT format)."""
    img = nib.load(nii_path)
    data = img.get_fdata()
    label_df = pd.read_csv(txt_path, sep="\t")

    indexs = label_df["New_Label"].values
    labels = label_df["Atlas_Label"].values

    region_dict = dict(zip(indexs, labels))
    return data, region_dict


def _load_tian(nii_path, txt_path):
    """Load Tian subcortex atlas (TXT format, no header)."""
    img = nib.load(nii_path)
    data = img.get_fdata()
    label_df = pd.read_csv(txt_path, sep="\t", header=None)

    # Tian atlas labels are usually 1-indexed matching the order in the txt file
    indexs = label_df.index + 1
    labels = label_df[0].values

    region_dict = dict(zip(indexs, labels))
    return data, region_dict