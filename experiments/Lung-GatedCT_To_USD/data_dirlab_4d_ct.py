"""
class pmDataDirLab4dCT:
This module contains the pmDataDirLab4DCT class, which is used to store the
data for the DirLab 4DCT dataset.

DirLab-4DCT's raw ``.mhd``/``.img`` volumes are not in Hounsfield units; run
``data/DirLab-4DCT/fix_downloaded_data.py`` (backed by
``DataDownloadTools.FixDirLab4DCTData``) once to write corrected ``.mha``
volumes before using this dataset. That script's output is what this
directory's ``case_names`` are meant to be read from.
"""


class DataDirLab4DCT:
    """
    This class is used to store the data for the DirLab 4DCT dataset.
    """

    def __init__(self):
        """Define the variables specific to DirLab data"""
        self.case_names = [
            "Case1Pack",
            "Case2Pack",
            "Case3Pack",
            "Case4Pack",
            "Case5Pack",
            "Case6Pack",
            "Case7Pack",
            "Case8Deploy",
            "Case9Pack",
            "Case10Pack",
        ]

    def get_case_names(self) -> list[str]:
        """Get the case names"""
        return self.case_names
