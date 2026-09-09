"""
Tools for reading and writing anatomical landmarks.

This module provides the :class:`LandmarkTools` class with utilities for
reading and writing point landmarks in 3D Slicer's Markups JSON
(``.mrk.json``) format and in a simple CSV format. Landmarks are kept in
memory in LPS world coordinates (ITK's native frame, matching the rest of
the platform). RAS files are converted to LPS on read; outputs are always
written in LPS.
"""

import csv
import json
import logging
from pathlib import Path

from .monai_physio_base import MONAIPhysioBase

LandmarkDict = dict[str, tuple[float, float, float]]

# Slicer Markups JSON schema URL emitted in the file envelope on write.
_MRK_JSON_SCHEMA = (
    "https://raw.githubusercontent.com/Slicer/Slicer/main/"
    "Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#"
)


class LandmarkTools(MONAIPhysioBase):
    """
    Read and write anatomical landmarks in LPS world coordinates.

    Landmarks are represented in memory as a dictionary keyed by label,
    with each value a three-tuple ``(x, y, z)`` of LPS millimeter
    coordinates::

        {
            'apex': (x, y, z),
            'base': (x, y, z),
            ...
        }

    Positions are always in LPS. RAS input files are converted on read;
    outputs are always written in LPS.

    Example:
        >>> tools = LandmarkTools()
        >>> landmarks = tools.read_landmarks_3dslicer('points.mrk.json')
        >>> tools.write_landmarks_csv(landmarks, 'points.csv')
    """

    def __init__(self, log_level: int | str = logging.INFO):
        """Initialize the LandmarkTools class.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

    def read_landmarks_3dslicer(self, path: str | Path) -> LandmarkDict:
        """Read landmarks from a 3D Slicer Markups JSON (``.mrk.json``) file.

        Reads the first markup node from the file and returns its control
        points as a ``{label: (x, y, z)}`` dictionary. Other Slicer fields
        (``id``, ``description``, ``orientation``, ...) are discarded.

        Coordinates are returned in LPS. If the file declares RAS (or the
        legacy numeric codes ``'0'`` for LPS and ``'1'`` for RAS), each
        position is converted by negating its X and Y components.

        Args:
            path: Path to the ``.mrk.json`` file.

        Returns:
            Dict mapping landmark label to ``(x, y, z)`` tuple in LPS.

        Raises:
            ValueError: If the file contains no markups, declares an
                unrecognized coordinate system, or has a control point
                without a 3D position.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        markups = data.get("markups", [])
        if not markups:
            raise ValueError(f"No markups found in {path}")
        markup = markups[0]

        coord_sys = str(markup.get("coordinateSystem", "LPS")).upper()
        if coord_sys in ("RAS", "1"):
            flip = True
        elif coord_sys in ("LPS", "0"):
            flip = False
        else:
            raise ValueError(f"Unrecognized coordinateSystem {coord_sys!r} in {path}")

        landmarks: LandmarkDict = {}
        for cp in markup.get("controlPoints", []):
            pos = cp.get("position")
            label = cp.get("label", "")
            if pos is None or len(pos) < 3:
                raise ValueError(
                    f"Control point {label!r} in {path} has no 3D position"
                )
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            if flip:
                x, y = -x, -y
            landmarks[label] = (x, y, z)

        return landmarks

    def write_landmarks_3dslicer(
        self, landmarks: LandmarkDict, path: str | Path
    ) -> None:
        """Write landmarks to a 3D Slicer Markups JSON file in LPS.

        Wraps the landmarks in the Slicer Markups schema envelope and
        writes the result to disk. The output always declares
        ``coordinateSystem == 'LPS'``; positions are written verbatim, so
        the caller must ensure they are already in LPS.

        Args:
            landmarks: Dict mapping label to ``(x, y, z)`` tuple, as
                returned by :meth:`read_landmarks_3dslicer` or
                :meth:`read_landmarks_csv`.
            path: Output ``.mrk.json`` path.
        """
        control_points = [
            {
                "label": label,
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            }
            for label, pos in landmarks.items()
        ]
        markup = {
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "coordinateUnits": "mm",
            "controlPoints": control_points,
        }
        data = {
            "@schema": _MRK_JSON_SCHEMA,
            "markups": [markup],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def read_landmarks_csv(self, path: str | Path) -> LandmarkDict:
        """Read landmarks from a CSV file with header ``Name,x,y,z`` (LPS).

        Coordinates are assumed to be in LPS. The returned dictionary
        matches the in-memory format used by
        :meth:`read_landmarks_3dslicer`, so the readers and writers are
        interchangeable.

        Args:
            path: Path to the CSV file. The first row must be the header
                ``Name,x,y,z`` (case-insensitive, surrounding whitespace
                tolerated); subsequent rows are ``label,x,y,z``.

        Returns:
            Dict mapping landmark label to ``(x, y, z)`` tuple in LPS.

        Raises:
            ValueError: If the file is empty, has the wrong header, or
                contains a malformed row.
        """
        landmarks: LandmarkDict = {}
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"Empty CSV file: {path}")
            normalized = [h.strip().lower() for h in header]
            if normalized[:4] != ["name", "x", "y", "z"]:
                raise ValueError(
                    f'Expected header "Name,x,y,z" in {path}, got {header!r}'
                )
            for row in reader:
                if not row or all(not c.strip() for c in row):
                    continue
                if len(row) < 4:
                    raise ValueError(f"Malformed landmark row in {path}: {row!r}")
                landmarks[row[0].strip()] = (
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                )

        return landmarks

    def write_landmarks_csv(self, landmarks: LandmarkDict, path: str | Path) -> None:
        """Write landmarks to a CSV file with header ``Name,x,y,z`` (LPS).

        Positions are written verbatim and assumed to be in LPS.

        Args:
            landmarks: Dict mapping label to ``(x, y, z)`` tuple.
            path: Output CSV path.
        """
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "x", "y", "z"])
            for label, pos in landmarks.items():
                writer.writerow([label, pos[0], pos[1], pos[2]])
