"""Data management utilities for coronablink.

This module provides utilities for managing and accessing example data files
used by coronablink. It uses pooch to handle data downloads and caching.
"""

import pooch
from pooch import Unzip

# Create a pooch registry for data files
REGISTRY = {
    "coronagraphs.zip": "md5:1537f41c20cb10170537a7d4e89f64b2",
    "scenes.zip": "md5:c777aefb65887249892093b1aba6d86a",
}

# Create a pooch instance for coronablink example data
PIKACHU = pooch.create(
    path=pooch.os_cache("coronablink"),
    base_url="https://github.com/CoreySpohn/coronablink/raw/main/data/",
    registry=REGISTRY,
)


def fetch_coronagraph() -> str:
    """Fetch and unpack example coronagraph data.

    Downloads the eac1_aavc_512 coronagraph (apodized vortex) for use
    with yippy and coronagraphoto.

    Returns:
        Path to the coronagraph directory.
    """
    PIKACHU.fetch("coronagraphs.zip", processor=Unzip())
    return str(
        PIKACHU.abspath / "coronagraphs.zip.unzip" / "coronagraphs" / "eac1_aavc_512"
    )


def fetch_scene() -> str:
    """Fetch and unpack example ExoVista scene data.

    Downloads a modified Solar System scene for demonstration.

    Returns:
        Path to the ExoVista FITS file.
    """
    PIKACHU.fetch("scenes.zip", processor=Unzip())
    return str(
        PIKACHU.abspath / "scenes.zip.unzip" / "scenes" / "solar_system_mod.fits"
    )


def fetch_all() -> tuple[str, str]:
    """Fetch all example data.

    Returns:
        Tuple of (coronagraph_path, scene_path).
    """
    return fetch_coronagraph(), fetch_scene()
