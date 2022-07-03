try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import example_magic_widget, normalize_channel, preprocess, get_seg

__all__ = [
    "example_magic_widget",
    "normalize_channel",
    "preprocess",
    "get_seg"
]