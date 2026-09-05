# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import tomllib
import warnings
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock


# Add the source directory to the path
sys.path.insert(0, os.path.abspath("../src"))

# Suppress the ImportWarning emitted when CuPy is absent during docs builds.
warnings.filterwarnings("ignore", category=UserWarning, message="CuPy is not installed")


# Create a more robust mock for complex packages
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


# Mock modules that need special handling
sys.modules["icon_registration.losses"] = Mock()
sys.modules["icon_registration.network_wrappers"] = Mock()

# -- Project information -----------------------------------------------------
project = "MONAI Physio"
copyright = f"{datetime.now().year}, Stephen R. Aylward, NVIDIA Corporation"
author = "Stephen R. Aylward"
_repo_root = Path(__file__).resolve().parents[1]
_pyproject = tomllib.loads((_repo_root / "pyproject.toml").read_text(encoding="utf-8"))
_project_version = _pyproject["project"]["version"]

# The full version, including alpha/beta/rc tags
release = _project_version
version = _project_version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/nvidia-logo.svg"

html_theme_options = {
    "logo_only": False,
    # 'display_version': True,  # Deprecated in sphinx_rtd_theme >= 1.0
    "prev_next_buttons_location": "both",  # Show navigation at top and bottom
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#111111",
    # Toc options - optimized for module browsing
    "collapse_navigation": False,  # Keep sidebar expanded
    "sticky_navigation": True,  # Sidebar follows scroll
    "navigation_depth": 5,  # Increased depth for module pages
    "includehidden": True,
    "titles_only": False,  # Show full TOC hierarchy
}

# Enable search language features
html_search_language = "en"
html_search_options = {"type": "default"}

# Show page source link
html_show_sourcelink = True
html_copy_source = True

# Add breadcrumbs and navigation
html_use_index = True
html_split_index = True
html_show_sphinx = True

html_context = {
    "display_github": True,
    "github_user": "Project-MONAI",
    "github_repo": "monai-physio",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Add custom CSS
html_css_files = [
    "custom.css",
]

# -- Extension configuration -------------------------------------------------

# Napoleon settings (for NumPy and Google style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_inherit_docstrings = True
autodoc_warningiserror = False  # Don't treat import warnings as errors

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "itk": ("https://docs.itk.org/en/latest/", None),
}

# Todo extension
todo_include_todos = True

# MyST parser settings (for Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Mock imports for packages that require heavy dependencies or CUDA
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "torchaudio",
    "monai",
    "totalsegmentator",
    "icon_registration",
    "unigradicon",
    "vtk",
    "pyvista",
    "itk",
    "nibabel",
    "pynrrd",
    "transformers",
    "SimpleITK",
    "cupy",
    "cupyx",
    "pxr",
    "matplotlib",
    "ants",
    "antspyx",
    "cv2",
    "skimage",
    "PIL",
    "Usd",
    "UsdGeom",
    "UsdShade",
    "Gf",
    "Vt",
    "Sdf",
]

# Copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Custom setup ------------------------------------------------------------


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Custom function to skip certain members during autodoc processing."""
    # Skip private methods unless explicitly documented
    if name.startswith("_") and not name.startswith("__"):
        return True
    return skip


def _replace_project_version_token(_app, _docname, source):
    source[0] = source[0].replace("{{ mphysio_project_version }}", release)


def setup(app):
    """Custom setup function for Sphinx."""
    app.connect("source-read", _replace_project_version_token)

    # Connect the autodoc-skip-member event
    app.connect("autodoc-skip-member", autodoc_skip_member)

    # Suppress specific warnings
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
