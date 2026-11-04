"""Sphinx configuration file."""

from importlib.metadata import version as get_version

project = "coronalyze"
copyright = "2026, Corey Spohn"
author = "Corey Spohn"
release = get_version("coronalyze")
version = ".".join(release.split(".")[:2])  # e.g. "1.0" from "1.0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "skyscapes": ("https://skyscapes.readthedocs.io/en/latest/", None),
    "optixstuff": ("https://optixstuff.readthedocs.io/en/latest/", None),
    "coronagraphoto": ("https://coronagraphoto.readthedocs.io/en/latest/", None),
    "yippy": ("https://yippy.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", ".build", "Thumbs.db", ".DS_Store", "jupyter_execute"]

language = "en"

autoapi_dirs = ["../src"]
autoapi_ignore = ["**/*version.py"]
# Document objects at their defining modules. Re-exports such as core.snr
# otherwise collide with the module of the same name in Sphinx's index.
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autodoc_typehints = "description"
# AutoAPI already indexes fields; Napoleon should render their descriptions only.
napoleon_use_ivar = True

myst_enable_extensions = ["amsmath", "dollarmath"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = []
master_doc = "index"
html_title = "coronalyze - Post-Processing for Coronagraphic Imaging"
html_sidebars = {"posts/*": ["sbt-sidebar-nav.html"]}

html_theme_options = {
    "repository_url": "https://www.github.com/CoreySpohn/coronalyze",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_toc_level": 2,
}
html_context = {
    "default_mode": "dark",
}
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
nb_execution_mode = "auto"  # Execute notebooks without stored outputs
nb_execution_timeout = 120  # Timeout per cell in seconds
nb_execution_raise_on_error = True  # Fail build on notebook errors
