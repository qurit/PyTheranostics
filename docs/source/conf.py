"""Sphinx configuration file for PyTheranostics documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyTheranostics"
copyright = "2024, Carlos Uribe"
author = "Carlos Uribe"

# The full version, including alpha/beta/rc tags. Read from the installed
# distribution rather than hard-coded, so a release never leaves the
# documentation claiming the previous version.
try:
    from importlib.metadata import version as _version

    release = _version("pytheranostics")
except Exception:  # PackageNotFoundError, or a bare checkout with no metadata
    release = "0.2.0"

version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_mock_imports = ["radiomics", "gatetools", "itk"]

# The name of the Pygments (syntax highlighting) style to use. Furo picks a
# different one per colour scheme, so both are set.
pygments_style = "friendly"
pygments_dark_style = "material"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"PyTheranostics {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/logo-light.svg"
html_copy_source = False
html_show_sphinx = False

html_theme_options = {
    # Two emblems, so the sidebar mark stays legible in either colour scheme.
    "light_logo": "logo-light.svg",
    "dark_logo": "logo-dark.svg",
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/qurit/PyTheranostics/",
    "source_branch": "dev",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#0f766e",
        "color-brand-content": "#0d7a70",
        "color-api-name": "#b45309",
        "color-api-pre-name": "#0f766e",
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Helvetica, Arial, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'SF Mono', Menlo, "
        "Consolas, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5eead4",
        "color-brand-content": "#7dd3fc",
        "color-api-name": "#fbbf24",
        "color-api-pre-name": "#5eead4",
    },
}

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pydicom": ("https://pydicom.github.io/pydicom/stable/", None),
}

# Shields and badge services rate-limit CI, and GitHub renders them on demand.
linkcheck_ignore = [
    r"https://img\.shields\.io/.*",
]

# -- Extension configuration -------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

nbsphinx_execute = "never"
nbsphinx_allow_errors = False
nbsphinx_codecell_lexer = "python"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "attrs_inline",
    "smartquotes",
]
myst_heading_anchors = 3

_CONTRIB_EXTENSION = "sphinxcontrib.contributors"
try:
    __import__(_CONTRIB_EXTENSION)
except ImportError:  # pragma: no cover - optional dependency
    _contributors_available = False
else:
    _contributors_available = True
    extensions.append(_CONTRIB_EXTENSION)


def setup(app):
    """Register a fallback contributors directive when the extension is missing."""
    if _contributors_available:
        return

    from docutils import nodes
    from docutils.parsers.rst import Directive

    class _ContributorsDirective(Directive):
        has_content = False
        required_arguments = 1

        def run(self):
            repo = self.arguments[0]
            paragraph = nodes.paragraph()
            paragraph += nodes.Text(
                "Install 'sphinx-contributors' to render the contributors list. "
                f"In the meantime see https://github.com/{repo}/graphs/contributors."
            )
            return [paragraph]

    app.add_directive("contributors", _ContributorsDirective)
