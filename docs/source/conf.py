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

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "myst_parser",
    "nbsphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_mock_imports = ["radiomics", "gatetools", "itk"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

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
]

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
