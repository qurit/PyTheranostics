import requests
from sphinx.util import logging

logger = logging.getLogger(__name__)


def fetch_github_contributors(app):
    """Fetch contributors via the GitHub API and write a simple RST list."""
    username = app.config.github_username
    repository = app.config.github_repository
    output_file = app.config.contributors_output_file

    if not username or not repository:
        logger.warning(
            "GitHub username or repository not configured. Skipping contributors fetch."
        )
        return

    url = f"https://api.github.com/repos/{username}/{repository}/contributors"
    response = requests.get(url)
    contributors = response.json()

    if "message" in contributors:
        logger.error(f"Error fetching contributors: {contributors['message']}")
        return

    contributors_list = []
    for contributor in contributors:
        contributors_list.append(
            f"- {contributor['login']} (contributions: {contributor['contributions']})"
        )

    contributors_text = "\n".join(contributors_list)

    with open(output_file, "w") as file:
        file.write("Contributors\n")
        file.write("============\n\n")
        file.write(contributors_text)

    logger.info(f"Contributors list written to {output_file}")


def setup(app):
    """Register config values and connect the fetch hook."""
    app.add_config_value("github_username", None, "env")
    app.add_config_value("github_repository", None, "env")
    app.add_config_value("contributors_output_file", "../contributors.rst", "env")
    app.connect("builder-inited", fetch_github_contributors)
