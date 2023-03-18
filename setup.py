#!/usr/bin/env python
# -*- coding: utf-8 -*-

import versioneer
import codecs
import os
import re
from setuptools import find_packages, setup

NAME = "jaxlinop"

# Handle builds of nightly release - adapted from BlackJax.
if "BUILD_JAXLINOP_NIGHTLY" in os.environ:
    if os.environ["BUILD_JAXLINOP_NIGHTLY"] == "nightly":
        NAME += "-nightly"

        from versioneer import get_versions as original_get_versions

        def get_versions():
            from datetime import datetime, timezone

            suffix = datetime.now(timezone.utc).strftime(r".dev%Y%m%d")
            versions = original_get_versions()
            versions["version"] = versions["version"].split("+")[0] + suffix
            return versions

        versioneer.get_versions = get_versions

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
]

INSTALL_REQUIRES = [
    "jax>=0.4.1",
    "jaxlib>=0.4.1",
    "jaxtyping",
    "simple-pytree",
]

EXTRA_REQUIRE = {
    "dev": [
        "pytest",
        "pre-commit",
        "pytest-cov",
    ],
}

GLOBAL_PATH = os.path.dirname(os.path.realpath(__file__))


def read(*local_path: str) -> str:
    """Read a file, given a local path.

    Args:
        *local_path (str): The local path to the file.

    Returns:
        str: The contents of the file.
    """
    with codecs.open(os.path.join(GLOBAL_PATH, *local_path), "rb", "utf-8") as f:
        return f.read()


# Read `__init__` file:
init_local_path = os.path.join('jaxlinop', "__init__.py")
init_file = read(init_local_path)


def find_meta(meta: str) -> str:
    """Extract `__*meta*__` from the `__init__.py` file. This is useful for extracting __version__, __author__, etc.

    Args:
        meta (str): The meta to extract.

    Returns:
        str: The meta.
    """

    matches = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), init_file, re.M
    )
    if matches:
        return matches.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        author=find_meta("authors"),
        author_email=find_meta("emails"),
        maintainer=find_meta("authors"),
        maintainer_email=find_meta("emails"),
        url=find_meta("uri"),
        license=find_meta("license"),
        description=find_meta("description"),
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        packages=find_packages(".", exclude=["tests"]),
        package_data={NAME: ["py.typed"]},
        include_package_data=True,
        python_requires=">=3.6",
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRA_REQUIRE,
        zip_safe=True,
    )
