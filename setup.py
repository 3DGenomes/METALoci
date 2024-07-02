import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely."""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        return open_file.read().strip()

def read_requirements(path):
    """Parse requirements file and ignore certain types of lines."""
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]

setup(
    name="metaloci",
    version=read("metaloci", "VERSION"),  # Assuming VERSION file exists in metaloci directory
    description="METALoci: spatially auto-correlated signals in 3D genomes",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/3DGenomes/METALoci/",
    author="Leo Zuber, Iago Maceda, Juan Antonio Rodríguez and Marc Martí-Renom",
    author_email="martirenom@cnag.eu",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: POSIX :: Linux",
    ],
    packages=find_packages(exclude=["tests", ".github"]),
    package_data={
        "metaloci": [
            "tests/data/*",
            "tests/data/hic/*.mcool",
            "tests/data/signal/*.bed"
        ]
    },
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "metaloci = metaloci.__main__:main"
        ]
    },
    # extras_require={"test": read_requirements("requirements-test.txt")},
)
