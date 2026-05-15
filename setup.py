# setup.py

from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# Read the README file for the long description
README = (HERE / "README.md").read_text(encoding="utf-8")

# Read the requirements.txt file
requirements = (HERE / "requirements.txt").read_text().splitlines()


setup(
    name="sigmarl",
    version="1.2.0",
    author="Jianye Xu",
    author_email="xu@embedded.rwth-aachen.de",
    description="SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bassamlab/SigmaRL",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10, <3.13",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pre-commit==4.0.1",
            "pytest",
            "sphinx",
        ],
    },
    include_package_data=True,
    package_data={
        "sigmarl": ["assets/*", "scenarios/*", "modules/*", "config.json"],
    },
)
