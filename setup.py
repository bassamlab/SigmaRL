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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9, <3.11",
    install_requires=[
        "numpy==1.26.1",
        "matplotlib==3.8.2",
        "gym==0.26.2",
        "tensordict==0.2.1",
        "torch==2.1.0",
        "torchrl==0.2.1",
        "torchdiffeq==0.2.4",
        "tqdm==4.66.1",
        "typing_extensions==4.8.0",
        "vmas==1.4.1",
        "termcolor==2.4.0",
        "pre-commit==3.7.1",
        "cvxpy==1.5.3",
        "scikit-learn==1.5.2",
    ],
    extras_require={
        "dev": ["pytest", "sphinx"],
    },
    include_package_data=True,
    package_data={
        "sigmarl": ["assets/*", "scenarios/*", "config.json"],
    },
)
