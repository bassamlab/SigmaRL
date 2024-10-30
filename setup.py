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
    name="sigmarl",  # Required
    version="1.2.0",  # Required
    author="Jianye Xu",  # Replace with your name
    author_email="xu@embedded.rwth-aachen.de",  # Replace with your email
    description="SigmaRL: A Sample-Efficient and Generalizable Multi-Agent Reinforcement Learning Framework for Motion Planning",  # Replace with a short description
    long_description=README,  # Long description read from README.md
    long_description_content_type="text/markdown",
    url="https://github.com/cas-lab-munich/SigmaRL",  # Replace with your repo URL
    packages=find_packages(),  # Automatically find packages in the directory
    classifiers=[  # Optional classifiers
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: MIT License",  # Change if you use a different license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9, <3.11",  # Specify your Python version
    install_requires=[
        "gym==0.26.2",
        "numpy==1.26.1",
        "tensordict==0.2.1",
        "torch==2.1.0",
        "torchrl==0.2.1",
        "tqdm==4.66.1",
        "typing_extensions==4.8.0",
        "vmas==1.4.1",
        "matplotlib==3.8.2",
        "termcolor==2.4.0",
        "SciencePlots==2.1.1",
        "pre-commit==3.7.1",
    ],
    extras_require={  # Optional
        "dev": ["pytest", "sphinx"],  # Development dependencies
    },
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    package_data={  # Include additional files within packages
        "sigmarl": ["assets/*", "scenarios/*", "config.json"],  # Adjust as needed
    },
)
