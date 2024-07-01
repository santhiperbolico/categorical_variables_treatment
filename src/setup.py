import os

import setuptools

SRC_DIR = os.path.abspath(os.path.dirname(__file__))

with open("../README.md", "r") as fh:
    long_description = fh.read()


def get_requirements():
    with open(os.path.join(SRC_DIR, "requirements.txt"), encoding="utf-8") as requirements_file:
        requirements = []
        for line in requirements_file.readlines():
            if line.startswith("-"):
                # Es un flag, no puede ir en `install_requires`
                continue
            requirements.append(line)
        return requirements


setuptools.setup(
    name="categorical_variables_treatment",
    version="0.0.1",
    author="Santiago Arranz Sanz",
    author_email="santiago.arranz.sanz@gmail.com",
    description="Libreria que investiga diferentes tratamientos de variables categoricas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/santhiperbolico/categorical_variables_treatment",
    install_requires=get_requirements(),
    packages=setuptools.find_packages(".", exclude=["tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
