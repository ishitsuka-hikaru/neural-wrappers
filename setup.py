import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
requirements = open("requirements.txt").read().splitlines()

setuptools.setup(
    name="neural_wrappers", # Replace with your own username
    version="0.4",
    author="Mihai Cristian Pîrvu",
    author_email="mihaicristianpirvu@gmail.com",
    description="Generic PyTorch high level wrapper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/mihaicristianpirvu/neural-wrappers",
    keywords = ["PyTorch", "neural network", "high level api"],
    packages=setuptools.find_packages(),
    install_requires=requirements,
    license="WTFPL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
