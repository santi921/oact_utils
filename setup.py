from setuptools import setup, find_packages

setup(
    name="oact_utils",
    version="0.1.0",
    description="Utilities for ORCA and actinide chemistry workflows",
    author="Santiago Vargas",
    author_email="santiagovargas921@gmail.com",
    zip_safe=False,
    packages=find_packages(),
    # package_dir={"": "source"},
    install_requires=[],
    python_requires=">=3.9",
)
