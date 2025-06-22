from setuptools import setup, find_packages

dependencies = []
for line in open("requirements.txt"):
    if line.strip() and not line.startswith("#"):
        dependencies += [line.strip()]

setup(
    name="qu3st",
    version="0.1.0",
    description="Qu3ST - Quantum Security Transaction Settlement Tool",
    author="Francesco Martini",
    author_email="f_martini+qc@outlook.it",
    packages=find_packages(),
    install_requires=dependencies,
    python_requires='>=3.11',
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
