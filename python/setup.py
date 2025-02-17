from setuptools import setup, find_packages

setup(
    name="goose",
    version="0.1",
    packages=find_packages(), 
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "seaborn",
    ],
    author="Dr. Jordan JA Weaver",
    description="Global Optimization and ODE Solving Extension, goose.",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Developed on Windows, please report Linux/MAC specific problems",
    ],
    python_requires=">=3.12.3",
)
