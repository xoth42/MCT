from setuptools import setup, find_packages

setup(
    name="mct",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sympy",
        "scipy",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "matplotlib", "jupyter", "ipykernel", "ruff"],
        "performance": ["numba"],  # Optional performance optimization
    },
    python_requires=">=3.9",
)
