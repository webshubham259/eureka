from setuptools import setup, find_packages

setup(
    name="eureka",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "tqdm",
        "sympy"
    ]
)
