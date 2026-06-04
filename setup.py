from setuptools import setup, find_packages

setup(
    name="icd_embeddings",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "pandas>=2.0",
        "numpy>=1.24",
        "pyarrow>=12.0",
        "streamlit>=1.32",
        "plotly>=5.0",
    ],
)
