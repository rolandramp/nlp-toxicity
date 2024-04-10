from setuptools import find_packages, setup

setup(
    name="nlp-toxicity",
    version="0.0.1",
    description="Project template for Natural Language Processing and Information Extraction course, 2022WS",
    author="Adam Kovacs",
    author_email="adam.kovacs@tuwien.ac.at",
    license="MIT",
    install_requires=[
        "nltk==3.7",
        "numpy==1.23.3",
        "pandas==1.5.0",
        "scikit-learn==1.1.2",
        "stanza==1.6.1"
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
