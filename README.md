# Detection of Toxicity in Online Comments

The goal of the experiment was to train a transformer model (BERT or others) to
do a binary classification on online comments to mark them as toxic. Toxic comments
are meant to get people to leave a online discussion. Further more a NER (Named
Entity Recognition) model was trained, to highlight vulgarities in comments. The idea
behind that was to gain some insights on vulgarities and toxicity.

## Prerequisites

This project is implemented in Python 3.10 and requires a Cuda capable GPU.
Using Anaconda to work with virtual environments is advisable, but using pythons venv environments is also possible.

First create a new conda environment with python 3.10 and activate it:

```bash
conda create -n nlp_toxicity python=3.10
conda activate nlp_toxicity
```
or with venv of a python 3.10 installation
```bash
python -m venv /path/to/nlp_toxicity
source /path/to/nlp_toxicity/bin/activate
```

Then install this repository as a package, the `-e` flag installs the package in editable mode, so you can make changes to the code and they will be reflected in the package.

```bash
pip install -r requirements.txt
pip install -e .
```

## Contents

Here the directory structure of the project is shown and described:

```
📦nlp-toxicity
 ┣ 📂data
 ┃ ┣ 📜README.md
 ┣ 📂docs
 ┃ ┗ README.md
 ┣ 📂model
 ┃ ┗ README.md
 ┣ 📂runs
 ┃ ┗ README.md
 ┣ 📂scripts
 ┃ ┣ 📜evaluate.py
 ┃ ┣ 📜ner_train.py
 ┃ ┣ 📜predict.py
 ┃ ┣ 📜prepare.py
 ┃ ┗ 📜train.py
 ┣ 📂nlptoxicity
 ┃ ┣ 📜projectdatasetcreator.py
 ┃ ┣ 📜projectdatasets.py
 ┃ ┣ 📜trainer.py
 ┃ ┣ 📜utils.py
 ┃ ┗ 📜__init__.py
 ┣ 📜.gitignore
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┣ 📜setup.py
```

- `data`: This folder contains the data that you will use for training and testing your models. You can also store your trained models in this folder. The best practice is to store the data elsewhere (e.g. on a cloud storage) and provivde download links. If your data is small enough you can also store it in the repository.
- `docs`: This folder contains the reports of your project. You will be asked to write your reports here in Jupyter Notebooks or in simple Markdown files.
- `model`: This folder contains the trained models.
- `runs`: This folder is used for storing training runs.
- `scripts`: This folder contains the scripts that you will use to train, evaluate and test your models. You can also use these scripts to evaluate your models.
- `tuwnlpie`: This folder contains the code of your project. This is a python package that is installed in the conda environment that you created. You can use this package to import your code in your scripts and in your notebooks. The `setup.py` file contains all the information about the installation of this repositorz. The structure of this folder is the following:
  - `milestone1`: This folder contains the code for the first milestone. 
  - `milestone2`: This folder contains the code for the second milestone.
  - `finalproject`: This folder contains the code for the final project.
  - `__init__.py`: This file is used to initialize the `tuwnlpie` package. You can use this file to import your code in your scripts and in your notebooks.
- `setup.py`: This file contains all the information about the installation of this repository. You can use this file to install this repository as a package in your conda environment.
- `LICENSE`: This file contains the license of this repository.
- `team.cfg`: This file contains the information about your team.

## Usage

First download all the resources that you need for the project. You can find all the information in the `data/README.md` file.

In the `scripts` folder you can find the scripts that you can use to train, evaluate and test your models. 


## License

This work is licensed under the MIT License. See LICENSE file.

## Citation (optional)

## Contact

https://orcid.org/0009-0003-5145-2197

email: e0055097@student.tuwien.ac.at

[![DOI](https://zenodo.org/badge/783249120.svg)](https://zenodo.org/doi/10.5281/zenodo.10938085)