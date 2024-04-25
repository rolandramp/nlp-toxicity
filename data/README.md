# Dataset

The data used in this project is a closed dataset.

## Metadata

The dataset is 2MB in size and of JSON data type.
The metadata to this dataset is located under the DOI [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11039040.svg)](https://doi.org/10.5281/zenodo.11039040)

## Structure

The dataset consists of a json file containing following columns

 Name          | Type                | Description 
---------------|---------------------|-------------
 Index         | int64               | 
 Article_title | str                 | title bolunging to the comment
 Comment       | str                 | full text of the comment
 Label         | int64               | 0 for non toxix and 1 for toxic
 Tags          | List[Dict] | futher descritpion below

Tag is a list of dicts with 'Tag' and 'Token' pairs.
Here an example:

[{'Tag': 'Target_Group', 'Token': 'name of a target group'}, {'Tag': 'Vulgarity', 'Token': 'swear word'}]

The following Tags exists:
* Target_Group
* Target_Individual
* Target_Other
* Vulgarity