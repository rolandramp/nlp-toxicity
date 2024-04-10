# Data preparation

- There are 522 different article titles in this dataset, with quite some of them containing only few or one comment.
- We decided not use this column, because we think, that it does not provide any information about toxicity of the
  comments.
- Furthermore, we also dropped the column with the annotations, because the annotations does not indicate a toxic
  comment.
- So only the columns Comments and Label are relevant for our NLP classification task of toxicity.
- We checked the comments and found out, that there are duplicate entries with also different labels.
- So we decided to remove these duplicates and use the majority on the labels (on even we went for the toxic label)
- Two of the comments contained only binary data, which we decoded back to text.
- We also checked the labels and found out that we have 2818 nontoxic and 1655 toxic entries. (data set is not very
  unbalanced)

## Export

on the stanza-pipeline we used following processors: (from https://stanfordnlp.github.io/stanza/pipeline.html)

- tokenize Tokenizes the text and performs sentence segmentation.
- mwt Expands multi-word tokens (MWT) predicted by the TokenizeProcessor. This is only applicable to some languages.
- lemma Generates the word lemmas for all words in the Document.
- pos Labels tokens with their universal POS (UPOS) tags, treebank-specific POS (XPOS) tags, and universal morphological
  features (UFeats).
- sentiment Assign per-sentence sentiment scores. (0, 1, or 2 (negative, neutral, positive).)

The pipeline operates on each comment as a separate document, so in order to preserve information on labels and the titles of the articles they belong to, the export was made with a helper function that effectively combines all the separate CoNNLu outputs into one file with a dedicated marker to separate comments, as well as a line for information that would otherwise be lost. Another helper function (both found in the utils.py for milestone 1) can be used to parse this back into a list of stanza documents.

## Additional data preparation

The data generated during Milestone 1 contains (among other information) the lemmas for each word in the comments. This data is not yet in a format to serve as input to machine learning models, an embedding of some sort is needed. As this milestone is about baselines we opted for a simple bag-of-words approach, in which all words that appear in the corpus are used to create a count vector which then represents each comment and serves as the numerical input to the models. This is a convenient, simple and easy-to-implement approach but has drawbacks, primarily of course as the content of speech is not only defined by the individual words, but also because variation in words (such as dialect or misspellings) are basically impossible to model with a limited dataset, even after lemmatization (which often fails on such cases).

The lemmatized version of the full dataset contains more than 17000 unique words (including numericals and punctuation). However, of these more than 11000 are singletons - that is, they only appear once. We made the decision to remove these singletons, as they provide no value in the bag-of-words embedding as the models would have no chance to learn the significance of these words and they just serve as dimension bloat. Following this pruning, some comments were left with zero words, which were also removed - a total of around 100 comments was affected by this, presumably these contained single or very few singleton words. 

This lemmatized, pruned and then vectorized data was then split into train, test and validation sets for the baseline experiments. Splits of 60:20:20 ratio were used and saved in pickled form for easy reuseability across different models so the experiments would be comparable.


