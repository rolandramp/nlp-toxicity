# Experiments

## Toxicity classification

We did experiments with pretrained language models.
The idea is the that training a transformer model from scratch, would take too much time and computational power.
Also, the amount of data we have would not be feasible to train a model.

In the final project we tried to do the toxicity classification on a pretrained german language model.
We decided to choose pretrained german BERT for out toxicity classification task. 
This model is GBERT from deepset (https://huggingface.co/deepset/gbert-base) which was mainly trained on wikipedia.
To this model we added a freshly initialized sequence classification header, which had to be trained on our labeld data.

To also benefit from results of a classification model we also tried a different pretrained model named toxic-bert-german also from hugginface (https://huggingface.co/ankekat1000/toxic-bert-german).
This model was already pretrained on toxicity classification task.
We did this to find out if a model trained on a specific task could lead to a better model in respect to accuracy.

As training data we had different approaches:
- we used the dataset without any preprocessing 
- we used our dataset from the previous milestones but without the vectorisation of the words. this included the lammatization of words.
- we use the 2 dataset above but with additional cleaning (stopword removal) and augmentation

## Training

Referring to the paper 'On Answer Position Bias in Transformers for Question Answering' (Rafael Glater, Rodrygo L. T. Santos
SIGIR '23: Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information RetrievalJuly 2023Pages 2215–2219https://doi.org/10.1145/3539618.3592029)
Bert seems to concentrate on the position on the lowest layers, to be sensitive to syntactic and semantic structure in the middle layers and
during fine-tuning of a down stream task the higher layers seem to have the highest influence on that task.
So we introduced a hyperparameter for freezing the top n layers.

-----------------
python train.py -ml GBERT -dp ../data/output_full.conllu -spp ../docs/splits.pkl -co
-----------------

## Toxicity NER (Named entity recognition)

To gain more insights we also tried to do a Named Entity Recognition (NER) on the term "Vulgarity"

The dataset also contains additional tags, which are
* Target_Group
* Target_Individual
* Target_Other
* Vulgarity

We decided to use the Vulgarity tag to mark the words which are considered to be an insult.
1306 Comments contained a Vulgarity, but 452 did not belong to a toxic considered comment.
These comments are split into 1484 number of sentences containing vulgarities.



## Program arguments

- -s Save model
- -c Use conllu file
- -st Train on reduced dataset
- -sp Path to save model
- -ml Model to use 
  - NN *neural network*
  - RF *random forest*
  - GBERT *german BERT model for classification*
  - GTOXBERT *pretrained toxic classification model*
  - GTOXBERT *pretrained toxic classification model*
- -e Number of epochs to train
- -dp Path to data
- -spp Path to splits file
- -fl Freeze first number of layers including embedding layer
- -cl apply simple data cleaning steps (removing punctuation & stopwords)
- -a apply class balancing augmentation
- -to apply cleaning & augmentation steps to training set only (if false / not set, will apply augmentation only to validation & test set too)

## GBERT-Fine-Tuning

A transformer model, pretrained on the german language, is used to train a classification down stream task.
The model deepset/gbert-base is taken from hugginface.io. 

model training is done from script path with python ./train.py -s -ml GBERT -e 15 -dp ../data/data_all.json -sp ../model/ -fl 6 -st
- use new train test split on original data

model training is done from script path with python ./train.py -s -ml GBERT -e 15 -dp ../data/output_full.conllu -spp ../docs/splits.pkl -sp ../model/ -fl 6 -st -c
- train on the conllu prepared data set

## GBERT predict

model prediction is done from script path with python ./predict.py -ml GBERT -mp ../run/20240114095504_GBERT_349/model -prp ../run/20240114095504_GBERT_349/prediction -t ../data/output_full.conllu -sd ../docs/splits.pkl -uc
- mp is the path to the fine-tuned bert model
- prp is the path were the output csv should be written

## NER training

NER training is done with a separate training script ner_train.py. 
*Unfortunately there were huge problems trying to integrate the ner code into the train structure of this project.*
Update: NER is now also integrated in to the train framework and can be started from the script path

Again the pretrained model deepset/gbert-base is used again, but trained on a named entity recognition downstream task.

### How to train
model training is done from script path with python ./ner_train.py 
- uses only the observations with vulgarity tags
- dataset was prepared to be able to do token classification
- the 6 lower layers were frozen

or model training is done from script path with python ./train.py -s -ml GBERTNER2 -e 10 -dp ../data/data_all.json -sp ../runs/ -fl 2 -co
- data is prepared on the fly
- only comments with vulgarities are selected and token classification is done on sentence level.

### Results
Form several test run this configuration showed the best results
- 10 Epochs
- 2 frozen layers 

Accuracy:  0.933 
Precision:  0.784
Recall:  0.797
F1-Score:  0.767

Example sentences with vulgarity tags are shown in the slides.

### Conclusion
NER on toxicity is a possibility to enhance somehow the expandability of the BERT model decision.
It is in fact a separate model, which can not explain the other model directly, but in the case of a toxicity classification it could point out the vulgarity in the sentence, which maybe indicates the toxicity. 


## Experiments with GBERT classification model

### Training Setup
In order to identify an optimally performing model for classifying toxic speech, a large set of models was trained and evaluated. These models were varied in the following ways:

- Freezing 2, 6, and 10 layers
- applying data augmentation (class balancing) and cleaning (removing stopwords & punctuation)
- applying data augmentation only
- applying neither

The models were trained for 10 epochs max, with 5 runs per setup (9 unique model / training configurations total). An early-stopping mechanism was put in place, so most training runs ended after 4-5 epochs as no more improvements were made to model performance. Model checkpoints were saved after each epoch, as well as at the end of the training run. 

The data used for training and evaluating these models was split in the exact same way as for milestone 2, in order to have optimal comparability especially for the test set. 

### Model Evaluation

Predictions on the test set were generated for every model as well as all epoch checkpoints, for a total of 256 sets of predictions. 
These predictions can be found in "all_preds.csv", and the metrics calculated from them (classification accuracy, F1-Score, Precision and Recall) along with the model setup in "resultsv02.csv". The column names in "all_preds.csv" correspond to the model IDs from "resultsv02.csv". 

The top-performing models used 2 frozen layers, no data cleaning, with both models with and without augmentation achieving good results. The best models as evaluated on the training set are the following:
id 99 - fl = 2, clean = false, augment = true, epoch = 3: best f1 (0.717), close to best accuracy (0.783)
id 107 - fl = 2, clean = false, augment = true, epoch = 1: best accuracy (0.794), close to best f1 (0.709)
id 158 - fl = 2, clean = false, augment = false, epoch = -1: best overall result for a "final" model (when training loop decided to end) (accuracy 0.785, f1 0.708)

### Comparison to other models

These results are evidently considerably better than the ones achieved with the baseline models in milestone 2. 
Furthermore, a BERT-based model that was already trained on german toxic speech classification was also evaluated, 
achieving a classification accuracy of 0.645 without any finetuning and 0.737 with finetuning akin to the best performing model of the ones trained from a basic german bert model. 
This performance is a fair bit worse than the best performing models, indicating that transfer learning between datasets is in this case detrimental compared to using a basic language model that hasn't already been trained for a specific task with a specific dataset.

# Presentation Slides

Slides are inside this folder: NLP Final Presentation.pdf
