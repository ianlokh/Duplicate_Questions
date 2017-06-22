# Classifying Duplicate Questions with TensorFlow

The following are the brief descriptions of the files that you may find useful when analysing the code.

#### qqp\_BaselineModels.py
This script is the base script that does EDA and text feature generation. Because the Quora dataset is large the feature generation is done in chunks and saved in HD5 file format.

Please note that because the generation of semantic\_similarity and word\_order\_similarity score takes too long, I have generated the data separately   into a HD5 file (df\_all\_train.wordnet.h5). This file contains all the semantic\_similarity and word\_order\_similarity scores for each of the training and test questions from Quora. A join by index is then used to tag these scores back to their respective questions.

Next, we will then use word2vec to generate image features and append these features into the dataframe in chunks and save it into another HD5 file (df\_all\_train.h5) due to memory constraints.

Once the df\_all_\train.h5 file is created, this is then basically the core set of text features to be used to train the models.

Included in this file are codes for GridSearchCV using SVM and XGBoost. Please note that these will take a long time to train.


#### global\_settings.py
Settings file that is being used by qqp_BaselineModels.py.


#### img\_feat\_gen.py
Generate the image features based on the word2vec similarity scores


#### wordnetutils.py
Generate WordNet based semantic\_similarity and word\_order\_similarity scores. You probably don't need to run this code to run the exercise. Just use the **df\_all\_train\_pres.h5** provided in the link below. However, if you want to use this code in your own work, please remember to attribute to [Sujit Pal](http://sujitpal.blogspot.sg/2014/12/semantic-similarity-for-short-sentences.html) and this site.


#### parallelproc.py
Implementation of parallel apply for Python on Mac. Unfortunately this code does not yet work on Windows as it relies on UNIX fork to spawn multiple threads.


#### qqp_TensorFlowCNN_Model.py
Code as shown in presentation and Jupyter notebook.


#### AllowedNumbers.csv & AllowedStopwords.csv
List of numbers and stopwords that will not be removed during pre-processing


### Links to HD5 files

You need to download the pre-built training data set before you can run the examples. Please note that these are large files.

[df_all_temp_pres.csv](https://drive.google.com/file/d/0B6w7LByoqo9YNlRZU3dFZ0dGcVk/view?usp=sharing) - Approx 92 MB

[df_all_train_pres.h5](https://drive.google.com/file/d/0B6w7LByoqo9Yd0tkdjREOGsyN0k/view?usp=sharing) - Approx 479 MB