# Offline-Signature-Verifier

This is repository containing implementation of experiments mentioned 
in the paper concerning offline verification of handwritten signatures. 
Original paper (Croatian) link: 
https://drive.google.com/file/d/1akPpLKbRkJUbThOnsDSyMiHOgVIyXicJ/view?usp=sharing

1st chapter training set: 
https://www.kaggle.com/divyanshrai/handwritten-signatures

2nd chapter training set: 
http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2009_Signature_Verification_Competition_(SigComp2009)

Image preprocessing functions are implemented in 
`preprocessor.py` script. In order to perform whole preprocessing as described in referenced 
paper, just run `preprocess()` method.

Feature extraction methods are implemented in `feature_extractor.py` 
script. In order to retrieve all features from preprocessed image, just 
run `extract()` method.

Euclidian model is implemeted via `euclid_model` class in 
`euclidian_model.py` script. All the other experiments from the first 
chapter are conducted in `simple_models.py` script.

Experiments from the second chapter using Neural Networks and SVM are 
implemented in `complex_models.py` script.
