#The following files test the regularized multilayered perceptron with various structures, where the first number as # of input nodes, last as # of output nodes, and in between are hiddens.
#These multiple files were necessary to run tests simultaneously on different corn machines. To run these tests, run "python [test_name]" on corn.
MLP_regularized_testing_784_100_10.py
MLP_regularized_testing_784_100_100_10.py
MLP_regularized_testing_784_100_100_100_10.py
MLP_regularized_testing_784_30_10.py
MLP_regularized_testing_784_30_30_10.py

#The following files test the multilayered perceptron (without regularization) with various structues, where the first number as # of input nodes, last as # of output nodes, and in between are hiddens.
MLP_testing_784_100_10.py
MLP_testing_784_100_100_10.py
MLP_testing_784_100_100_100_10.py
MLP_testing_784_30_10.py
MLP_testing_784_30_30_10.py

#The following files test the autoencoder paired with a linear classifier. The different files contain autoencoders with different number of hidden nodes, representing how much the orignial pixels are compressed
autoencoder_testing_1.py
autoencoder_testing_2.py
autoencoder_testing_3.py
autoencoder_testing_4.py
autoencoder_testing_5.py
autoencoder_testing_6.py

#The following file tests the multiclass linear classifier with various hyperparameters
linear_classifier_testing.py

#The following files are involved with the post office extension, where a heuristic to determine a confidence level in a classification is implemented. The "lc_falsepos.py" file defines the multiclass linear classifier
#with the modification, and LC_constraint_testing.py tests the setup.
lc_falsepos.py
LC_constraint_testing.py

#The following files are involved with the automated hyperparameter tuning extension, where we attempted to design a system for tuning a linear classifier automatically by changing hyperparameters a little bit 
#and see how the accuracies change. The "lc_hyperparameter.py" contains a multiclass linear classifier that is suited for this system and "hyperparameter_tuning_testing.py" actually runs the tuner.
hyperparameter_tuning_testing.py
lc_hyperparameter.py

#The following files contain our implementations of the multiclass linear classifier, the multilayered perceptron, the regularized multilayered perceptron, and the autoencoder. 
linear_classifier.py
multilayered_perceptron.py
multilayered_perceptron_improved.py
autoencoder.py

#The following file defines a function for reading in the excel spreadsheet that we downloaded from Kaggle as our dataset.
util.py

#The following files contain testing scripts for generating the training history for our best-case, tuned classifiers. Prints out accuracies every iteration with the current weights.
max_LC_testing.py
max_MLP_testing.py
max_MLP_regularized_testing.py
max_autoencoder_testing.py

#The following file was used to test our code during development and has changed a lot. We used it mostly for sanity checking when we wrote parts of all the implementations..
main.py

#The results of our testing are summarized in the PDF file in code.zip, named results.pdf.