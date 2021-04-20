# NeuroPpred-Fuse
NeuroPpred-Fuse: an interpretable stacking model for prediction of neuropeptides by fusing sequence information and feature selection methods
This repository is about stacking model named NeuroPred-Fuse.
The file 'firstlayer' in folder script is the first layer in our model.
run python firstlayer --mode train, you can train the model;
run python firstlayer --mode test, you can get the output of first layer about the file you want to test named 'resultof_1layer'. If you really wan to test your file, please run the first command firstly.

Similarly,
run python secondlayer --mode train can get the final output about ourself dataset, and run python firstlayer --mode test --filename resultof_1layer can get the final output of youself datasets you want to test.
