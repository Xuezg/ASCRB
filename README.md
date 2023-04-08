# ASCRB
The tool is developed for A unified Multi-view-based Joint representation learning for CircRNA Binding Sites prediction
# Requirements
- Python 3.6 (64-bit)
- Keras 2.2.5 in Python
- TensorFlow-GPU 1.13.0 in Python
- Numpy 1.18.0 in Python
- Gensim 3.8.3
- PyTorch 1.8.1
- Ubuntu 18.04 (64-bit)
# data
- circRNA-RBP
- lnRNA-RBP 
# code
- BertDealEmbedding.py 
- AnalyseFASTA.py
- Deal_Kmer.py
- DProcess.py
- JLCRB.py
## How to train the CircRNA model
You can train the model of 5-fold cross-validation with a very simple way by the command blow:  
*Python JLCRB.py* and make sure the RNA embedding flag is set to circRNA_model. The script of if **name == "main"** calls training process which trains several models of each model type for a circRNA and finds the best set of hyperparameters. The main function then trains the models several times (num_final_runs) and saves the best model.
## How to train the Linear-RNA model
You can also test the linear-RNA model of 5-fold cross-validation, and make sure the RNA embedding flag is set to linRNA2Vec_model and the file path is set to *linRNA-RBP*.
The prediction results will be displayed automatically. If you need to save the results, please specify the path yourself. Thank you and enjoy the tool!
If you have any suggestions or questions, please email me at *dxqllp@163.com*.
