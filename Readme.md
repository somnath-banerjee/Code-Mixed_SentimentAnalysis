#Code-Mixed Sentiment Analysis
***
This work describes the participation of **LIMSI\_UPV** team in **SemEval-2020 Task 9: Sentiment Analysis for Code-Mixed Social Media Text**.
The proposed approach competed in SentiMix Hindi-English subtask, that addresses the problem of predicting the sentiment of a given Hindi-English code-mixed tweet. 
We propose **Recurrent Convolutional Neural Network** that combines both the recurrent neural network and the convolutional network to better capture the semantics of the text, for code-mixed sentiment analysis. 

If you use this code please cite our paper:

***
@inproceedings{banerjee2020sentimix,

title={LIMSI\_UPV at SemEval-2020 Task 9: Recurrent Convolutional Neural Network for Code-mixed Sentiment Analysis},

author={Banerjee, Somnath and Ghannay,  Sahar and Rosset, Sophie and Vilnat, Anne and Rosso, Paolo},

booktitle = {Proceedings of the 14th International Workshop on Semantic Evaluation ({S}em{E}val-2020)},

year = {2020},

month = {December},

address = {Barcelona, Spain},

publisher = {Association for Computational Linguistics},

}
***

- Repository structure
	+ code
		+ config.py
		+ utility.py
		+ model.py
		+ train.py
	+ embeddings
	+ data
	+ paper.pdf
