This folder is the code for unsupervised sentence-topic classification.

User can input any topic. The model can compute similarity scores and classify a document (the sentences) to the topics. 
Or, the model can cluster all the sentences that relate to a topic, given a user-specified threshold.

The model uses pre-trained GloVe (840B, 300d) model.

Similarity metrics include cosine similarity, Euclidean distance, Manhattan distance, Word Mover's distance, LDA, 
and Siamese MaLSTM (trained on Quora data).
