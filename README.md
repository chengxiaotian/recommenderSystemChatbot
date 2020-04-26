# Recommender System Chatbot for Travellers to Beijing

## Objective: The ultimate goal of this project is to build a recommender system which can give suggestions to travellers basing on their preference. At the back end, will be using machine learning models including conventional one such as the KNN, in addition, a neural network is built basing on the concept of content-based filtering algorithm. Eventually, the chatbot will be connected with a Faceboot conversational interface. 

### Dataset acquisition:
Dataset used in this project is derived from a crawler built with Python spider. Information collected including three parts:

- POI (point of interest) information of Beijing
- review information
- user information

### Data Pre-processing:
#### Numeric Data Preprocessing
Useing Matrix Factorization to impute missing data. R ≈ P x Q^T=R ̂
#### Text data pre-processing
tokenization -> stop-word removal -> stemming/lemmetization -> 
feature engineering with word embedding using spaCy

Constructing User Profile
(entertainment, nature, culture, art, shopping)
Constructing POI Profile
(entertainment, nature, culture, art, shopping)

### Modelling:

1. Model Design:
![model_design]('./image/model_design.jpg')
- Initially, model will ask user for his perference defined by a 5-dimensioned vector. 
- Content-based filtering model is triggered basing on the given user vector, and uses a neural network to find out end point with closest similarity with the user's preference. This point is then picked as recommended place to the user. 
- The model will then interact with user ask if they want to check-out other similar places like this one. 
- Given the answer to above question is YES, K-nearest-neighbour model will be triggered to find out 5 neighbourhood POIs with the place chosen by the user. At this point, will be following collaborative filtering algorithm. 


### Implementation with chatbot as interface:

