# Data-Agnostic Local Neighborhood Generation (DAG)

In this repository we provide the source code and the public available datasets used in the paper *"Data-Agnostic Local Neighborhood Generation"*. 

Synthetic data generation has been widely adopted in different fields such as software testing, data privacy, imbalanced learning, machine learning explanation, etc. In such contexts, it can be important to generate data samples located within "local" areas surrounding specific instances. Indeed, local synthetic data can help the learning phase of predictive models, and it is fundamental for methods explaining the local decision-making behavior of obscure classifiers. In explainable machine learning, each local explainer either introduces an ad-hoc procedure for neighborhood generation designed for a particular type of data, or uses a general-purpose approach having different effects on different data types.
The contribution of the paper and of this repository is twofold. 
First, we introduce a method based on generative operators allowing the synthetic neighborhood generation by applying specific perturbations on a given input instance. 
The key factor of the proposed method consists in performing a data transformation that makes agnostic the data generation, i.e., applicable to any type of data. 
Second, we design a framework for evaluating the goodness of local synthetic neighborhoods exploiting both supervised and unsupervised methodologies.
A deep experimentation on a wide range of datasets of different types shows the effectiveness of the proposal in generating realistic neighborhoods which are also compact and dense.

## Citation
> R. Guidotti, A. Monreale *"
Data-Agnostic Local Neighborhood Generation"*, ICDM, 2020. [Paper](to appear)
