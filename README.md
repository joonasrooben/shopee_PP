# shopee_price_police 
### Jesse Dingley and Joonas Järve

## Motivation
We are competing in the [Shopee product matching](https://www.kaggle.com/c/shopee-product-matching/overview) competition. The goal of the competition is to help e-commerce platform [Shopee](https://shopee.com/) to identity postings that are in fact the same product. We have two main motivations for doing this project. Firstly, the fact that this project takes the form of a Kaggle competition excites us. The challenge to beat and compare our results with others entices us. Secondly, we are both interested in NLP. Each posting is made up of the product image and the product description. So, this project combines a nice blend of computer vision and NLP which suits our interests well. Finally, the idea of being able to find the best deal on a product is a great motivation.    

## Method
* General idea: Fine-tune pretrained visual and textual embeddings (unsupervised) then apply a clustering algorithm to find similar postings. We have found inspiration for implementation from [Roman Beaumont](https://rom1504.medium.com/image-embeddings-ed1b194d113e) and [Erdem Isbilen](https://towardsdatascience.com/image-similarity-detection-in-action-with-tensorflow-2-0-b8d9a78b2509) blog posts.
* Secondary approach: once again pretrain embeddings but instead of performing clustering we will pass these embeddings through a feed-forward neural network that acts as a multi-label multi-class classifier, each class being a different posting. So for example our classifier might label posting 3 with postings 2, 3, 6 and 10 meaning that posting 3 corresponds to the same products in postings 2, 3, 6 and 10. 
* Visual embeddings: efficientNet pretrained model (and maybe our own small model as well like we did in the HW).
* Text embeddings: BERT 
* Clustering: KNN/GMM/RandomForest
* Frameworks: Keras & Tensorflow 
* Blog format: in git-pages

## Data
As we are doing a Kaggle competition, the data available is very clean, well documented and well organized.
<br>
More precisely, the training data comes in the form of a metadata csv file and a folder containing all <b>32400</b> postings images. The csv file gives us the following information for each posting:
  - Posting ID
  - Image ID
  - Image perceptual hash
  - Posting description
  - Group ID code for all postings that map to the same product.

The test data is in the same format except that we don't have the group ID code field. The test data is much bigger at 70 000 examples.  

## ToDos
* Initial notebook ready to cook
* Setting up BERT
* Setting up Viz Embeddings
* clustering 

