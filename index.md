# *Shopee* **Price Police**

<i>How to detect if two products are the same by their images and descriptions with the help of embeddings.</i>

<b>Jesse Dingley & Joonas Järve</b>

<div style="text-align:center"><img src="blog_meme.jpg" alt="meme" class="center" height="300"></div>

## Background & Task

Do you love a good deal and don't like being scammed into spending more than you need to ? Then you've come to the right place. Kaggle held a competition throughout a few months of 2021: <i>Shopee - Price Match Guarantee</i> [see here](https://www.kaggle.com/c/shopee-product-matching/overview). More than 2000 teams participated but only three came out victorious and with a fatter wallet. This competition was hosted by Shopee [see here](https://shopee.com/), the leading e-commerce platform in Southeast Asia and Taiwan. Their goal is to filter out matching products at higher prices to ensure the best possible customer experience.

Our team, the <i>Shopee Price Police</i> took part in this competition as a part of the neural networks course project at the University of Tartu in Estonia. We're definetly not grandmasters of Deep Learning (yet) but nevertheless we study a few interesting approaches to this problem. 

## Data

As this being a Kaggle competition, we were treated to very well organised and freely avaiable data. So what exactly did Shopee provide us with to crack on with the task?

Made available for training was a set of 34250 postings. Each posting has the following attributes:
  - posting ID
  - product image
  - perceptual hash of the product image
  - product description
  - ID code for all postings that map to the same product *<i><b>not made available for test set</b></i>
  
Note that there are 11 014 unique ID codes for all postings that map to the same product.
  
For those wondering, perceptual hashing creates a "fingerprint" of some multimedia item, in this case: images. Perceptual hashes act like embeddings representing the features of the image in question. An example hash might be `b94cb00ed3e50f78`. So if two postings have the same perceptual hash, we can essentially consider these postings of showing the same product. [See here](https://en.wikipedia.org/wiki/Perceptual_hashing) for more information.

In untypical fashion, the test set is bigger than the train set, at around 60 000 examples! We usually see much smaller test sets but we can assume Shopee wants to ensure the proposed approaches will also work very well when deployed in real life. Testing on a small test set might overvalue some models. Testing on larger sets tests to see if the models can perform well when dealing with a wider range of different postings.

And of course in typical Kaggle fashion, the test set is hidden. Because otherwise we might be seeing scores close to 100%! 


## Masterplan

<b><u>General Approach:</u></b> It's first best to explain are general approach to the problem at hand. We first contruct an embedding for the whole description combining image and text. Here is a diagram explaining the situation because let's be honest diagrams are just better than textual descriptions.

<div style="text-align:center"><img src="emb_diagram.png" alt="emb_diagram" class="center" height="500"></div>

Then to determine all matches for a posting:

<div style="text-align:center"><img src="process_scheme.png" alt="prcess" class="center" height="500"></div>


### Images

### Text

The second major part to solving the problem is dealing with the product descriptions. It's nice having the image embeddings but using description embeddings can also help us in finding similar postings. Two postings with similar descriptions are likely to represent the same product. We tried many approaches to extract description embeddings:
  - fine-tuned BERT
  - Just BERT
  - TF-IDF
  - Doc2Vec

Let's dive into each of these approaches and see which ones worked, which ones didn't, which ones were efficient and which ones were sadly not.

## The Endgame

## 3rd vs 1st place

## Discussion

i.e what we learned and such
