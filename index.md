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

And of course in typical Kaggle fashion, the test set is hidden. Because otherwise we might be seeing scores close to 100% 😅 


## Masterplan

<b><u>General Approach / Pipeline :</u></b> It's first best to explain our general approach to the problem at hand. For each posting we start by contructing an embedding for the whole posting, combining image and text (posting description). Here is a diagram explaining the situation because let's be honest diagrams are just better than textual descriptions.

<div style="text-align:center"><img src="emb_diagram.png" alt="emb_diagram" class="center" height="500"></div>

Once we have our embeddings for each posting we can proceed onto matching them. The following diagram show how to determine all matches for a posting. In this example we're matching all similar postings to posting 312.

<div style="text-align:center"><img src="process_scheme.png" alt="prcess" class="center" height="500"></div>

The previous diagrams simply present our pipeline for finding similar products and well you might now be asking yourselves <b>hOw dO wE ExTrAcT tHeSe "EmBedDiNgS"??</b>. This is where the fun begins and Deep Learning finally arrives into the picture. In the next parts of this blog we explore how we extract the image and text embeddings with help from neural networks.  

### Images

### Text

The second major part to solving the problem is dealing with the product descriptions. It's nice having the image embeddings but using description embeddings can also help us in finding similar postings. Two postings with similar descriptions are likely to represent the same product. We try many approaches to extract description embeddings:
  - fine-tuned BERT
  - Just BERT
  - TF-IDF (#underrated)
  - Doc2Vec

We won't cover each point extensively, but we'll explore in detail the most interesting approaches (BERT and fine-tuned BERT). Don't worry we will also touch upon the other approaches 😅.

<b><u>BERT & fine-tuning BERT</u></b> 
<br>
Now some of you may not know what BERT is or what it can do so here's a brief overview of BERT-like models.

BERT is a pre-trained language model (PLM) (or language representation model). PLMs are models trained in an unsupervised fashion on huge amounts of text by predicting masked words in sentences (and sometimes next sentence prediction - which is the case for BERT). These models learn to understand and represent language. A common architecture used today is the transformer architecture that uses the attention mechanism. As the model processes each word of the input sequence, attention allows the model to look at other words in the input sequence for clues that can help lead to a better encoding (or representation) for this word. The BERT architecture follows the following scheme (a stack of encoders- each encoder consisting of an attention mechanism layer a linear layer) [source](http://jalammar.github.io/illustrated-bert/)

<div style="text-align:center"><img src="bert_scheme.png" alt="bert" class="center" height="250"></div>

So BERT takes as input a sequence and outputs:
  - a contextualized embedding that represents the whole sequence 
  - contextualized embeddings for each separate token.

So one approach to extracting description embeddings is to simply pass each description through BERT and get an embedding (of length 768).

Another but more interesting approach is to fine-tune BERT on our training data. Now the useful thing about PLMs is that they can be fine-tuned to any supervised learning task end-to-end. Meaning we can alter and hence "fine-tune" the pre-trained weights of BERT to teach BERT to be more accustomed to our data, to grasp a better understanding of the data at hand. BERT was pre-trained on wikipedia-like data and our data is messy descriptions: Theoretically, BERT alone won't understand the descriptions very well and won't get us optimal embeddings without fine-tuning.

So now you must be begging to know what task we fine-tuned BERT on ? Well idk lol
<br>
**Joonas take over here plz**

...

As mentioned we also try two other models: TF-IDF and Word2Vec. These two model do not take attention and context into account and are much smaller than BERT but nevertheless they still produce great results and are much more efficient (🚨#spoiler🚨 but Doc2Vec did the worst).


<b><u>TF-IDF</u></b> 
<br>
TF-IDF is a purely <b>statistical</b> approach. TF-IDF evaluates how relevant a word is to a document in a collection of documents. This is done by multiplying two metrics: how many times a word appears in a document (TF), and the inverse document frequency of the word across a set of documents (IDF).


<b><u>Doc2Vec</u></b> 
<br>
Doc2vec is another neural approach to embeddings. It is based off and (very) similar to the famous Word2Vec model. To give a brief overview, Word2vec models train a simple neural net with one hidden layer to predict the next word in a sequence. Word2Vec is <b>NOT</b> contextual: there is only one unique embedding for each word in the vocabulary, full stop, unlike BERT where a word embedding depends on the other words around it. Doc2Vec is basically the same as Word2Vec with the only difference being a paragraph matrix added to the input. In practice we train a Doc2vec model on our training data.

<b><u>Some (interesting but somewhat concerning 😐) results</u></b> 
<br>

Here is a summary of the mean F1 scores (on training data) for each textual model (<b>without</b> concatenation with image embeddings) using optimal threshold:

<style type="text/css">
.tg  {border:none;border-collapse:collapse;border-color:#ccc;border-spacing:0; text-align:center;}
.tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:0px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:0px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-btxf{background-color:#f9f9f9;border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg" >
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-0pky">Mean F1-score</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-btxf">Fine-tuned BERT</td>
    <td class="tg-btxf">59%<br></td>
  </tr>
  <tr>
    <td class="tg-0pky">Just BERT</td>
    <td class="tg-0pky">59%</td>
  </tr>
  <tr>
    <td class="tg-btxf">TF-IDF</td>
    <td class="tg-btxf"><b>62%</b></td>
  </tr>
  <tr>
    <td class="tg-0pky">Doc2Vec</td>
    <td class="tg-0pky">57%</td>
  </tr>
</tbody>
</table>

Here is a summary of the mean F1 scores (on training data) for each textual model (<b>with</b> concatenation with image embeddings) using optimal threshold:
<style type="text/css" >
.tg  {border:none;border-collapse:collapse;border-color:#ccc;border-spacing:0; text-align:center;}
.tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:0px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:0px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-btxf{background-color:#f9f9f9;border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-0pky">Mean F1-score</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-btxf">Fine-tuned BERT</td>
    <td class="tg-btxf">85.1%<br></td>
  </tr>
  <tr>
    <td class="tg-0pky">Just BERT</td>
    <td class="tg-0pky"><b>85.4%</b></td>
  </tr>
  <tr>
    <td class="tg-btxf">TF-IDF</td>
    <td class="tg-btxf">84.8%</td>
  </tr>
  <tr>
    <td class="tg-0pky">Doc2Vec</td>
    <td class="tg-0pky">83%</td>
  </tr>
</tbody>
</table>

<b>Now just hold on before you start doubting the power and effectiveness of BERT 🚨</b>. As we can see, when using only the language models for matching, TF-IDF comes out on top 🤯 ! By a margin of 3%: a small but not insignificant difference between the BERT models (we can leave Doc2Vec out of the discussion at this point now).

This suggests two things:
  1. Never understimate the power of smaller simpler approaches such as TF-IDF. They deliver surprisingly good results!
  2. Even when fine-tuned on the product descriptions, BERT still has a tough time grasping these short messy descriptions. This suggests BERT should be left to deal with "classical" language (proper sentences). A statistical approach seems to fit this problem better.

Looking at the results when concatenating image and text embeddings, un-fine-tuned BERT comes out on top with a very small margin of 0.3% over fine-tuned BERT. The margins aren't huge between all four models (maximum 2%). 

We can form a few hypotheses from all these results.
  - Textual information doesn't help much in matching products (only improves the F1 score by 1% (note that when only using image models we get F1 of 84.0%)) 
  - We weren't combining the two embeddings together in the right way (we used concatenation)
  - We used the wrong models
  - we didn't fine-tune BERT in the right way or for long enough
  
## The Endgame

## 3rd vs 1st place

## Discussion

i.e what we learned and such
