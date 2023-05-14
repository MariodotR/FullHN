# Enhancing intra-modal similarity in a cross-modal triplet loss

Tensorflow Code for the paper. Results and embeddings are available at the following link:

https://usmcl-my.sharepoint.com/:u:/r/personal/mario_mallea_sansano_usm_cl/Documents/rep-limpio.zip?csf=1&web=1&e=eNNbaz


# Qualitative Analysis

<img width="474" alt="image" src="https://user-images.githubusercontent.com/70358709/235372548-cd0e5385-925b-4e38-931f-5505ac63839b.png">


We will compare image retrieval, considering F HN as the best model for intra-modal search and HN for cross-modal search. In Fig (a), we can observe the top 5 retrievals with the image as a query and Figures (b)-(f) for each of the associated captions as queries.

We can see that with intra-modal search, the results are very similar to the query, with predominantly black and white dogs jumping over a fence like the query (top 1). This is reflected in the R and S values, which are notably better. However, for the t2i task, the ground truth image is not included in the top 5 retrievals in any of the 5 opportunities. This can be explained because the retrieval is highly susceptible to the quality of the caption, generating variability based on certain key concepts of the query, which leads to the retrieval of elements with low similarity. For example, with respect to the ground truth image, it is key to mention "jump pole" in Fig (b) (top 5), "striped gate" in Fig(d) (top 4), and "barrier" in Fig (f) (top 5) to include at least one image that is really similar to the ground truth. This is reflected in the notably better relevance values. On the other hand, the concepts of "hurdle" in Fig (c) and "obstacle" in Fig (e) are not good descriptors and increase the variability of the retrieval. In addition, dogs jumping have a significant influence even if it is with a ball, which in some cases allows semantically related images to be retrieved.

The visual case seems to be more robust, but it also allows for retrieval based on semantic information rather than visual characteristics, such as the fact that it is black and white dogs in Fig (a) (top 4), which is frequently retrieved in cross-modal retrieval scenarios. According to our analysis and as validated by the experiments, the effectiveness of intra-modal retrieval is enhanced by the proposed F HN loss.
