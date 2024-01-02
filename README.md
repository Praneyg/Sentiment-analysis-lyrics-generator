Introduction:

This project aims to build a GPT-based model that generates new song lyrics with a
specified sentiment. The model will be trained using a dataset containing song lyrics
with associated sentiment labels (happy, sad, or neutral). The goal is to fine-tune the
GPT model so that it can create coherent and creative lyrics that match the given
sentiment.

This text generation project combines sentiment analysis with creative text
generation to produce sentiment-specific song lyrics. By fine-tuning a GPT-based
model, it aims to generate coherent and creative lyrics matching the desired
sentiment. The approach involves using sentiment labels as prefixes during
fine-tuning to help the model learn the relationship between sentiment and lyrics. The
soundness of the task relies on the GPT model's capabilities, sentiment-guided
generation, and fine-tuning on a domain-specific dataset, resulting in a reliable model
for sentiment-specific lyrics generation.

The implementation makes sense because it leverages the strengths of GPT-based
models and follows a well-established fine-tuning process to learn the intricacies of
song lyrics and the relationship between sentiment and lyrics. This approach enables
the generation of creative, coherent, and sentiment-specific lyrics
The dataset we used for training the model is “lyrics-data.csv”[1]. Below is a
representation of the dataset.

II. Methodology
We first removed unnecessary columns (refer to Figure 1) namely, “ALink”,
“SName”, and “SLink” as they are not necessary for this task. Next, we dropped all
the languages except English as it is the only language we are interested in for this
task.

Since we want to generate lyrics with a specific sentiment attached to it we
implemented that using TextBob library, which is a simple and convenient tool for
natural language processing tasks, including sentiment analysis. TextBlob calculates
sentiment polarity, which is a score ranging from -1 (most negative) to 1 (most
positive), with 0 representing neutral sentiment. Here, 0 was the threshold, if the
score > 0 then “happy” else if the score < 0 “sad”, else “neutral”.

On further data analysis, we discovered that there was a significant class imbalance.

In order to avoid any kind of bias in our model we downsampled “happy” instances to
bring it closer to the number of “sad” instances and we completely dropped “neutra”
instances as it was very small in quantity.

Finally, we concatenated the “sentiment” column with the “Lyric” column as such,
"<happy/sad>: <yrics>". 

In the process of fine-tuning, the GPT model acquires knowledge of the correlation
sentiment as a prefix to the lyrics and learns to produce lyrics that align with the given
sentiment.

As the model is fine-tuned, it takes in the concatenated input and makes an effort to
anticipate the subsequent token in the sequence by considering the context. The
sentiment prefix and the lyrics are both taken into account during this process,
allowing the model to identify the patterns and connections between them. As a
consequence, the model improves its ability to generate lyrics that match the desired
sentiment.

Below is a brief description of our fine-tuning methodology:
1. Load a pre-trained GPT-2 model from the Hugging Face Transformers library.
2. Define the training arguments, such as the number of training epochs, batch
size, and evaluation strategy, to configure the fine-tuning process.
3. Create a Trainer object from the Hugging Face library, which takes the
pre-trained model, training arguments, data collator, and datasets for training
and evaluation as inputs.
4. Fine-tune the GPT-2 model using the Trainer's train() method. This step
adjusts the pre-trained model's parameters based on the provided training
dataset, tailoring the model to the specific task of sentiment-guided lyrics
generation.

III. Evaluation

During training, we get the validation-set loss. We think this is a sign of overfitting as the training loss is monotonically decreasing.
This may not be a problem at all in case we do more training iterations as the mode
could be learning patterns in our training data set and hence causing a slight loss.
But in any case, we could be more cautious and apply some regularization and tune
the hyperparameters by adjusting learning rates, batch sizes, etc.

We split our training data into train and validation (80:20 split) data sets and will be
using perplexity for evaluation purposes. Using perplexity as an evaluation metric in
this sentiment-guided lyrics generation project makes sense as perplexity evaluates
the model's ability to predict the next token in a sequence, which is an essential
aspect of text generation which is essentially our task. Lower perplexity means the
model has learned the structure of the input text, including the lyrics and the
sentiment prefixes, more effectively.

GPT-based models are pre-trained on a large corpus of text and are fine-tuned for
specific tasks. Perplexity is a relevant metric for these models, as it measures how
well the fine-tuned model generalizes to the given dataset.
We got a perplexity score of 49.783832694649114, which is not a desired result but
given the amount of data we trained I think we can improve with more data,
regularization, and better hyperparameter tuning.


IV. Generated Lyrics
After training we finally generate our lyrics with a sentiment attached to them. 
The lyrics’ sentiment does seem to follow our expectations!

References
[1] Dataset, “lyrics-data.csv”,
https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres
[2] HuggingFace, “GPT-2”, https://huggingface.co/gpt2
[3] Shah, P. (2020, November 6). My Absolute Go-To for Sentiment Analysis—TextBlob.
Medium.
https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d
524
