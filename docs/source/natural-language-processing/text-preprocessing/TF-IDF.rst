===================================================================================================
TF-IDF (Term Frequency * Inverse Document Frequency)
===================================================================================================


What's the setting?
---------------------------------------------------------------------------------------------------
- You have a collection of documents, let's call that :math:`C`. This is also called a corpus.
- Each document :math:`D`: is a list of "term"s (i.e. words). A document could be a sentence, a paragraph, a research paper, a book, whatever.
- The set of unique words across your corpus is called your vocabulary :math:`V`. :math:`W` is a particular word in the vocabulary, i.e. :math:`W \in V`.


Why are we calculating this?
---------------------------------------------------------------------------------------------------
We want to find out which words are part of the "jargon" of the documents you're reading. These are words which you see popping up again and again, but they aren't part of the normal English vocabulary.

Words like "the", "at", "who" are not words the words you are looking for. They appear a **lot**, but they appear **everywhere**. We want to find words which appear a **lot** within **a small number of documents**. TF-IDF helps highlight the second kind of words.

How do you calculate it?
---------------------------------------------------------------------------------------------------
TF-IDF is computed for each word :math:`W` in a particular document :math:`D`.

It is a multiplication of two scores: Term Frequency and Inverse Document Frequency

Term Frequency
===================================================================================================

**Term Frequency** is the count of the word :math:`W` in document :math:`D`, normalized by the number of words in the document.

  .. math::

    \text{TF}(W, D) = \frac{count(W \text{ in } D)}{len(D)}

    
Note, we normalize by the length of the document, since we want to make a fair comparison between documents of different lengths. 

- If we just used the raw counts of a word :math:`count(W \text{ in } D)`, then a longer document (like a textbook) would have a much larger effect on the value than a shorter one (like a Wikipedia article) since it is likely to have more occurrences of almost any word :math:`W`. We want to give all documents an equal weight, so we normalize by the length of the document. 

- This also allows us to compare TF values between documents, i.e. :math:`TF(W, D_1)`, :math:`TF(W, D_2)` etc. are now directly comparable to each other.



Inverse Document Frequency
===================================================================================================

**Document frequency** is simple: it is the number of documents in the corpus :math:`C`, that contain the word :math:`W` at least once. 

We normalize this value by the total number of documents in the corpus, so that this value can be compared across corpora.

  .. math::

    \text{DF}(W, C) = \frac{ count(D \text{ where } count(W, D) >= 1)}{len(C)}

Note that we get one value for each word in the corpus.

Document frequency calculates what percentage of the corpus has this word, i.e. "how common is this word in our corpus". 

- If our corpus just contains repitions of the sentence "Green eggs and ham", the DF of all four words "Green", "eggs", "and", "ham" will be 1, i.e. 100%.

- In a more realistic corpus, very common words like "the", "at", "who" etc. will have high DF values like 0.978, 0.994, etc. If our corpus is selected from issues of "Automobile Weekly" magazine, words like "auto" and "drive" might also have high DF values like 0.89, 0.864, etc.

**Inverse Document Frequency** is just the opposite; it calculates "how rare is this word in our corpus".

  .. math::

    \text{IDF}(W, C) = \log \left[ {\frac{1}{\text{DF}(W, C)}} \right] =  \log \left[ \frac{len(C)}{ count(D \text{ where } count(W, D) >= 1)} \right]

Note that we take the log since the numerator can be large when our corpus is big, so we want to scale down these values. The base of the log does not really matter.

- The denominator must be at least one, i.e. we can only calculate IDF for words which occur at least once in the corpus.

- The IDF value is maximum for words which occur only once in the corpus. Common words are given a smaller weight.

Assuming you are dealing with a single, fixed corpus :math:`C`, you can pre-compute the IDF values for every word in your vocabulary :math:`V`. This is usually denoted by :math:`\text{IDF}(W)`.


TF-IDF
===================================================================================================


TF-IDF is a simple multiplication of Term Frequency and Inverse Document Frequency.

It is a composite metric with a value for each word in a particular document, in a particular corpus. 

We can denote this by :math:`\text{TF-IDF}(W, D, C)`, or simply :math:`\text{TF-IDF}(W, D)` if we are dealing with a single, fixed corpus (which is usually the case).


    .. math::

      \text{TF-IDF}(W, D) = \text{TF}(W, D) \times \text{IDF}(W)


Example calculation
---------------------------------------------------------------------------------------------------

Let's calculate this with an example [1]_:

1. Consider a document in our corpus, say the poem "The Tale of Frisky Whiskers". It contains 100 words wherein the word "cat" appears 3 times, i.e. 

  .. math::

    len(D_{\text{FriskyWhiskers}}) = 100

    count(\text{W="cat"} \text{ in } D_{\text{FriskyWhiskers}} ) = 3

2. The term frequency (i.e. TF) for "cat" in this poem is: 
    
  .. math::

    \text{TF}(\text{W="cat"}, D_{\text{FriskyWhiskers}}) = \frac{3}{100} = 0.03

3. Now, assume we have 10 million documents and the word "cat" appears in 1,000 of these. Then, the inverse document frequency (i.e. IDF) is calculated as: 


  .. math::

    \text{IDF}(\text{W="cat"}) = \log \left[ \frac{10,000,000}{1,000} \right] = 4.0

4. Thus, the Tf-IDF weight is the product of these quantities:

  .. math::

    \text{TF-IDF}(\text{W="cat"}, D_{\text{FriskyWhiskers}}) = \text{TF}(\text{W="cat"}, D_{\text{FriskyWhiskers}}) \times \text{IDF}(\text{W="cat"}) \\
      = 0.03 \times 4.0 \\
      = 0.12


What do different TF-IDF values indicate?
===================================================================================================

TF-IDF tries to weigh down words which occur in most documents, and weight up those which occur frequently in a small, clustered set of documents.

More specifically, the value of :math:`\text{TF-IDF}(W, D)` is [2]_:

1. Highest when :math:`W` occurs many times in document :math:`D`, and only occurs within a small number of documents (thus lending high discriminating power to those documents).
    - Note that the TF-IDF value will be high for a word only in the documents where it occurs frequently, not in all documents.
    - The maximum possible TF-IDF is for a document which contains just a single word which is found nowhere else in the corpus. In this case: 
    
      .. math::
    
        \text{TF-IDF}_{\text{max possible}} = \frac{1}{1} \times \log \left[ \frac{len(C)}{1} \right] = \log \left( len(C) \right) 

    - Thus the max possible value is fixed for a particular corpus, but might vary from corpus to corpus.

2. Lower when the word occurs fewer times in a document, or occurs in many documents (thus offering a less pronounced relevance signal).

3. Lowest when the term occurs in virtually all documents.
    - The minimum possible TF-IDF value is for a word which is present in every document in the corpus (the number of times does not matter). In this case: 
        
      .. math::
    
        \text{TF-IDF}_{\text{min possible}} = \text{TF}(W, D) \times \log \left[ \frac{len(C)}{len(C)} \right] \\
         = \text{TF}(W, D) \times \log \left( 1 \right) \\
         = \text{TF}(W, D) \times 0 \\
        \therefore \text{TF-IDF}_{\text{min possible}} = 0
    
    - As **neither TF not IDF can be negative**, the minimum value of TF-IDF for a word in a document is thus 0.



TF-IDF vectors
---------------------------------------------------------------------------------------------------

Some ML libraries have a text preprocessor utility for creating TF-IDF vectors from each input piece of text (in sklearn, this is TFIDFVectorizer). 

What they do is, simply, take in a given corpus (i.e. a list of strings) and calculate the IDF values of every word in the vocabulary of the corpus. 

Then, when you input a particular document (i.e. a string) from the **same** corpus, it will output a TF-IDF score for each word in the document. For words not in the document, it will output zero.


.. jupyter-execute::

    import pandas as pd
    from IPython.display import display
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'And this is the...first document?',
    ]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    tfidf_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names()).round(2)
    tfidf_df.index = corpus
    display(tfidf_df)

Each row here is a TF-IDF vector, which can be used as a fixed-size numeric representation of each text document (and hence be used in several Machine Learning algorithms).

References
---------------------------------------------------------------------------------------------------
.. [1] http://www.tfidf.com/
.. [2] [Manning, Manning, Schutze][2008] Introduction to Information Retrieval, section 6.2.2 Tf–idf weighting