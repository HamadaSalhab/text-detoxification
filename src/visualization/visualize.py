import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Plot distribution of toxicity for reference and translation texts
def plot_txc_distr(df):

    plt.figure(figsize=(14, 6))

    # For reference
    plt.subplot(1, 2, 1)
    sns.histplot(df['ref_tox'], kde=True, bins=30, color='blue')
    plt.title('Distribution of Toxicity for Reference Texts')
    plt.xlabel('Toxicity Level')
    plt.ylabel('Number of Texts')

    # For translation
    plt.subplot(1, 2, 2)
    sns.histplot(df['trn_tox'], kde=True, bins=30, color='green')
    plt.title('Distribution of Toxicity for Translation Texts')
    plt.xlabel('Toxicity Level')
    plt.ylabel('Number of Texts')

    plt.tight_layout()
    plt.show()

def plot_lengths(df, max_range=None):
    # Calculate text lengths
    df['reference_length'] = df['reference'].apply(len)
    df['translation_length'] = df['translation'].apply(len)

    # Plotting distribution of text lengths for reference and translation texts
    plt.figure(figsize=(14, 6))

    # For reference text length
    plt.subplot(1, 2, 1)
    sns.histplot(df['reference_length'] if not max_range else df[df['reference_length'] <= max_range]['reference_length'], kde=True, bins=30, color='blue')
    plt.title('Distribution of Text Length for Reference Texts')
    plt.xlabel('Text Length')
    plt.ylabel('Number of Texts')

    # For translation text length
    plt.subplot(1, 2, 2)
    sns.histplot(df['reference_length'] if not max_range else df[df['reference_length'] <= max_range]['reference_length'], kde=True, bins=30, color='green')
    plt.title('Distribution of Text Length for Translation Texts')
    plt.xlabel('Text Length')
    plt.ylabel('Number of Texts')

    plt.tight_layout()
    plt.show()

def plot_cos_similarity(df):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Assuming you have two lists: reference_texts and translation_texts
    reference_texts = df['reference']
    translation_texts = df['translation']

    # Use TF-IDF Vectorizer to convert texts into vector format
    vectorizer = TfidfVectorizer().fit(reference_texts + translation_texts)
    reference_vecs = vectorizer.transform(reference_texts)
    translation_vecs = vectorizer.transform(translation_texts)

    # Compute the cosine similarity between each pair
    similarities = [cosine_similarity(reference_vecs[i], translation_vecs[i])[0][0] for i in range(len(reference_texts))]

    # Plot the distribution
    plt.figure(figsize=(10, 5))
    plt.hist(similarities, bins=100, color='skyblue', edgecolor='black')
    plt.title('Distribution of Cosine Similarities between Reference and Translation Texts')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Number of Texts')
    plt.show()
    
def plot_word_cloud(df):
    from collections import Counter
    from wordcloud import WordCloud
    import nltk
    from nltk.corpus import stopwords


    # If you haven't downloaded the stopwords from nltk, do so
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        # Tokenize, convert to lowercase, and remove stopwords
        return [word for word in nltk.word_tokenize(text.lower()) if word.isalpha() and word not in stop_words]

    def generate_wordcloud(column_name):
        # Flatten the list of words and create a Counter
        words = [word for sublist in df[column_name].apply(preprocess) for word in sublist]
        word_freq = Counter(words)
        
        # Generate word cloud
        wordcloud = WordCloud(background_color="white", width=800, height=800).generate_from_frequencies(word_freq)
        
        # Plot
        plt.figure(figsize=(8, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud for {column_name}")
        plt.show()
        
        return word_freq

    # Generate word clouds and get word frequencies
    ref_word_freq = generate_wordcloud("reference")
    trans_word_freq = generate_wordcloud("translation")

    # Finding rare words (e.g., words that appear only once)
    rare_words_ref = [word for word, freq in ref_word_freq.items() if freq == 1]
    rare_words_trans = [word for word, freq in trans_word_freq.items() if freq == 1]

    print("Rare words in Reference:", rare_words_ref)
    print("Rare words in Translation:", rare_words_trans)

def main():
    RAW_DATASET_PATH = './data/raw/filtered_paranmt/filtered.tsv'

    df = pd.read_csv(RAW_DATASET_PATH, delimiter='\t')

    # Call your functions with the dataframe
    plot_txc_distr(df)
    plot_lengths(df)
    plot_cos_similarity(df)
    plot_word_cloud(df)

if __name__ == '__main__':
    main()