from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
import os


#get english stopwords
en_stopwords = set(stopwords.words('english'))


def rightTypes(ngram):
    """
    - filter for ADJ/NN bigrams
    - usage: filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]
    """
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False


def rightTypesTri(ngram):
    """
    - filter for trigrams
    - usage: filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]
    """
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False


def text_collocations(nltk_text):
    """
    Find the most common phrases, using bigram
    """

    print(nltk_text.collocations())

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(nltk_text)
    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(nltk_text)

    # only bigrams that appear 10+ times
    bigramFinder.apply_freq_filter(10)
    trigramFinder.apply_freq_filter(10)

    #bigrams
    bigram_freq = bigramFinder.ngram_fd.items()
    bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram','freq']).sort_values(by='freq', ascending=False)

    #trigrams
    trigram_freq = trigramFinder.ngram_fd.items()
    trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram','freq']).sort_values(by='freq', ascending=False)

    bigramPMITable = pd.DataFrame(list(bigramFinder.score_ngrams(bigram_measures.pmi)), columns=['bigram','PMI']).sort_values(by='PMI', ascending=False)
    trigramPMITable = pd.DataFrame(list(trigramFinder.score_ngrams(trigram_measures.pmi)), columns=['trigram','PMI']).sort_values(by='PMI', ascending=False)

    bigramTtable = pd.DataFrame(list(bigramFinder.score_ngrams(bigram_measures.student_t)), columns=['bigram','t']).sort_values(by='t', ascending=False)
    trigramTtable = pd.DataFrame(list(trigramFinder.score_ngrams(trigram_measures.student_t)), columns=['trigram','t']).sort_values(by='t', ascending=False)

    #filters
    filteredT_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]
    filteredT_tri = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: rightTypesTri(x))]

    filteredT_bi['text'] = filteredT_bi['bigram'].apply(lambda x: ' '.join(x))
#    print(filteredT_bi.head())

    filteredT_tri['text'] = filteredT_tri['trigram'].apply(lambda x: ' '.join(x))
#    print(filteredT_tri.head())
#    print('***************************')
#    for k,v in finder.ngram_fd.items():
#        print(k,v)
    # return the 500 n-grams with the highest PMI
#    L = bigramFinder.nbest(bigram_measures.pmi, 500)

    return filteredT_bi, filteredT_tri


def word_count(nltk_text):
    """
    Count frequency of word by type
    """

    text_collocations_list = text_collocations(nltk_text)

    tagged = nltk.pos_tag(nltk_text)
    tag_fd = nltk.FreqDist(word for (word, tag) in tagged if tag in ('JJ', 'JJR', 'JJS'))
    common_word = tag_fd.most_common()

    return common_word


def word_cloud(df):
    """
    Word cloud plot from frequency table
    """

    d = dict(zip(df['text'], df['freq']))

    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(d)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    #plt.show()
    plt.savefig('propn.png', bbox_inches='tight')
    return plt


def main():
    input_file = '../github/colectica_api/RCNIC_covid_lable.csv'
    df_input = pd.read_csv(input_file, sep='\t')
    # lower case
    df_input = df_input.astype(str).apply(lambda x: x.str.lower())

    # combine question literal with response
    df_input['words'] = df_input['QuestionLiteral'] + df_input['Response']

    df_qg = df_input.loc[df_input['QuestionGroupLabel'] == 'mental health and mental processes', ['words']]
    question_text = df_qg['words'].str.cat(sep=' ')

    # keep word only
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(question_text)

    #tokens = word_tokenize(question_text)
    nltk_text = nltk.Text(tokens)
    # print(nltk_text)

    output_dir = 'word_count_nltk'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    common_words = word_count(nltk_text)
    print(common_words)

    filteredT_bi, filteredT_tri = text_collocations(nltk_text)
    print(filteredT_bi.head())
    print(filteredT_tri.head())

    filteredT_bi.to_csv(os.path.join(output_dir, 'bigramTtable.csv'), index=False)
    filteredT_tri.to_csv(os.path.join(output_dir, 'trigramTtable.csv'), index=False)

    plt_bi = word_cloud(filteredT_bi)
    plt_bi.savefig(os.path.join(output_dir, 'tow_words.pdf'))

    plt_tri = word_cloud(filteredT_bi)
    plt_tri.savefig(os.path.join(filteredT_tri, 'three_words.pdf'))


if __name__ == '__main__':
    main()
