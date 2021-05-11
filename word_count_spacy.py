from wordcloud import WordCloud
import matplotlib.pyplot as plt

import pandas as pd
import spacy
import os

nlp = spacy.load("en_core_web_sm")

def word_count(input_text):
    """
    Count frequency of word by type
    """
    doc = nlp(input_text)

    rows_column = ['text', 'base_form', 'coarse_grained', 'fine_grained', 'dependency']

    rows_list = []
    for token in doc:
        rows_list.append([token.text.lower(), token.lemma_, token.pos_, token.tag_, token.dep_])

    df = pd.DataFrame(rows_list, columns=rows_column)   

    print(df['coarse_grained'].unique())

    df_sub = df.loc[df['coarse_grained'] == 'PROPN', ['text']]

    df_out = df_sub['text'].value_counts(dropna=True, sort=True).rename_axis('text').reset_index(name='counts')

    return df_out   


def word_cloud(df):
    """
    Word cloud plot from frequency table
    """

    d = dict(zip(df['text'], df['counts']))

    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies = d)

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
    df_qg = df_input.loc[df_input['QuestionGroupLabel'] == 'Income', ['QuestionLiteral']]
    question_text = df_qg['QuestionLiteral'].str.cat(sep=' | ')
    print(len(question_text))
    nlp.max_length = len(question_text) + 100

    output_dir = 'work_count'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_word_count = word_count(question_text)

    df_word_count.head(100).to_csv(os.path.join(output_dir, 'propn.csv'), index=False)

    plt = word_cloud(df_word_count)
    plt.savefig(os.path.join(output_dir, 'propn.pdf'))

if __name__ == '__main__':
    main()
