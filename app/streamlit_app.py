import streamlit as st
import re

from os.path import join
import pandas as pd

from util import return_most_similar_node, return_most_similar_patent, node_text
#from PassageSim import SimilarityCompare


@st.cache
def load_data():

    df_embeding_node = pd.read_csv(join('data', 'embeding.csv'), index_col = 0)
    df_embeding_tfidf = pd.read_csv(join('data', 'tfidf_svd_all.csv'), index_col=0)

    df_patent_year = pd.read_csv(join('data', 'patent_year.csv'))

    df_patent_title = pd.read_csv(join('data', 'patent_title.csv'), index_col='PID')

    set_patent_node = set(df_embeding_node.index)
    set_patent_tfidf = set(df_embeding_tfidf.index)

    possible_patent = set_patent_node.intersection(set_patent_tfidf)

    return df_embeding_node, df_embeding_tfidf, possible_patent, df_patent_year, df_patent_title


def generate_url_img(patent_id):
    url = 'http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO1&Sect2=HITOFF&d=PALL&p=1&u=%2Fnetahtml%2FPTO%2Fsrchnum.htm&r=1&f=G&l=50&s1=5702467.PN.&OS=PN/5702467&RS=PN/5702467'
    img_url = 'https://pdfpiw.uspto.gov/.piw?Docid=05702467&homeurl=http%3A%2F%2Fpatft.uspto.gov%2Fnetacgi%2Fnph-Parser%3FSect1%3DPTO1%2526Sect2%3DHITOFF%2526d%3DPALL%2526p%3D1%2526u%3D%25252Fnetahtml%25252FPTO%25252Fsrchnum.htm%2526r%3D1%2526f%3DG%2526l%3D50%2526s1%3D5702467.PN.%2526OS%3DPN%2F5702467%2526RS%3DPN%2F5702467&PageNum=&Rtype=&SectionNum=&idkey=NONE&Input=View+first+page'
    return re.sub('5702467', str(patent_id), url), re.sub('5702467', str(patent_id), img_url)


df_embeding_node, df_embeding_tfidf, possible_patent, df_patent_year, df_patent_title = load_data()


if __name__ == '__main__':


    st.title("Welcome to patent search")



    #st.dataframe(df_embeding_node.head())

    #st.sidebar.slider("price", 0, 1)

    patent_ID = st.sidebar.number_input('Patent ID (ex: 5702467)', value =5702467, step=1)

    st.sidebar.write('The search can be based on the more on the similarity of the citation or base on the text')

    st.sidebar.write('Weight 0 mean search base only on citation, weight 1 mean search base only on text')

    model_weight = st.sidebar.slider('Citation/text ratio weight',0.0, 1.0, step = 0.01)


    year_range = st.sidebar.slider('year range search',1990, 1999, (1990, 1999))

    st.write('the search of the patent ID ', patent_ID)





    cond1 = df_patent_year['GYEAR'] >= year_range[0]
    cond2 = df_patent_year['GYEAR'] >= year_range[1]

    list_ID_filter = possible_patent.intersection(set(df_patent_year[cond1 & cond2]['PATENT']))
    list_ID_filter = list(list_ID_filter)


    df_similar = return_most_similar_patent(patent_ID,
                                            df_embeding_1=df_embeding_node,
                                            df_embeding_2=df_embeding_tfidf,
                                            weight_1=1-model_weight,
                                            weight_2=model_weight,
                                            df_title=df_patent_title,
                                            list_ID_filter=list_ID_filter)

    link_page = [generate_url_img(patent_id)[0] for patent_id in df_similar['patent ID']]
    link_image = [generate_url_img(patent_id)[1] for patent_id in df_similar['patent ID']]

    df_similar['link page'] = link_page
    df_similar['link image'] = link_image

    df_result = df_similar.to_html(formatters={'link page':lambda x:f'<a href="{x}">Link to patent</a>',
                                  'link image':lambda x:f'<a href="{x}">Link to image</a>'}, escape=False)



    st.write(df_result, unsafe_allow_html=True)

