from os.path import join
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def return_most_similar_node(patent_id, df_embeding, nb_result=10):

    dist = cosine_similarity(df_embeding.loc[patent_id].to_numpy().reshape(1, 128), df_embeding).reshape(-1, )
    df_result = pd.DataFrame({'patent ID':df_embeding.index, 'similarity': dist}).sort_values(by = 'similarity' , ascending = False)
    return df_result.head(nb_result)


def rank_patent(patent_id, df_embeding, list_ID_filter=None):
    dist = cosine_similarity(df_embeding.loc[patent_id].to_numpy().reshape(1, 128), df_embeding).reshape(-1, )
    df_result = pd.DataFrame({'patent ID': df_embeding.index, 'score': dist})

    df_result_order = df_result.sort_values(by='score', ascending=False)
    df_result_order = df_result_order.reset_index()
    df_result_order['score'] = df_result_order.index

    list_patent_id = list(df_result['patent ID'])
    df_result.index = list_patent_id
    df_result = df_result.loc[df_result_order['patent ID']]
    df_result['score'] = list(df_result_order['score'])
    df_result = df_result.loc[list_patent_id]

    return df_result


def return_most_similar_patent(patent_id,
                               df_embeding_1,
                               df_embeding_2,
                               weight_1=0.5,
                               weight_2=0.5,
                               nb_result=10,
                               df_title=None,
                               list_ID_filter=None):

    df_score1 = rank_patent(patent_id, df_embeding_1, list_ID_filter=list_ID_filter)
    df_score2 = rank_patent(patent_id, df_embeding_2, list_ID_filter=list_ID_filter)

    df_score1['score'] = weight_1 * df_score1['score'] + weight_2 * df_score2['score']

    df_score1 = df_score1.sort_values(by='score', ascending=True)

    df_score1 = df_score1.head(nb_result)

    if df_title is not None:
        df_score1['title'] = list(df_title.loc[df_score1['patent ID']]['TTL'])

    return df_score1


def node_text(node, df):
    """
    Grab the text from the dataframe for a given node
    """
    for row in df.itertuples():
        if df.at[row[0], 'PID'] == int(node):
            return (df.at[row[0], 'TEXT_TYPE'], df.at[row[0], 'TEXT'])

    return (None, None)