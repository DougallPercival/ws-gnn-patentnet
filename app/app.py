from flask import Flask, request, jsonify, send_file, render_template

from os.path import join
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from util import return_most_similar_node, return_most_similar_patent, node_text
from PassageSim import SimilarityCompare

year_range_default = [1990, 1999]

default_weight_node =0.5
default_weight_tfidf =0.5
import re

text_df = pd.read_csv(join('data',"extracted_patent_text.csv"))

demo_nodes = [5702467, 5782543]
sim_nodes0 = ['5383937', '5489310', '5499205', '5548050', '5475777', '5940331', '5777249', '5712928', '5836451', '5670570']
sim_nodes1 = ['5470136', '5620239', '5839799', '5385393', '5403076', '5915800', '5531514', '5700069', '5472266', '5584541']
sim_node = list(text_df['PID'])
print('node', sim_node)



app = Flask(__name__, template_folder='web_page_interaction')


df_embeding_node = pd.read_csv(join('data', 'embeding.csv'), index_col = 0)
df_embeding_tfidf = pd.read_csv(join('data', 'tfidf_svd_all.csv'), index_col = 0)

df_patent_year = pd.read_csv(join('data', 'patent_year.csv'))

df_patent_title = pd.read_csv(join('data', 'patent_title.csv'), index_col = 'PID')

set_patent_node = set(df_embeding_node.index)
set_patent_tfidf = set(df_embeding_tfidf.index)

possible_patent = set_patent_node.intersection(set_patent_tfidf)

def generate_url_img(patent_id):
    url = 'http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO1&Sect2=HITOFF&d=PALL&p=1&u=%2Fnetahtml%2FPTO%2Fsrchnum.htm&r=1&f=G&l=50&s1=5702467.PN.&OS=PN/5702467&RS=PN/5702467'
    img_url = 'https://pdfpiw.uspto.gov/.piw?Docid=05702467&homeurl=http%3A%2F%2Fpatft.uspto.gov%2Fnetacgi%2Fnph-Parser%3FSect1%3DPTO1%2526Sect2%3DHITOFF%2526d%3DPALL%2526p%3D1%2526u%3D%25252Fnetahtml%25252FPTO%25252Fsrchnum.htm%2526r%3D1%2526f%3DG%2526l%3D50%2526s1%3D5702467.PN.%2526OS%3DPN%2F5702467%2526RS%3DPN%2F5702467&PageNum=&Rtype=&SectionNum=&idkey=NONE&Input=View+first+page'
    return re.sub('5702467', str(patent_id), url), re.sub('5702467', str(patent_id), img_url)


@app.route('/compaire_part', methods=['POST', 'GET'])
def similarity_part():
    if request.method == 'GET':
        df_similar = pd.DataFrame()
    elif request.method == 'POST':
        patent_id = request.form.get('patent_id', type=float)
        compare_patent_id = request.form.get('compare_patent_id', type=float)
        if int(patent_id) not in sim_node:
            raise ValueError("demo not working in the possible patent for "+str(patent_id))
        if int(compare_patent_id) not in sim_node:
            raise ValueError("demo not working in the possible patent for "+str(patent_id))

        patent_id = int(patent_id)
        compare_patent_id = str(int(compare_patent_id))

        main_node_text = node_text(patent_id, text_df)[1]



        sim_node_pid = [compare_patent_id]
        t, text = node_text(compare_patent_id, text_df)
        sim_node_type = [t]
        sim_node_text = [text]

        stops = ['FIG', 'FIGS', 'invention', 'refer', 'referring', 'now', 'detail', 'description', 'described',
                'understood', 'illustrate', 'illustrated', 'depicted', 'embodiment']
        sc = SimilarityCompare(main_node_text, patent_id, sim_node_text, sim_node_type, sim_node_pid, add_stopwords=stops)
        sc.setup()
        sc.compare()
        most_similar = sc.getMostSimilar()
        df_similar = pd.DataFrame(most_similar,
                              columns=['PatendID', 'PatentPassage', 'SimPatID', 'SimPatTextType', 'SimPassage',
                                       'SimScore', 'tokens'])
        df_similar.drop(['SimScore', 'tokens'], axis=1, inplace=True)

    return render_template('predict_part.html', tables=[df_similar.to_html()], titles=df_similar.columns.values)


@app.route('/main_page', methods=['POST', 'GET'])
def similarity_page():
    if request.method == 'GET':
        df_similar = pd.DataFrame()
    elif request.method == 'POST':
        patent_id = request.form.get('patent_id', type = float)

        year_begin = request.form.get('year_begin', type = float)
        year_end = request.form.get('year_end', type=float)
        if year_begin is None:
            year_begin = year_range_default[0]
        if year_end is None:
            year_end = year_range_default[1]

        weight_node = request.form.get('weight_node', type = float)
        weight_tfidf = request.form.get('weight_tfidf', type=float)
        if weight_node is None:
            weight_node = default_weight_node
        if weight_tfidf is None:
            weight_tfidf = default_weight_tfidf



        cond1 = df_patent_year['GYEAR'] >= year_begin
        cond2 = df_patent_year['GYEAR'] >= year_end

        list_ID_filter = possible_patent.intersection(set(df_patent_year[cond1 & cond2]['PATENT']))
        list_ID_filter = list(list_ID_filter)



        patent_id = int(patent_id)
        df_similar = return_most_similar_patent(patent_id,
                                                df_embeding_1 = df_embeding_node,
                                                df_embeding_2 = df_embeding_tfidf,
                                                weight_1 = weight_node,
                                                weight_2 = weight_tfidf,
                                                df_title = df_patent_title,
                                                list_ID_filter = list_ID_filter)

        link_page = [generate_url_img(patent_id)[0] for patent_id in df_similar['patent ID']]
        link_image = [generate_url_img(patent_id)[1] for patent_id in df_similar['patent ID']]

        df_similar['link page'] = link_page
        df_similar['link image'] = link_image

    return render_template('predict_similarity.html', tables = [df_similar.to_html(formatters={'link page':lambda x:f'<a href="{x}">Link to patent</a>',
                                                                                               'link image':lambda x:f'<a href="{x}">Link to image</a>'}, escape=False)], titles=df_similar.columns.values)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)