# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:06:29 2019

@author: Xiao
"""
'''
Convert POI tags and desc/reviews to 5 dimension vectors
dimensions:
1.entertainment
2.nature
3.culture
4.art
5.shopping
'''
import numpy
from pandas import DataFrame
import spacy
from spacy.lang.en import English


f = open ( "user_dataset.txt" , 'r',encoding = 'utf-8-sig')
l = []
l = [ line.strip().split("\t") for line in f]
user_dataset = numpy.array(l)

columns = user_dataset[0,1:].tolist()
df_user = DataFrame(user_dataset[1:,1:],columns = columns)

"prepare for the 5 criteria dimensions"
nlp = spacy.load('en_core_web_lg')
doc_entertainment = nlp("entertainment")
doc_nature = nlp("nature")
doc_culture = nlp("culture")
doc_art = nlp("art")
doc_shopping = nlp("shopping")


######################### tags processing #####################################
tag_arry = numpy.zeros([146,5])
tag_columns = ["tags","entertainment_tag","nature_tag","culture_tag","art_tag","shopping_tag"]
tag_names = df_user['tags'].to_frame()

for tag_index in range(len(df_user['tags'])):
    
    nlp_tag = nlp(df_user['tags'][tag_index])
    entertainment_similarity = doc_entertainment.similarity(nlp_tag)
    nature_similarity = doc_nature.similarity(nlp_tag)
    culture_similarity = doc_culture.similarity(nlp_tag)
    art_similarity = doc_art.similarity(nlp_tag)
    shopping_similarity = doc_shopping.similarity(nlp_tag)
    
    tag_arry[tag_index] = (entertainment_similarity,nature_similarity,culture_similarity,art_similarity,shopping_similarity)
    
df_tags_vectors = DataFrame(tag_arry)
df_tags_full = tag_names.join(df_tags_vectors)
df_tags_full.columns = tag_columns

df1 = df_user['name'].to_frame().join(df_tags_full)

######################### reviews processing ###################################
review_arry = numpy.zeros([146,5])
place_columns = ["name","entertainment","nature","culture","art","shopping"]
place_names = df_user['name'].to_frame()

for desc_index in range(len(df_user['review'])):
    
    nlp_desc = nlp(df_user['review'][desc_index])
    entertainment_similarity = doc_entertainment.similarity(nlp_desc)
    nature_similarity = doc_nature.similarity(nlp_desc)
    culture_similarity = doc_culture.similarity(nlp_desc)
    art_similarity = doc_art.similarity(nlp_desc)
    shopping_similarity = doc_shopping.similarity(nlp_desc)
    
    review_arry[desc_index] = (entertainment_similarity,nature_similarity,culture_similarity,art_similarity,shopping_similarity)
    
df_desc_vectors = DataFrame(review_arry)
df_desc_full = place_names.join(df_desc_vectors)
df_desc_full.columns = place_columns

df2 = df_desc_full

########################## complete POI dimentsion DF #########################
combine_arry = numpy.zeros([146,5])
"0.3 influence of desc and 0.7 influence of tag"
ratio = 0.3 

for user_index in range(len(combine_arry)):
    for tag_index in range(len(combine_arry[user_index])):
        if df1['tags'][user_index].isnumeric():
            calculation = df_desc_vectors.iloc[user_index][tag_index]
        else:
            calculation = (1-ratio)*df_tags_vectors.iloc[user_index][tag_index] + \
                    ratio*df_desc_vectors.iloc[user_index][tag_index]     
                    
        combine_arry[user_index][tag_index] = calculation
    
complete_columns = ["name","tags","entertainment","nature","culture","art","shopping"]
name_tags_df = df_user.iloc[:,0:2]

complete_df_user = name_tags_df.join(DataFrame(combine_arry))
complete_df_user.columns = complete_columns

    
