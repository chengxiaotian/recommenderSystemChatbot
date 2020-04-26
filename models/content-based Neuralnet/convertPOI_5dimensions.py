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


f = open ( "poi-dataset.txt" , 'r',encoding = 'utf-8-sig')
l = []
l = [ line.strip().split("\t") for line in f]
poi_datset = numpy.array(l)

columns = poi_datset[0,1:].tolist()
df_poi = DataFrame(poi_datset[1:,1:],columns = columns)

"prepare for the 5 criteria dimensions"
nlp = spacy.load('en_core_web_lg')
doc_entertainment = nlp("entertainment")
doc_nature = nlp("nature")
doc_culture = nlp("culture")
doc_art = nlp("art")
doc_shopping = nlp("shopping")


######################### tags processing #####################################
tag_arry = numpy.zeros([180,5])
tag_columns = ["tags","entertainment_tag","nature_tag","culture_tag","art_tag","shopping_tag"]
tag_names = df_poi['tags'].to_frame()

for tag_index in range(len(df_poi['tags'])):
    
    nlp_tag = nlp(df_poi['tags'][tag_index])
    entertainment_similarity = doc_entertainment.similarity(nlp_tag)
    nature_similarity = doc_nature.similarity(nlp_tag)
    culture_similarity = doc_culture.similarity(nlp_tag)
    art_similarity = doc_art.similarity(nlp_tag)
    shopping_similarity = doc_shopping.similarity(nlp_tag)
    
    tag_arry[tag_index] = (entertainment_similarity,nature_similarity,culture_similarity,art_similarity,shopping_similarity)
    
df_tags_vectors = DataFrame(tag_arry)
df_tags_full = tag_names.join(df_tags_vectors)
df_tags_full.columns = tag_columns

df1 = df_poi['nameofPlace'].to_frame().join(df_tags_full)

######################### reviews processing ###################################
desc_arry = numpy.zeros([180,5])
place_columns = ["nameofPlace","entertainment_desc","nature_desc","culture_desc","art_desc","shopping_desc"]
place_names = df_poi['nameofPlace'].to_frame()


doc = nlp(df_poi['Desc'][0])
newwordString = ""
for token in doc:
    if token.is_stop != True and token.is_punct != True:
        newwordString = newwordString + " "+token.text

for desc_index in range(len(df_poi['Desc'])):
    
    nlp_desc = nlp(df_poi['Desc'][desc_index])
    entertainment_similarity = doc_entertainment.similarity(nlp_desc)
    nature_similarity = doc_nature.similarity(nlp_desc)
    culture_similarity = doc_culture.similarity(nlp_desc)
    art_similarity = doc_art.similarity(nlp_desc)
    shopping_similarity = doc_shopping.similarity(nlp_desc)
    
    desc_arry[desc_index] = (entertainment_similarity,nature_similarity,culture_similarity,art_similarity,shopping_similarity)
    
df_desc_vectors = DataFrame(desc_arry)
df_desc_full = place_names.join(df_desc_vectors)
df_desc_full.columns = place_columns

df2 = df_desc_full

########################## complete POI dimentsion DF #########################
combine_arry = numpy.zeros([180,5])
"0.3 influence of desc and 0.7 influence of tag"
ratio = 0.3 

for place_index in range(len(combine_arry)):
    
    for tag_index in range(len(combine_arry[place_index])):
        
        calculation = (1-ratio)*df_tags_vectors.iloc[place_index][tag_index] + \
                    ratio*df_desc_vectors.iloc[place_index][tag_index]
        combine_arry[place_index][tag_index] = calculation
    
complete_columns = ["nameofPlace","tags","entertainment","nature","culture","art","shopping"]
name_tags_df = df_poi.iloc[:,0:2]

complete_df_POI = name_tags_df.join(DataFrame(combine_arry))
complete_df_POI.columns = complete_columns

    
