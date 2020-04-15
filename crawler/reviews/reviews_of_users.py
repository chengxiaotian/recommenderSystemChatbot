# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:47:41 2019

@author: Xiao
"""
from scrapy import Selector
import requests


fileopen = "userprofile_links.txt"
with open(fileopen,"r",encoding='utf-8-sig') as file:
    userprofile_links = file.readlines()
    
entityList = []

for url in userprofile_links:

    html = requests.get(url).content
    sel = Selector(text = html)
    
    reviewxPath = '//div[@class = "social-sections-ReviewSection__review--3qryC"]'
    
    
    reviews = sel.xpath(reviewxPath)
    titlexPath = './div[1]/text()'
    textxPath = './div[2]/q/text()'
    
    
    for review in reviews:
        title = review.xpath(titlexPath).extract_first()
        text = review.xpath(textxPath).extract_first()
        
        entity = {'title':title,'text':text,'url':url}
        entityList.append(entity)

fileWriteReviews = "user_reviews.txt"
with open(fileWriteReviews,"w",encoding='utf8') as file:
    for eachReview in entityList:
            
        writeline = eachReview['title']+"\t"+eachReview['text']\
        +"\t"+eachReview['url']+"\t"+"\n"
        file.write(writeline)