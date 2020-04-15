# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:57:23 2019

@author: Xiao
For scraping reviews. 
Input: POI links read from file
"""
#import os
#os.getcwd()
from selenium import webdriver
from scrapy import Selector
import time

fileopen = "a.txt"
with open(fileopen,"r",encoding='utf-8-sig') as file:
    POILinks = file.readlines()

#expand all contents
browser = webdriver.Chrome('C:\\Users\\User\\Downloads\\chromedriver_win32\\chromedriver')
completeReviewList = []
for i in range(len(POILinks)):
    url = POILinks[i].strip()

    browser.get(url)
    try:
        browser.execute_script("document.querySelector('div.entry > p > span').click();")
        time.sleep(1)
        new_page = browser.page_source
    except:
        new_page = browser.page_source
    
    #scrape started
    sel = Selector(text = new_page)
    
    reviewPagesxPath = '//div[@class = "pageNumbers"]//a[@class = "pageNum taLnk "]/@href'
    reviewPages = sel.xpath(reviewPagesxPath).extract()
    
    finalLinks = []
    forelink = "https://www.tripadvisor.com.my"

    for e in reviewPages:
        if e != '':
            finalLinks.append(forelink+e)
    finalLinks.append(url)
    
    reviewList = []
    for pageLink in finalLinks:
        
        url = pageLink
        browser.get(url)
        try:
            browser.execute_script("document.querySelector('div.entry > p > span').click();")
            time.sleep(1)
            new_page = browser.page_source
        except:
            new_page = browser.page_source 
        
        time.sleep(1)
        new_page = browser.page_source
        
        sel = Selector(text = new_page)
        
        reviewxPath = '//div[@class = "reviewSelector"]'
        reviews = sel.xpath(reviewxPath)
        
        titlexPath = '//*[@id="HEADING"]/text()'
        POItitle = sel.xpath(titlexPath).extract_first()
        
    
        
        usernamexPath = './/div[@class = "info_text"]/div/text()'
        datexPath = './/span[@class = "ratingDate"]/@title'
        titlexPath = './/span[@class = "noQuotes"]/text()'
        textxPath = './/p[@class = "partial_entry"]//text()'
        ratingxPath = './/span[contains(@class,"ui_bubble_rating bubble")]/@class'
        
        for j in range(len(reviews)):
            
            username = reviews[j].xpath(usernamexPath).extract_first()
            date = reviews[j].xpath(datexPath).extract_first()
            title = reviews[j].xpath(titlexPath).extract_first()
            text = reviews[j].xpath(textxPath).extract()
            cleanText = []
            for t in text:
                tt = t.replace('\n','')
                cleanText.append(tt)
                
            ratingString = reviews[j].xpath(ratingxPath).extract_first()
            ratings = ratingString.split("_")
            rating = ratings[-1]
    
            reviewEntity = {
                    'POItitle':POItitle,
                    'username':username,
                    'date':date,
                    'title':title,
                    'text':cleanText,
                    'rating':rating,
                    'link':pageLink
                    }
            
            completeReviewList.append(reviewEntity)
            
            
            
        
#end of loop
browser.quit()


fileWriteReviews = "reviews.txt"
with open(fileWriteReviews,"w",encoding='utf8') as file:
    for eachReview in completeReviewList:
            
        writeline = eachReview['POItitle']+"\t"+eachReview['username']\
        +"\t"+eachReview['date']+"\t"+eachReview['title']+"\t"\
        +",".join(eachReview['text'])+"\t"+eachReview['rating']+"\t"+eachReview['link']+"\n"
        file.write(writeline)
    
