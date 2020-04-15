# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:01:41 2019

@author: Xiao
"""
import scrapy
from scrapy.crawler import CrawlerProcess




# Create the Spider class
class POISpider(scrapy.Spider):
    name = "tripadvisor spider"
    # start_requests method
    def start_requests(self):
        url = "https://www.tripadvisor.com.my/Attractions-g294212-Activities-Beijing.html"
        yield scrapy.Request(url = url,callback = self.parse_front)
                         
    # First parsing method
    def parse_front(self, response):
        
#        #scrape for different pages
#        pagexPath = '//div[contains(@class,"attractions-attraction-overview-main-Pagination")]/a/@href'
#        pageLinks = response.xpath(pagexPath).extract()
        
        #scrape for details on first page
        itemxPath = '//li[contains(@class,"attractions-attraction-overview-main-TopPOIs__item")]'
        #link
        linkxPath = './/a[contains(@class,"attractions-attraction-overview-main-TopPOIs")]/@href'
        linkList = response.xpath(itemxPath).xpath(linkxPath).extract()
        
        
        
#        duplicateList = []
#        for i in range(len(pageLinks)):
#            if pageLinks[i] not in duplicateList:
#                duplicateList.append(pageLinks[i])
    
        finalLinks = []
        forelink = "https://www.tripadvisor.com.my"
        for e in linkList:
            finalLinks.append(forelink+e)
            
        for link in finalLinks:
            yield response.follow(url = link,callback = self.parse_details)
                    
    # Second parsing method, to scrape general information
    def parse_pages(self, response):
        #item path
        itemxPath = '//div[contains(@class,"listing_info")]'
        #item selector
        itemlist = response.xpath(itemxPath)

        #xpath for items to be scraped
        namexPath = './/div[contains(@class,"tracking_attraction_title listing_title")]/a/text()'
        tagsxPath = './/span[contains(@class,"matchedTag noTagImg")]/text()'
        linkxPath = './/div[contains(@class,"tracking_attraction_title listing_title")]/a/@href'


        for item in itemlist:
        
            name = item.xpath(namexPath).extract_first()
            tags = item.xpath(tagsxPath).extract()
            link = item.xpath(linkxPath).extract_first()
            
            #form entity with xpath
            entity = {
                    'name': name,
                    'tags':tags,
                    'link':link
                    }
            
            entityList.append(entity)
            
        #write into file
        writeFile = "output.txt"
        with open(writeFile,'w') as file:
            for i in range(len(entityList)):
                writeline = str(i+1)+"\t"+entityList[i]['name']+'\t'\
                +",".join(entityList[i]['tags'])+"\t"+entityList[i]['link']+"\n"
                
                file.write(writeline) 
        
        linksxPath = '//div[contains(@class,"tracking_attraction_title listing_title")]/a/@href'
        linkList = response.xpath(linksxPath).extract()
        forelink = "https://www.tripadvisor.com.my"
        finalLinks = []
        for e in linkList:
            finalLinks.append(forelink+e)
            
        for link in finalLinks:
            yield response.follow(url = link,callback = self.parse_details)   
            
    # third parsing method, to scrape detailed information
    def parse_details(self, response):
        #get the div contains the first portion of information
        headingDiv = response.xpath('//div[@class = "ui_columns is-multiline is-mobile contentWrapper"]')

        headingxPath = './/h1[@id = "HEADING"]/text()'
        addressxPath = './/span[@class = "detail"]//text()'
        rankxPath = './/span[@class = "header_popularity popIndexValidation "]//text()'
        noofReviewsxPath = './/span[@class = "reviewCount"]/text()'
        averageRatexPath = './/span[contains(@class,"ui_bubble_rating")]/@alt'
        
        heading = headingDiv.xpath(headingxPath).extract_first()
        address = headingDiv.xpath(addressxPath).extract()
        rank = headingDiv.xpath(rankxPath).extract()
        noofReviews = headingDiv.xpath(noofReviewsxPath).extract_first()
        if noofReviews is None:
            noofReviews = ''
        averageRate = headingDiv.xpath(averageRatexPath).extract_first()
        if averageRate is None:
            averageRate = ''
        
        ratingDiv = response.xpath('//div[contains(@class,"prw_rup prw_common_ratings_histogram_overview overviewHistogram")]')
        
        ratingDistributionxPath = './/li[contains(@class,"chart_row")]//span[contains(@class,"row_count row_cell")]/text()'
        ratingDistribution = ratingDiv.xpath(ratingDistributionxPath).extract()
        
        if not ratingDistribution:
            ratingDistribution = ['','','','','']
            
        entity = {
                'heading':heading,
                'address':address,
                'rank':rank,
                'noofReviews':noofReviews,
                'averageRate':averageRate,
                
                'ratingDistribution':ratingDistribution
                }
        
        entityListDetails.append(entity)
        
'''
process start from here
'''
entityList = []
entityListDetails = []

# Run the Spider
process = CrawlerProcess()
process.crawl(POISpider)
process.start()



#write detailed information into file
writeFile = "outputDetails1.txt"
with open(writeFile,'w') as file:
    for i in range(len(entityListDetails)):
        writeline = str(i+1)+"\t"+entityListDetails[i]['heading']+'\t'\
        +"".join(entityListDetails[i]['address'])+'\t'+"".join(entityListDetails[i]['rank'])+'\t'\
        +entityListDetails[i]['noofReviews']+'\t'+entityListDetails[i]['averageRate']+'\t'\
        +",".join(entityListDetails[i]['ratingDistribution'])+"\n"
        
        file.write(writeline) 






