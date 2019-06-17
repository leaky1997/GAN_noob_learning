# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:10:08 2019

@author: 李奇
"""


import time

from urllib import parse

import pandas as pd

from selenium import webdriver

from selenium.webdriver.chrome.options import Options
driver = webdriver.PhantomJS(r'.\大作业\phantomjs-2.1.1-windows\bin\phantomjs.exe')
# =============================================================================
# 
# chrome_options = Options()
# 
# chrome_options.add_argument('--headless')
# 
# chrome_options.add_argument('--disable-gpu')
# 
# driver = webdriver.Chrome(executable_path='./chromedriver', chrome_options=chrome_options)
# 
# =============================================================================
base_url = 'https://www.aqistudy.cn/historydata/daydata.php?city='

def get_month_set():

    month_set = list()

    for i in range(12, 13):

        month_set.append(('2013%s' % i))

    for i in range(1, 10):

        month_set.append(('20140%s' % i))

    for i in range(10, 13):

        month_set.append(('2014%s' % i))

    for i in range(1, 10):

        month_set.append(('20150%s' % i))

    for i in range(10, 13):

        month_set.append(('2015%s' % i))

    for i in range(1, 10):

        month_set.append(('20160%s' % i))

    for i in range(10, 13):

        month_set.append(('2016%s' % i))

    for i in range(1, 10):

        month_set.append(('20170%s' % i))

    for i in range(10, 13):

        month_set.append(('2017%s' % i))

    for i in range(1, 10):

        month_set.append(('20180%s' % i))

    for i in range(10, 13):

        month_set.append(('2018%s' % i))
        
    for i in range(1, 7):

        month_set.append(('20190%s' % i))




    return month_set

month_set = get_month_set()

city = '张家港'#苏州0.、太仓、昆山、无锡、常熟、张家港、上海

file_name = city + '补充.csv'

fp = open(file_name, 'w')

fp.write('%s,%s,%s,%s,%s,%s,%s,%s,%s\n'%('date','AQI','grade','PM25','PM10','SO2','CO','NO2','O3_8h'))#表头

for i in range(len(month_set)):

        str_month = month_set[i]

        weburl = ('%s%s&month=%s' % (base_url, parse.quote(city), str_month))

        driver.get(weburl)

        dfs = pd.read_html(driver.page_source,header=0)[0]

        time.sleep(5)#防止页面一带而过，爬不到内容

        for j in range(0,len(dfs)):

            date = dfs.iloc[j,0]

            aqi = dfs.iloc[j,1]

            grade = dfs.iloc[j,2]

            pm25 = dfs.iloc[j,3]

            pm10 = dfs.iloc[j,4]

            so2 = dfs.iloc[j,5]

            co = dfs.iloc[j,6]

            no2 = dfs.iloc[j,7]

            o3 = dfs.iloc[j,8]

            print(date)

            print(aqi)

            fp.write(('%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (date,aqi,grade,pm25,pm10,so2,co,no2,o3)))

            print('%d---%s,%s---DONE' % (city.index(city), city, str_month))

fp.close()

driver.quit()

print ('请查漏补缺！')
