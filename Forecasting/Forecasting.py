#Import packages
from bs4 import BeautifulSoup
import requests
import yfinance
from stop_words import get_stop_words
import re
import datetime
from fuzzywuzzy import process
import datetime as dt
import pandas
import numpy
import math
import pickle
import os
import json

###Functions
#Clean html tags in string
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
    
#Get stock symbol
def getCompany(text):
    r = requests.get('https://api.iextrading.com/1.0/ref-data/symbols')
    stockList = r.json()
    return process.extractOne(text, stockList)[0]

#Rounds time downwards to earlier five minute interval
def roundtime(time):
    while 1<2:
        if float(time.minute/5).is_integer() is True:
            break
        time += dt.timedelta(minutes = -1)
    return time

#Loads relevant data from web scraping process
try:
    wd = os.path.join(os.getcwd(), 'SDA-Oliver-Kostorz-SMART-Sentiment-Analysis')
except:
    print('Please specify path to working directiory manually.')
    
data = pandas.read_csv(os.path.join(wd, 'data.csv'))

#Forecasting
#List containing relevant URLs for testing forecasts
#News must be at least 5 days old!!!

with open(os.path.join(wd, 'test_links.txt'), "rb") as fp:   # Unpickling
    test_urls = pickle.load(fp)

#Add additional links to news articels to use for training the algorithm
#Refer to Meta-Information for full guide and requierments
try:
    test_urls.append()
except:
    pass

with open(os.path.join(wd, 'test_links.txt'), "wb") as fp:   #Pickling
    pickle.dump(test_urls, fp)

#Defines dataframe to collect relevant info
voi = {'stock':[], 'date':[], 'return prediction':[], 'return realization':[],
        'volatility prediction':[], 'volatility realization':[]}
evaluation = pandas.DataFrame(voi)


#Same steps as during web scraping to process news
stop_words = get_stop_words('german')
stop_words.append('dpaafx')
fault_counter = 0
for link in test_urls:
    website = requests.get(link) 
    news = website.content 
    soup = BeautifulSoup(news, 'lxml')
    container = soup.find('div', class_='teaser teaser-xs color-news')
    parts_html = list()
    for para in container.find_all('p', recursive=False):
        parts_html.append(para)
    parts = remove_html_tags(str(parts_html))
    expression = "[^a-zA-Z äüöß]" 
    text_raw = re.sub(expression, '', str(parts))
    text_raw_lower = text_raw.lower()
    word_tokens = text_raw_lower.split()
    filtered_sentence = list()
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    name_section = soup.find('div', class_='chart-block relative')
    name_parts_html = list()
    for para in name_section.find_all('a', recursive=False):
        name_parts_html.append(para)
    name_parts = remove_html_tags(str(name_parts_html))
    name = name_parts[1:len(name_parts)-18]
    date_section = soup.find(class_="pull-left mright-20")
    date = str(date_section)[33:49]
    date_time = datetime.datetime.strptime(date, '%d.%m.%Y %H:%M')
    try:
        date = date_time.strftime('%Y-%m-%d')
        rounded_time = roundtime(date_time)
        t_end = (date_time + datetime.timedelta(days=3)).strftime('%Y-%m-%d')
        t_start = (date_time + datetime.timedelta(days=-5)).strftime('%Y-%m-%d')
        symbol = getCompany(name).get('symbol')
        stock_data = yfinance.Ticker(symbol)
        return_t = stock_data.history(start = t_start, end = t_end, interval = "5m")
        return_24 = 100*((((return_t.loc[(rounded_time).strftime('%Y-%m-%d %H:%M:%S') : t_end])["Open"].iloc[77])-
                          ((return_t.loc[t_start : (rounded_time).strftime('%Y-%m-%d %H:%M:%S')])["Open"].iloc[-1]))/((return_t.loc[t_start : (rounded_time).strftime('%Y-%m-%d %H:%M:%S')])["Open"].iloc[-1]))
        var_till_t = (numpy.var((return_t.loc[t_start : (rounded_time).strftime('%Y-%m-%d %H:%M:%S')])["Open"])) # variance until news
        var_after_t = (numpy.var((return_t.loc[(rounded_time).strftime('%Y-%m-%d %H:%M:%S') : t_end])["Open"])) # variance after news
        volatility_24 = math.sqrt(var_after_t)/math.sqrt(var_till_t)
        #Predict return and volatility by words in news article
        return_prediction = 0
        volatility_prediction = 0
        counter = 1
        for word in filtered_sentence:
            if word in data.values:
                return_prediction = ((return_prediction * counter) + data.get_value(data.loc[data['word'] == word].index[0], 2, takeable = True))/ (counter + 1)
                volatility_prediction = ((volatility_prediction * counter) + data.get_value(data.loc[data['word'] == 'ab'].index[0], 3, takeable = True))/(counter + 1)
                counter = counter + 1
            else:
                pass
        #Save result in table
        evaluation = evaluation.append({'stock' : symbol, 'date' : rounded_time,
                                        'return prediction' : return_prediction, 'return realization' : return_24,
                                        'volatility prediction' : volatility_prediction, 'volatility realization' : volatility_24},
                                        ignore_index = True)
    except:
        fault_counter = fault_counter + 1

success_counter = len(test_urls)-fault_counter

print('Hello ' + os.getlogin() + ',')
print('as a test-sample of the predictive power of the liabary, ' + str(success_counter) + ' news articels could be assessed.')
print('Unfortunately, ' + str(fault_counter) + ' articels did not fullfil the requierments and were not included in the calculation.')
print('Among the most common reasons for exclusion are:')
print('-Publication was too recent to gather enough return data')
print('-Articel was published on weekends and might be outdated by the time the stock exchange opens again')
print('-Articel was not published on Finanzen.net directly but rather refers to another website')
print('-Publication date is outside requestable return range (60 days)')
print('Please refer to the Meta-Information for further explanations.')
print('However, we will continue with the valid data.')


evaluation.to_csv(os.path.join(wd, 'evaluation.csv'), index = False, header = True)





