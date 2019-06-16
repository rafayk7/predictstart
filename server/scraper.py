import requests
from bs4 import BeautifulSoup
import json
import threading
from datetime import datetime
from datetime import timedelta
from country_list import countries_for_language

class KickstarterScraper:
    def __init__(self, url):
        self.url = url
        self.html = requests.get(url)
        self.content = ''
        self.name = ''
        self.pledged = 0
        self.currency = ''
        self.location = ''
        self.category = ''
        self.mainCategory = ''
        self.goal = 0
        self.backers = ''
        self.launched = ''
        self.deadline = ''
        self.usdpledged = 0
        self.country = ''

    # def run(self):
        # getHTML()

    def getHTML(self):
        content = BeautifulSoup(self.html.content, features="html.parser")
        self.content = content
        return content

    def getName(self):
        name = self.content.findAll('h2', {
                            "class": "type-24 type-28-sm type-38-md navy-700 medium mb3 project-name"})[0].get_text()

        self.name = name
        return self.name

    def getPledged(self):
        pledgedText = self.content.findAll(
            'span', {"class": "ksr-green-700"})[0].get_text()
        pledged, currency = self.parseMoney(pledgedText)
        self.getCurrency(currency)
        self.pledged = pledged
        return pledged

    def getCurrency(self, c):
        self.currency = c
        return c

    def getLocation(self):
        locationCategoryContainer = self.content.findAll('span', {"class": "ml1"})
        location, category = self.getLocationAndCategory(locationCategoryContainer)
        self.location = location
        self.getCategory(category)
        return location

    def getCategory(self, c):
        self.category = c
        return c
    
    def seeCat(self):
        return self.category

    def getMainCategory(self):
        mainCat = ''
        with open('categories.json', 'r') as f:
            arrays = json.load(f)

            for key in arrays.keys():
                array = arrays[key]
                for cat in array:
                    if cat == self.category:
                        mainCat = key

        self.mainCategory = mainCat
        return mainCat

    def getGoal(self):
        goal, currency = self.parseMoney(self.content.findAll('span', {
                                    "class": "inline-block hide-sm"})[0].findAll('span', {"class": "money"})[0].get_text())

        self.goal = goal
        return goal

    def getBackers(self):
        backers = int(self.content.findAll('div', {
                      "class": "block type-16 type-24-md medium soft-black"})[0].get_text().replace(',', ''))
        self.backers = backers
        return backers

    def getDeadline(self):
        deadlineContainer = self.content.findAll(
            'span', {"class": "block type-16 type-24-md medium soft-black"})[0].get_text()
        deadline = (datetime.today() +
                    timedelta(days=int(deadlineContainer))).strftime(r"%Y-%m-%d")

        self.deadline = deadline
        return self.deadline

    def getLaunched(self):
        launchDate = ''
        url = ''
        if "?ref=" in self.url:
            refIndex = self.url.find("ref")
            url = self.url[0:refIndex-1]+"/updates"
            page = requests.get(url)

            content = BeautifulSoup(page.content, features="html.parser")
            className = 'timeline__divider timeline__divider--launched timeline__divider--launched--' + self.mainCategory.lower()
            launchDate = content.findAll('div', {'class': className})[0].findAll('time')[0].get_text()
            launchDate = datetime.strptime(launchDate, r"%B %d, %Y").strftime(r"%Y-%m-%d")

        launched = str(launchDate)
        self.launched = launched
        return self.launched

    def getUSDPledged(self):
        conversionRateDict = {
            "USD": 1.0,
            "CAD": 0.75,
            "MXN": 0.052,
            "SGD": 0.73,
            "EUR": 1.12,
            "AUD": 0.69,
            "CHF": 1.00,
            "DKK": 0.15,
            "GBP": 1.26,
            "HKD": 0.13,
            "JPY": 0.0092,
            "NOK": 0.11,
            "NZD": 0.65,
            "SEK": 0.11
        }

        USDPledged = self.pledged*(conversionRateDict[self.currency])
        self.usdpledged = USDPledged
        return USDPledged

    def getCountry(self):
        location = self.location.split(',')[1].strip(' ')
        countries = dict(countries_for_language('en'))

        if location=='UK':
            self.country = 'GB'
        elif len(location)==2:
            self.country = 'US'
        
        for cCode, country in countries.items():
            if country == location:
                self.country = cCode
        
        return self.country

    def parseMoney(self, pledged):
        currencyDict = {
            "$": "USD",
            "CA$": "CAD",
            "MX$": "MXN",
            "S$": "SGD",
            "€": "EUR",
            "AU$": "AUD",
            "CHF": "CHF",
            "DKK": "DKK",
            "£": "GBP",
            "HK$": "HKD",
            "¥": "JPY",
            "NOK": "NOK",
            "NZ$": "NZD",
            "SEK": "SEK"
        }

        pledged = pledged.replace(',', '').replace(' ', '')

        costStartIndex = 0
        currency = ''
        for i in range(len(pledged)):
            try:
                int(pledged[i])
                costStartIndex = i
                break
            except:
                currency += pledged[i]

        pledged = int(pledged[costStartIndex:len(pledged)])
        currency = currencyDict[currency]

        return pledged, currency
    
    def displayInfo(self):
        print("NAME: " + self.name)
        print("PLEDGED: " + str(self.pledged))
        print("CURRENCY: " + self.currency)
        print("LOCATION: " + self.location)
        print("CATEGORY: " + self.category)
        print("MAIN CATEGORY: " + self.mainCategory)
        print("GOAL: " + str(self.goal))
        print("BACKERS: " + str(self.backers))
        print("LAUNCHED: " + self.launched)
        print("DEADLINE: " + str(self.deadline))
        print("USDPLEDGED: " + str(self.usdpledged))
        print("Country: " + str(self.country))
    
    def getLocationAndCategory(self, locationCategoryContainer):
        location = ''
        country = ''

        try:
            locationCategoryContainer[4]
            location = locationCategoryContainer[5].get_text()
            category = locationCategoryContainer[4].get_text()  # 4
        except:
            location = locationCategoryContainer[3].get_text()
            category = locationCategoryContainer[2].get_text()

        return location, category
    
    def scrape(self):
        self.getHTML()
        self.getName()
        self.getPledged()
        self.getLocation()
        self.getMainCategory()
        self.getGoal() #Goals, Backers, deadline, launched, country
        self.getBackers()
        self.getDeadline()
        self.getLaunched()
        self.getUSDPledged()
        self.getCountry()

        self.displayInfo()




    


    
