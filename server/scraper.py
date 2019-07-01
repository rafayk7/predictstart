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
        self.html = requests.get(self.url)
        self.content = None
        self.name = None
        self.pledged = None
        self.currency = None
        self.location = None
        self.category = None
        self.mainCategory = None
        self.goal = None
        self.backers = None
        self.launched = None
        self.deadline = None
        self.usdpledged = None
        self.country = None

    def getHTML(self):
        if self.content is None:
            content = BeautifulSoup(self.html.content, features="html.parser")
            self.content = content

        return self.content

    def getName(self):
        if self.name is None:
            try:
                name = self.content.findAll('h2', {
                                "class": "type-24 type-28-sm type-38-md navy-700 medium mb3 project-name"})[0].get_text()
            except IndexError:
                name = 'aaaaaaaaaa'

            self.name = name
        return self.name

    def getPledged(self):
        if self.pledged is None:
            try:
                pledgedText = self.content.findAll(
                'span', {"class": "ksr-green-700"})[0].get_text()
            except IndexError:
                pledgedText = "$15,000"

            pledged, currency = self.parseMoney(pledgedText)
            self.getCurrency(currency)

            self.pledged = pledged
        return self.pledged

    def getCurrency(self, c):
        if self.currency is None:
            self.currency = c
        return self.currency

    def getLocation(self):
        if self.location is None:
            locationCategoryContainer = self.content.findAll('span', {"class": "ml1"})
                
            location, category = self.getLocationAndCategory(locationCategoryContainer)
            self.location = location
            self.getCategory(category)
            
        return self.location

    def getCategory(self, c=None):
        if self.category is None and c is not None:
            self.category = c

        return self.category

    def getMainCategory(self):
        if self.mainCategory is None:
            mainCat = ''
            with open('categories.json', 'r') as f:
                arrays = json.load(f)

                for key in arrays.keys():
                    array = arrays[key]
                    for cat in array:
                        if cat == self.category:
                            mainCat = key

            self.mainCategory = mainCat
        return self.mainCategory

    def getGoal(self):
        if self.goal is None:
            try:  
                goal, currency = self.parseMoney(self.content.findAll('span', {
                                        "class": "inline-block hide-sm"})[0].findAll('span', {"class": "money"})[0].get_text())
            except:
                goal = 0
                currency = ''

            self.goal = goal
        return self.goal

    def getBackers(self):
        if self.backers is None:
            try:
                backers = int(self.content.findAll('div', {
                        "class": "block type-16 type-24-md medium soft-black"})[0].get_text().replace(',', ''))
            except:
                backers = 0
            self.backers = backers

        return backers

    def getDeadline(self):
        if self.deadline is None:
            try:
                deadlineContainer = self.content.findAll(
                'span', {"class": "block type-16 type-24-md medium soft-black"})[0].get_text()
                print("HOURS OR DAYS " + self.content.findAll('span', {'class': 'block navy-600 type-12 type-14-md lh3-lg'})[0].get_text())
                deadline = (datetime.today() +
                        timedelta(days=int(deadlineContainer))).strftime(r"%Y-%m-%d")
            except:
                deadline = datetime.today()

            self.deadline = deadline
        return self.deadline

    def getLaunched(self):
        if self.launched is None:
            launchDate = ''
            url = ''
            if "?ref=" in self.url:
                refIndex = self.url.find("ref")
                url = self.url[0:refIndex-1]+"/updates"
                page = requests.get(url)

                content = BeautifulSoup(page.content, features="html.parser")
                className = 'timeline__divider timeline__divider--launched timeline__divider--launched--' + self.mainCategory.lower()
                try:
                    launchDate = content.findAll('div', {'class': className})[0].findAll('time')[0].get_text()
                    launchDate = datetime.strptime(launchDate, r"%B %d, %Y").strftime(r"%Y-%m-%d")
                except:
                    datetime.now()

            launched = str(launchDate)

            self.launched = launched
        return self.launched

    def getUSDPledged(self):
        if self.usdpledged is None:
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
        if self.country is None:
            try:
                location = self.location.split(',')[1].strip(' ')
            except:
                location = ''
            countries = dict(countries_for_language('en'))

            if location=='UK':
                self.country = 'GB'
                return self.country
            elif location=='NZ':
                self.country = 'NZ'
                return self.country
            elif len(location)==2:
                self.country = 'US'
                return self.country
        
            for cCode, country in countries.items():
                if country == location:
                    self.country = cCode
        
        return self.country

    def getLocationAndCategory(self, locationCategoryContainer):
        location = ''
        country = ''

        try:
            locationCategoryContainer[4]
            location = locationCategoryContainer[5].get_text()
            category = locationCategoryContainer[4].get_text()  # 4
        except:
            try:
                location = locationCategoryContainer[3].get_text()
                category = locationCategoryContainer[2].get_text()
            except:
                location = ''
                category = ''

        return location, category

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
    
    def getJsonResult(self):
        json = {
        "title": self.name,
        "pledged": self.pledged,
        "currency": self.currency,
        "location": self.location,
        "category": self.category,
        "main_category": self.mainCategory,
        "goal": self.goal,
        "backers": self.backers,
        "launched_date": self.launched,
        "deadline_date": self.deadline,
        "usd_pledged": self.usdpledged,
        "country": self.country
        }

        return json
    
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
        print("COUNTRY: " + str(self.country))
    
    def scrape(self, display=False, json=True):
        #Must be called in this order
        self.getHTML()
        self.getName()
        self.getPledged()
        self.getLocation()
        self.getMainCategory()
        self.getGoal() 
        self.getBackers()
        self.getDeadline()
        self.getLaunched()
        self.getUSDPledged()
        self.getCountry()

        if display and json:
            self.displayInfo()
            return self.getJsonResult()
        if display and not json:
            self.displayInfo()
        if json and not display:
            return self.getJsonResult()



    


    
