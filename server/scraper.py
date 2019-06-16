import requests
from bs4 import BeautifulSoup
import json
from pygeocoder import Geocoder

url = ''
with open('config.json', 'r') as f:
    vars = json.load(f)
    url = vars['url7']


def getHTML(url):
    page = requests.get(url)
    content = BeautifulSoup(page.content, features="html.parser")

    printVar = url.find("ref")

    name = content.findAll('h2', {
                           "class": "type-24 type-28-sm type-38-md navy-700 medium mb3 project-name"})[0].get_text()

    pledgedText = content.findAll(
        'span', {"class": "ksr-green-700"})[0].get_text()
    pledged, currency = getCurrencyAndPledged(pledgedText)

    locationCategoryContainer = content.findAll('span', {"class": "ml1"})
    location, category = getLocationAndCategory(locationCategoryContainer)
    mainCategory = getMainCategory(category)

    country = getCountry(location)
    # category = content.findAll('span', {"class": "ml1"})[4].get_text()

    # x = content.findAll('span', {"class": "ml1"})[4].get_text()

    print("NAME: " + name)
    print("PLEDGED: " + str(pledged))
    print("CURRENCY: " + currency)
    print("LOCATION: " + location)
    print("CATEGORY: " + category)
    print("MAIN CATEGORY: " + mainCategory)

# with open('content.html', 'w') as file:
#     for item in list(content.children):
#         file.write(str(item))
#         file.write("\n")


def getLocationAndCategory(locationCategoryContainer):
    location = ''
    country = ''

    try:
        locationCategoryContainer[4]
        location = locationCategoryContainer[5].get_text()
        category = locationCategoryContainer[4].get_text()  # 4
    except:
        location = locationCategoryContainer[3].get_text()
        category = locationCategoryContainer[2].get_text()
        print("TRY")

    return location, category


def getMainCategory(category):
    with open('categories.json', 'r') as f:
        arrays = json.load(f)

        for key in arrays.keys():
            array = arrays[key]
            for cat in array:
                if cat == category:
                    return key


def getCountry(location):
    return


def getCurrencyAndPledged(pledged):
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


getHTML(url)
