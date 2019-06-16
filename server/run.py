from scraper import KickstarterScraper

url = input("Pls enter url\n")
x = KickstarterScraper(url)
x.scrape()
