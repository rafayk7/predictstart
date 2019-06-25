from scraper import KickstarterScraper

url = input("Pls enter url\n")
htmlpath = 'server/content.html'
x = KickstarterScraper(html=htmlpath)
x.scrape()
