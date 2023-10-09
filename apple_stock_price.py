import scrapy
import telegram
import pandas as pd

class AppleStockPriceSpider(scrapy.Spider):
    name = 'apple_stock_price'

    def start_requests(self):
        url = 'https://finance.yahoo.com/quote/AAPL'
        yield scrapy.Request(url=url)

    def parse(self, response):
        stock_price = response.css('.Trsdu(0)::text').get()
        MA50 = response.css('.MvL(2)::text').get()
        MA200 = response.css('.MvL(3)::text').get()
        yield {
            'stock_price': stock_price,
            'MA50': MA50,
            'MA200': MA200
        }

    def send_telegram_notification(self, stock_price, MA50, MA200):
        bot = telegram.Bot(token='YOUR_TELEGRAM_BOT_TOKEN')
        chat_id = 'YOUR_TELEGRAM_CHAT_ID'
        if MA50 > MA200 and stock_price > MA50:
            message = 'The 50-day moving average has crossed above the 200-day moving average. This is a bullish signal.'
        elif MA50 < MA200 and stock_price < MA50:
            message = 'The 50-day moving average has crossed below the 200-day moving average. This is a bearish signal.'
        bot.send_message(chat_id, message)

if __name__ == '__main__':
    crawler = scrapy.crawler.CrawlerRunner()
    crawler.crawl(AppleStockPriceSpider)
    crawler.start()
