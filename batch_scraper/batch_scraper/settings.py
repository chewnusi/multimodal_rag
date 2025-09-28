BOT_NAME = 'batch_scraper'

SPIDER_MODULES = ['batch_scraper']
NEWSPIDER_MODULE = 'batch_scraper'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure delays
DOWNLOAD_DELAY = 1
RANDOMIZE_DOWNLOAD_DELAY = 0.5

# Configure concurrent requests
CONCURRENT_REQUESTS = 1
CONCURRENT_REQUESTS_PER_DOMAIN = 1

# Configure User-Agent
USER_AGENT = 'batch_scraper (+http://www.yourdomain.com)'

# Configure pipelines
ITEM_PIPELINES = {
    'scrapy.pipelines.files.FilesPipeline': 1,
}

# Configure extensions
EXTENSIONS = {
    'scrapy.extensions.telnet.TelnetConsole': None,
}

# Configure logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'batch_scraper.log'