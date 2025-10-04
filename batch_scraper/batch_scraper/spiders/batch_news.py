import scrapy
import os
import json
import hashlib
import requests
from urllib.parse import urljoin, urlparse
from pathlib import Path
import re
from datetime import datetime
from unidecode import unidecode


class BatchNewsSpider(scrapy.Spider):
    name = 'batch_news'
    allowed_domains = ['www.deeplearning.ai']
    start_urls = ['https://www.deeplearning.ai/the-batch/']
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS': 1,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'BatchNewsScraper 1.0',
        'ITEM_PIPELINES': {},
        'SCHEDULER_MEMORY_QUEUE': 'scrapy.squeues.FifoMemoryQueue'
    }
    
    def __init__(self, max_articles=None, *args, **kwargs):
        super(BatchNewsSpider, self).__init__(*args, **kwargs)
        self.max_articles = int(max_articles) if max_articles else None
        self.articles_scraped = 0
        self.setup_directories()
        self.load_scraped_urls()
    
    def setup_directories(self):
        """Create necessary directories"""
        project_root = Path(__file__).parent.parent.parent.parent
        self.data_dir = project_root / 'data'
        self.txt_dir = self.data_dir / 'txt'
        self.img_dir = self.data_dir / 'img'
        
        for directory in [self.data_dir, self.txt_dir, self.img_dir]:
            directory.mkdir(exist_ok=True)
        
        self.scraped_urls_file = self.data_dir / 'scraped_urls.json'
        self.article_counter_file = self.data_dir / 'article_counter.txt'
    
    def load_scraped_urls(self):
        """Load previously scraped URLs to avoid duplicates"""
        try:
            with open(self.scraped_urls_file, 'r') as f:
                self.scraped_urls = set(json.load(f))
        except FileNotFoundError:
            self.scraped_urls = set()
        
        try:
            with open(self.article_counter_file, 'r') as f:
                self.article_id = int(f.read().strip())
        except FileNotFoundError:
            self.article_id = 1
        
        self.metadata_file = self.data_dir / 'metadata.json'
        try:
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            self.metadata = {}
    
    def save_scraped_urls(self):
        """Save scraped URLs and metadata to files"""
        with open(self.scraped_urls_file, 'w') as f:
            json.dump(list(self.scraped_urls), f, indent=2)
        
        with open(self.article_counter_file, 'w') as f:
            f.write(str(self.article_id))
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def clean_text(self, text):
        """Clean text by replacing Unicode characters with ASCII equivalents"""
        if not text:
            return text
        
        return unidecode(text)
    
    def parse(self, response):
        """Parse the main page and extract article links"""
        article_links = response.css('article a::attr(href)').getall()
        
        batch_links = []
        for link in article_links:
            if '/the-batch/' in link and '/issue-' in link and link not in batch_links:
                batch_links.append(link)
        
        self.logger.info(f"Found {len(batch_links)} article links on page {response.url}")
        
        if not batch_links:
            self.logger.warning("No article links found with 'article a' selector, trying alternatives...")
            
            alt_selectors = [
                'a[href*="/the-batch/"]::attr(href)',
                '.post-title a::attr(href)',
                '.entry-title a::attr(href)',
                'h2 a::attr(href)',
                'h3 a::attr(href)'
            ]
            
            for selector in alt_selectors:
                alt_links = response.css(selector).getall()
                if alt_links:
                    batch_links.extend([link for link in alt_links if '/the-batch/' in link])
                    break
        
        batch_links = list(dict.fromkeys(batch_links))
        
        def extract_issue_number(url):
            try:
                import re
                match = re.search(r'/issue-(\d+)/', url)
                return int(match.group(1)) if match else 0
            except:
                return 0
        
        batch_links.sort(key=extract_issue_number, reverse=True)
        
        issue_numbers = [extract_issue_number(link) for link in batch_links]
        self.logger.info(f"Issue numbers found (sorted newest first): {issue_numbers[:10]}...")
        
        estimated_articles_from_page = len(batch_links) * 4
        
        if self.max_articles and self.articles_scraped >= self.max_articles:
            self.logger.info(f"Max articles limit ({self.max_articles}) already reached, stopping crawl")
            return
        
        processed_links = 0
        for link in batch_links:
            estimated_total = self.articles_scraped + (processed_links * 4)
            if self.max_articles and estimated_total >= self.max_articles:
                self.logger.info(f"Estimated total articles ({estimated_total}) would exceed limit ({self.max_articles}), stopping link processing")
                break
                
            full_url = urljoin(response.url, link)
            
            if full_url not in self.scraped_urls:
                processed_links += 1
                yield response.follow(link, self.parse_article, meta={'from_page': response.url})
        
        current_page = 1
        if '/page/' in response.url:
            try:
                current_page = int(response.url.split('/page/')[1].rstrip('/'))
            except:
                current_page = 1
        
        conservative_estimate = self.articles_scraped + (processed_links * 4)
        
        should_continue_pagination = (
            processed_links > 0 and
            (not self.max_articles or conservative_estimate < self.max_articles) and
            current_page < 50
        )
        
        self.logger.info(f"Page {current_page}: processed {processed_links} links, "
                        f"current articles: {self.articles_scraped}, "
                        f"estimated total: {conservative_estimate}, "
                        f"limit: {self.max_articles}, "
                        f"continue pagination: {should_continue_pagination}")
        
        if should_continue_pagination:
            next_page = current_page + 1
            next_page_url = f"https://www.deeplearning.ai/the-batch/page/{next_page}/"
            
            yield response.follow(
                next_page_url, 
                self.parse, 
                priority=-current_page * 10,
                dont_filter=True
            )
        else:
            self.logger.info(f"Stopping pagination at page {current_page}")

    def parse_article(self, response):
        """Parse individual article pages - extract News section content"""
        if self.max_articles and self.articles_scraped >= self.max_articles:
            self.logger.info(f"Skipping article processing - limit reached: {response.url}")
            return
        
        if response.url in self.scraped_urls:
            self.logger.info(f"Already scraped: {response.url}")
            return
        
        news_headers = response.xpath('//h1[@id="news"]/following-sibling::*//h1 | //h1[@id="news"]/following::h1')
        
        if not news_headers:
            news_section_h2 = response.xpath('//h2[@id="news"]')
            if news_section_h2:
                news_headers = news_section_h2[0].xpath('./following-sibling::*//h2 | ./following::h2')
                if news_headers:
                    self.logger.info(f"Found News section in h2, processing {len(news_headers)} h2 headers")
        
        if not news_headers:
            self.logger.warning(f"No News section found on page: {response.url}, using backup extraction")
            all_headers = response.css('h1').getall()
            if len(all_headers) > 1:
                news_headers = response.css('h1')[1:]
                self.logger.info(f"Backup extraction: found {len(news_headers)} h1 headers")
            else:
                h2_headers = response.css('h2').getall()
                if h2_headers:
                    news_headers = response.css('h2')
                    self.logger.info(f"Backup extraction: found {len(news_headers)} h2 headers")
                else:
                    self.logger.warning(f"No suitable headers (h1 or h2) found for backup extraction on: {response.url}")
                    return
        
        if not news_headers:
            return
        
        self.logger.info(f"Processing {len(news_headers)} articles from {response.url}")
        
        # Process each news article on the page
        for i, header in enumerate(news_headers):
            if self.max_articles and self.articles_scraped >= self.max_articles:
                break
            
            title = header.css('::text').get()
            if not title:
                continue
            
            title = title.strip()
            title = self.clean_text(title)
            
            if title.upper().startswith('A MESSAGE FROM'):
                self.logger.info(f"Skipping message section: {title}")
                continue
            
            all_following = header.xpath('./following-sibling::*')
            
            content_elements = []
            for elem in all_following:
                if elem.root.tag.lower() in ['h1', 'h2']:
                    break
                content_elements.append(elem)
            
            content_parts = []
            image_urls = []
            
            preceding_elements = header.xpath('./preceding-sibling::*[position()<=3]')
            for elem in reversed(preceding_elements):
                imgs = elem.css('img::attr(src)').getall()
                if imgs:
                    for img in imgs:
                        if img and not img.startswith('data:'):
                            full_img_url = urljoin(response.url, img)
                            if full_img_url not in image_urls:
                                image_urls.append(full_img_url)
                    break
            
            for element in content_elements:
                text_content = ' '.join(element.css('::text').getall()).strip()
                if text_content:
                    content_parts.append(text_content)
            
            content = '\n\n'.join(content_parts)
            content = self.clean_text(content)
            
            if len(content.strip()) < 50:
                self.logger.warning(f"Insufficient content for article: {title}")
                continue
            
            safe_title = re.sub(r'[^\w\s-]', '', title)
            safe_title = re.sub(r'[-\s]+', '_', safe_title)[:50]
            
            filename_base = f"{self.article_id}_{safe_title}"
            
            txt_file = self.txt_dir / f"{filename_base}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            downloaded_images = []
            for i_img, img_url in enumerate(image_urls, 1):
                try:
                    img_response = requests.get(img_url, timeout=10)
                    if img_response.status_code == 200:
                        parsed_url = urlparse(img_url)
                        ext = os.path.splitext(parsed_url.path)[1] or '.jpg'
                        
                        if len(image_urls) == 1:
                            img_filename = f"{filename_base}{ext}"
                        else:
                            img_filename = f"{filename_base}_{i_img}{ext}"
                        
                        img_path = self.img_dir / img_filename
                        
                        with open(img_path, 'wb') as f:
                            f.write(img_response.content)
                        
                        downloaded_images.append(img_filename)
                    
                except Exception as e:
                    self.logger.error(f"Error downloading image {img_url}: {str(e)}")
            
            self.metadata[str(self.article_id)] = {
                "title": title,
                "url": response.url,
                "date": datetime.now().isoformat(),
                "txt_file": f"{filename_base}.txt",
                "img_files": downloaded_images,
                "content_length": len(content)
            }
            
            self.scraped_urls.add(response.url)
            self.articles_scraped += 1
            self.article_id += 1
            
            if self.articles_scraped % 5 == 0:
                self.save_scraped_urls()
            
            self.logger.info(f"Scraped article {self.articles_scraped}: {title} ({len(content)} chars, {len(downloaded_images)} images)")
            
            yield {
                'id': self.article_id - 1,
                'title': title,
                'url': response.url,
                'content_length': len(content),
                'images_downloaded': len(downloaded_images),
                'txt_file': str(txt_file),
                'img_files': downloaded_images
            }
    
    def closed(self, reason):
        """Called when spider closes"""
        self.save_scraped_urls()
        self.logger.info(f"Spider closed. Scraped {self.articles_scraped} articles.")
