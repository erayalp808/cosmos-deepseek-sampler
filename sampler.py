import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
from scrapy.http import Request
from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.utils.response import response_status_message
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
import time
import json
import re
import random
import time


class ExponentialBackoffRetryMiddleware(RetryMiddleware):
    def _retry(self, request, reason, spider):
        retries = request.meta.get('retry_times', 0) + 1
        retry_times = spider.settings.getint('RETRY_TIMES')

        if retries <= retry_times:
            wait_time = random.uniform(2 ** retries, 2 ** (retries + 1))  # Exponential backoff
            spider.logger.warning(
                f"Retrying {request.url} (attempt {retries}/{retry_times}) after {wait_time:.2f} seconds due to {reason}"
            )

            time.sleep(wait_time)

            new_request = request.copy()
            new_request.dont_filter = True
            new_request.meta['retry_times'] = retries
            return new_request

        spider.logger.error(f"Gave up retrying {request.url} (failed {retries} times): {reason}")
        return None

class DeepSeekSample(scrapy.Item):
    bolum = scrapy.Field()
    konu = scrapy.Field()
    soru = scrapy.Field()
    cevap = scrapy.Field()
    secenekler = scrapy.Field()
    harf_secenekler = scrapy.Field()
    dusunce = scrapy.Field()
    cikti = scrapy.Field()
    deepseek_cevap = scrapy.Field()
    dogru_cevap = scrapy.Field()
    isabet = scrapy.Field()

class DeepSeekSamplerSpider(scrapy.Spider):
    name = "deepseek_sampler"
    custom_settings = {
        "LOG_LEVEL": "INFO",

        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0,
        "AUTOTHROTTLE_MAX_DELAY": 60,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
        "AUTOTHROTTLE_DEBUG": True,

        "RETRY_ENABLED": True,
        "RETRY_TIMES": 8,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504, 522, 524, 408],

        "DOWNLOADER_MIDDLEWARES": {
            "scrapy.downloadermiddlewares.retry.RetryMiddleware": None,
            "__main__.ExponentialBackoffRetryMiddleware": 550,
        },

        "FEEDS": {
            "data.csv": {
                "format": "csv",
                "encoding": "utf-8",
                "store_empty": False,
                "fields": ["bolum", "konu", "soru", "cevap", "secenekler", "harf_secenekler", "dusunce", "cikti", "deepseek_cevap", "dogru_cevap", "isabet"],
                "overwrite": True,
            },
        },
    }

    dataset = load_dataset("alibayram/turkish_mmlu")
    instructions = pd.DataFrame(dataset['train'])
    batch = instructions[:250]

    api_url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = ""

    model = "deepseek-r1-distill-llama-70b"

    request_count = 0
    completed_requests = 0
    progress_bar = None

    def start_requests(self):
        self.request_count = len(self.batch)
        self.progress_bar = tqdm(
            total=self.request_count,
            desc="Sampling Progress",
            unit="req"
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for _, row in self.batch.iterrows():
            prompt = self.build_prompt(row['soru'], row['secenekler'])

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4096,
                "top_p": 0.95,
                "stream": False,
                "reasoning_format": "raw",
            }

            yield Request(
                url=self.api_url,
                method="POST",
                body=json.dumps(payload),
                headers=headers,
                meta={"data": row},
                callback=self.parse
            )

    def format_options(self, options):
        option_labels = ['A', 'B', 'C', 'D', 'E'][:len(options)]
        formatted_options = "\n".join(f"{label}) {opt}" for label, opt in zip(option_labels, options))
        return formatted_options
    
    def build_prompt(self, question, options):
        formatted_options = self.format_options(options)
        return f"Soru: {question}\nSeçenekler:\n{formatted_options}\nDoğru cevabın harfini (A, B, C, D, E) ver."

    def parse_reasoned_output(self, output):
        think_tag_pattern = r'<think>(.*?)</think>'
        think_elements = re.findall(think_tag_pattern, output, re.DOTALL)
        real_output = ''
        end_of_thinking_str_index = output.rfind('</think>')
        thought_process = output[:end_of_thinking_str_index] if not think_elements else think_elements[0]

        if end_of_thinking_str_index != -1:
            real_output = output[end_of_thinking_str_index + len('</think>'):].strip()

        return thought_process, real_output
    
    def parse_correct_answer(self, output):
        # Search for patterns like "Doğru cevap: **A**" or "Cevap: **B**"
        match = re.search(r"(?:Doğru cevap|Cevap):\s*\*\*([A-Ea-e])\*\*", output)
        if match:
            return match.group(1).upper()
        
        # Search only the first and last 200 characters of the output
        search_areas = [output[:200], output[-200:]]
        
        for area in search_areas:
            # Search for patterns like "**A) " or "**c) "
            match = re.findall(r"\*\*([A-Ea-e])\)\s", area)
            if match:
                return match[-1].upper()

            # Handle single-letter outputs like "b"
            match = re.search(r"\b([A-Ea-e])\b", area)
            if match:
                return match.group(1).upper()

        return "Bilinmiyor"
    
    def letter_to_index(self, letter):
        return ord(letter.upper()) - ord('A')
    
    def index_to_letter(self, index):
        return chr(index + ord('A'))
    
    def get_accuracy(self, deepseek_answer, correct_answer):
        if deepseek_answer != "Bilinmiyor":
            deepseek_answer = self.letter_to_index(deepseek_answer)
            return "Doğru" if deepseek_answer == correct_answer else "Yanlış"
        else:
            return "Bilinmiyor"

    def parse(self, response):
        row = response.meta.get("data", {})
        response = response.json()
        output = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        reasoning, output_message = self.parse_reasoned_output(output)
        deepseek_answer = self.parse_correct_answer(output_message)
        accuracy = self.get_accuracy(deepseek_answer, row["cevap"])

        self.completed_requests += 1
        self.progress_bar.update(1)

        yield DeepSeekSample(
            bolum=row["bolum"],
            konu=row["konu"],
            soru=row["soru"],
            cevap=row["cevap"],
            secenekler=row["secenekler"],
            harf_secenekler=self.format_options(row["secenekler"]),
            dusunce=reasoning.strip(),
            cikti=output_message.strip(),
            deepseek_cevap=deepseek_answer,
            dogru_cevap=self.index_to_letter(row["cevap"]),
            isabet=accuracy
        )

    def closed(self, reason):
        self.progress_bar.close()
        self.logger.info(f"Spider closed: {reason}")


if __name__ == "__main__":
    configure_logging()
    process = CrawlerProcess()
    process.crawl(DeepSeekSamplerSpider)
    process.start()