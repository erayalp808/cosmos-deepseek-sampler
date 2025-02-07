import requests
import re
from dotenv import load_dotenv
import os

load_dotenv()

def get_deepseek_response(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer { os.getenv('API_KEY') }"
    }
    data = {
        "model": "deepseek-r1-distill-llama-70b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            },
        ]
    }

    return requests.post(url, headers=headers, json=data)

def parse_reasoned_output(output):
        think_tag_pattern = r'<think>(.*?)</think>'
        think_elements = re.findall(think_tag_pattern, output, re.DOTALL)

        print('THINK_START', think_elements[0], 'THINK_END')

        print('OUTPUT_START')
        last_think_end = output.rfind('</think>')

        if last_think_end != -1:
            content_after_last_think = output[last_think_end + len('</think>'):].strip()
            print(content_after_last_think)
        else:
            print('No closing </think> tag found.')
        print('OUTPUT_END')