import requests
import re
from dotenv import load_dotenv
import os

load_dotenv()

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
            "content": "what are the prompts that make you think the most"
        }
    ]
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    response = response.json()
    output_content = response['choices'][0]['message']['content']
    think_tag_pattern = r'<think>(.*?)</think>'
    think_elements = re.findall(think_tag_pattern, output_content, re.DOTALL)

    print('THINK_START')
    for think_element in think_elements:
        print(think_element)
    print('THINK_END')

    print('OUTPUT_START')
    last_think_end = output_content.rfind('</think>')

    if last_think_end != -1:
        content_after_last_think = output_content[last_think_end + len('</think>'):].strip()
        print(content_after_last_think)
    else:
        print('No closing </think> tag found.')
    print('OUTPUT_END')
else:
    print(f"Error: {response.status_code}, {response.text}")