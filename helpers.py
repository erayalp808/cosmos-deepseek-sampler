import requests
import re
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np

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
    real_output = ''
    end_of_thinking_str_index = output.rfind('</think>')

    if end_of_thinking_str_index != -1:
        real_output = output[end_of_thinking_str_index + len('</think>'):].strip()

    return think_elements[0], real_output

def get_output_df(file_path):
    try:
        data = pd.read_csv(file_path)

        return data
    except FileNotFoundError:
        print(f"File { file_path } not found. Generating a new CSV file.")
        columns = ['talimat_no', 'talimat', 'giriş', 'düşünce', 'çıktı']
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path)

        return df