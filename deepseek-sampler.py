import pandas as pd
import numpy as np
from helpers import *

instructions = pd.read_csv('instructions.csv')

for row in instructions[:3].itertuples():
    index, _, prompt, input_values, output = row
    input_values = '' if pd.isna(input_values) else input_values
    prompt_final = prompt + input_values

    response = get_deepseek_response(prompt_final)

    if response.status_code == 200:
        response = response.json()
        generated_output = response['choices'][0]['message']['content']
        parse_reasoned_output(generated_output)
    else:
        print(f"Error: {response.status_code}, {response.text}")

