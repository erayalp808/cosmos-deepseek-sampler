import pandas as pd
import numpy as np
from helpers import *

verbose = True
instructions = pd.read_csv('instructions.csv')
output_file_path = 'turkish_instructions.csv'
output_df = get_output_df(output_file_path)

for row in instructions[:10].itertuples():
    index, _, prompt, input_values, output = row
    input_values = '' if pd.isna(input_values) else input_values
    prompt_final = prompt + input_values

    response = get_deepseek_response(prompt_final)

    if response.status_code == 200:
        response = response.json()
        generated_output = response['choices'][0]['message']['content']
        thought_process, real_output = parse_reasoned_output(generated_output)
        sample = [
            index,
            prompt,
            input_values,
            thought_process,
            real_output
        ]
        output_df.loc[len(output_df)] = sample
        if index % 25 == 0: output_df.to_csv(output_file_path)

        if verbose:
            print(index, prompt, input_values)
            print('THINK_START', thought_process, 'THINK_END')
            print('OUTPUT_START', real_output, 'OUTPUT_END')
    else:
        print(f"Error: { response.status_code }, { response.text }")
        print('PROMPT INDEX: ' + index)
        output_df.to_csv(output_file_path)
        exit()


output_df.to_csv(output_file_path)