import pandas as pd
import numpy as np
import time
from helpers import *

verbose = True
instructions = pd.read_csv('instructions.csv')
output_file_path = 'turkish_instructions.csv'
output_df = get_output_df(output_file_path)

for row in instructions[:1000].itertuples():
    index, _, prompt, input_values, output = row
    input_values = '' if pd.isna(input_values) else input_values
    prompt_final = prompt + input_values

    retry_attempts = 3
    attempt = 0
    while attempt < retry_attempts:
        try:
            attempt += 1
            response = get_deepseek_response(prompt_final)

            if response.status_code == 200:
                generated_output = response.json()['choices'][0]['message']['content']
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

                break
            else:
                raise Exception(f"Error: { response.status_code } - { response.text }")
        except Exception as exception:
            delay_amount = attempt * 30 # attempt * 30 seconds
            print(f"Attempt { attempt } failed: { exception }")

            if attempt < retry_attempts:
                time.sleep(delay_amount)
            else:
                output_df.to_csv(output_file_path)
                print(f"Final error on index { index }: { exception }")
                exit()

output_df.to_csv(output_file_path)