# -*- coding;utf-8 -*-
"""
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
"""
import pandas as pd
from Local_LLM import LocalExtractor
from Evaluation import evaluate_predictions
from Remote_LLM import optimize_prompt

def main(input_file, model_path, tokenizer_path, N=6):
    df = pd.read_csv(input_file)
    local_extractor = LocalExtractor(model_path, tokenizer_path)

    for i in range(N):
        print(f"Iteration {i + 1}")
        analyzed_df = local_extractor.analyze_patient_records(df.copy())
        evaluation_summary = evaluate_predictions(analyzed_df, correct_column='correct_answer', predicted_column='冲动')
        optimized_prompt = optimize_prompt(local_extractor.prompt1, evaluation_summary)
        print(f"Optimized Prompt (Iteration {i + 1}):")
        print(optimized_prompt)
        local_extractor.prompt1 = optimized_prompt

if __name__ == "__main__":
    input_file = 'validation_records.csv'
    model_path = "/data/personal/baoyh/baichuan/baichuan-lastest"
    tokenizer_path = "/data/personal/baoyh/baichuan/baichuan-lastest"
    N = 5
    main(input_file, model_path, tokenizer_path, N)

