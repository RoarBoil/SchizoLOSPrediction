# -*- coding;utf-8 -*-
"""
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
"""
import pandas as pd

def evaluate_predictions(df, correct_column='correct_answer', predicted_column='冲动'):
    correct_count = 0
    incorrect_count = 0
    unsure_count = 0

    for _, row in df.iterrows():
        correct_answer = row[correct_column]
        predicted_answer = row[predicted_column]

        if predicted_answer == correct_answer:
            correct_count += 1
        elif predicted_answer == '不确定':
            unsure_count += 1
        else:
            incorrect_count += 1

    total_count = len(df)
    correct_percentage = (correct_count / total_count) * 100
    incorrect_percentage = (incorrect_count / total_count) * 100
    unsure_percentage = (unsure_count / total_count) * 100

    summary = (
        f"在验证集上，共有 {total_count} 个样本。\n"
        f"其中，正确回答的样本数为 {correct_count} 个\n"
        f"错误回答的样本数为 {incorrect_count} 个。\n"
        f"不确定回答的样本数为 {unsure_count} 个\n"
    )

    return summary
