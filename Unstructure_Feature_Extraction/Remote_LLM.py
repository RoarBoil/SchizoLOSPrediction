# -*- coding;utf-8 -*-
"""
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
"""

import openai
import pandas as pd
from Evaluation import evaluate_predictions

openai.api_key = 'sk-...'  # openai api key, you can get it from https://platform.openai.com/account/api-keys


def optimize_prompt(prompt, evaluation_summary):
    optimization_template = f"""
    Here is the current prompt for a language model and its evaluation summary on a validation set:

    Current Prompt:
    {prompt}

    Evaluation Summary:
    {evaluation_summary}

    Please optimize the prompt to improve the model's performance, ensuring it is clear, concise, and effective.
    Your response should only contain optimized prompt

    """
    def request_optimized_prompt(template):
        response = openai.Completion.create(
            engine="gpt-4o",
            prompt=template,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].text.strip()

    response_text = request_optimized_prompt(optimization_template)
    if response_text and len(response_text) > 10:
        return response_text

