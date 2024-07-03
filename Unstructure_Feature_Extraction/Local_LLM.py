# -*- coding;utf-8 -*-
"""
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import pandas as pd


class LocalExtractor:
    def __init__(self, model_path="/data/personal/baoyh/baichuan/baichuan-lastest", tokenizer_path="/data/personal/baoyh/baichuan/baichuan-lastest"):
        """
        Initialize the model and tokenizer.

        Parameters:
        model_path (str): Path to the model.
        tokenizer_path (str): Path to the tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                          trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.prompt1 = '''
        你是一个严谨的患者特征提取专家，请查阅下面的病历，告诉我病历是否直接显示患者有冲动行为。请回答是或否
        ===
        ### 病历：
        '''
        self.prompt2 = '''
        ===
        ### 要求：
        1. 回答只需要是：“是”或“否”
        2. 如不确定，请回答不确定
        '''

    def generate_response(self, prompt, max_attempts=5):
        """
        Generate a response from the model for a given prompt.

        Parameters:
        prompt (str): The input prompt.
        max_attempts (int): Maximum number of attempts to generate a consistent response.

        Returns:
        str: The final determined response ('是', '否', or '不确定').
        """
        messages = [{"role": "user", "content": prompt}]
        response_list = []

        for _ in range(max_attempts):
            response = self.model.chat(self.tokenizer, messages)
            if '是否' in response:
                response_list.append('不确定')
            elif '是' in response:
                response_list.append('是')
            elif '否' in response:
                response_list.append('否')
            else:
                response_list.append('不确定')

        # Check if the responses are consistent
        if all(response == response_list[0] for response in response_list):
            return response_list[0]
        else:
            return '不确定'

    def analyze_patient_records(self, df):
        """
        Analyze patient records to determine impulsive behavior.

        Parameters:
        df (pd.DataFrame): The input dataframe containing patient records.

        Returns:
        pd.DataFrame: The dataframe with the analysis results.
        """
        df['冲动'] = ''
        for index, row in df.iterrows():
            class_message = row['harm_sentence']
            combined_prompt = self.prompt1 + class_message + self.prompt2
            result = self.generate_response(combined_prompt)
            df.loc[index, '冲动'] = result
        return df

# Example usage:
# from your_module import LocalExtractor
#
# model_path = "/data/personal/baoyh/baichuan/baichuan-lastest"
# tokenizer_path = "/data/personal/baoyh/baichuan/baichuan-lastest"
#
# extractor = LocalExtractor(model_path, tokenizer_path)
# df = pd.read_csv('path_to_patient_records.csv')
# analyzed_df = extractor.analyze_patient_records(df)
# print(analyzed_df)



