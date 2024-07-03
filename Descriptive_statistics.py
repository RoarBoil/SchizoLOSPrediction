# -*- coding;utf-8 -*-
"""
File Create By: Yihang Bao
email: baoyihang@sjtu.edu.cn
"""

import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt


def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, header=0)
    df.dropna(subset=['label'], inplace=True)
    df = df[df['住院天数'] <= 365]
    return df


def print_label_distribution(df):
    print("Label distribution:")
    print(df.value_counts('label'))


def describe_column_by_label(df, label, column):
    print(f"Describing column '{column}' for label {label}:")
    description = df[df['label'] == label][column].describe()
    print(description)


def create_violin_plots(df, feature_list):
    plt.rcParams['font.family'] = 'SimHei'
    df = df[feature_list]
    df.columns = ['CRP', 'Neutrophil', 'Neutrophil%', 'Monocytes', 'Monocytes%', 'Basophilic granulocyte',
                  'Basophilic granulocyte%', 'Acidophilic granulocyte', 'Acidophilic granulocyte%', 'Total bilirubin',
                  'Chlorine', 'Lymphocyte', 'Lymphocyte%', 'Globular proteins', 'Albumin', 'ALP', 'Platelet',
                  'Hemoglobin',
                  'ALT', 'γ-GT', 'Natrium', 'Kalium', 'NLR', 'PLR', 'SII', 'MLR', 'label']
    df_short = df[df['label'] == 0].drop(columns=['label'])
    df_long = df[df['label'] == 1].drop(columns=['label'])
    columns = df_short.columns
    cols_per_row = 4
    num_rows = (len(columns) + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(nrows=num_rows, ncols=min(len(columns), cols_per_row),
                             figsize=(4 * min(len(columns), cols_per_row), 3 * num_rows), dpi=600)
    axes = axes.flatten() if num_rows > 1 else [axes]
    colors = sns.color_palette("viridis", as_cmap=False)
    units = ['mg/L', '10^9/L', '%', '10^9/L', '%', '10^9/L', '%', '10^9/L', '%', 'umol/L', 'mmol/L', '10^9/L',
             '%', 'g/L', 'g/L', 'IU/L', '10^9/L', 'g/L', 'IU/L', 'IU/L', 'mmol/L', 'mmol/L', ' ', ' ', ' ', ' ']
    for ax, col, unit in zip(axes, columns, units):
        stat, p_value = mannwhitneyu(df_short[col], df_long[col], alternative='two-sided')
        temp_df = pd.concat([df_short[col], df_long[col]], keys=['short', 'long']).reset_index(level=0).rename(
            columns={'level_0': 'Length of Stay', 0: col})
        sns.violinplot(x='Length of Stay', y=col, data=temp_df, ax=ax, palette=colors)
        ax.set_title(f'{col}\n(p={p_value:.3e})')
        ax.set_facecolor('none')
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.set_ylabel(unit)
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()


def create_count_plots(df):
    df['sex'] = df['性别'].apply(lambda x: 'Male' if x == '男性' else 'Female')
    df['Marital status'] = df['婚姻状态'].apply(
        lambda x: 'Married' if x == '已婚' else ('Divorced/Widowed' if x in ['离婚', '丧偶'] else 'Unmarried'))
    df.rename(columns={'dis_days': 'Disease course', '入院年龄': 'Age of admission'}, inplace=True)
    df_short = df[df['label'] == 0].drop(columns=['label']).reset_index(drop=True)
    df_long = df[df['label'] == 1].drop(columns=['label']).reset_index(drop=True)
    columns = df_short.columns
    cols_per_row = 2
    num_rows = (len(columns) + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(nrows=num_rows, ncols=min(len(columns), cols_per_row),
                             figsize=(4 * min(len(columns), cols_per_row), 3 * num_rows), dpi=600)
    axes = axes.flatten() if num_rows > 1 else [axes]
    units = ['Days', 'Years', ' ', ' ']
    colors = sns.color_palette("viridis", as_cmap=False)
    for ax, col, unit in zip(axes, columns, units):
        if col in ['Disease course', 'Age of admission']:
            stat, p_value = mannwhitneyu(df_short[col], df_long[col], alternative='two-sided')
            temp_df = pd.concat([df_short[col], df_long[col]], keys=['short', 'long']).reset_index(level=0).rename(
                columns={'level_0': 'Type', 0: col})
            sns.violinplot(x='Type', y=col, data=temp_df, ax=ax, palette=colors)
        else:
            temp_df = pd.concat([df_short[col], df_long[col]], keys=['short', 'long']).reset_index(level=0).rename(
                columns={'level_0': 'Length of Stay', 0: col})
            contingency_table = pd.crosstab(temp_df[col], temp_df['Length of Stay'])
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            sns.countplot(x=col, hue='Length of Stay', data=temp_df, ax=ax, palette=colors)
            ax.set_ylabel('Count')
        ax.set_title(f'{col}\n(p={p_value:.3e})')
        ax.set_facecolor('none')
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.set_ylabel(unit)
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()


def create_correlation_heatmap(df, feature_list):
    plt.rcParams['font.family'] = 'SimHei'
    df = df[feature_list]
    df.columns = ['CRP', 'Neutrophil', 'Neutrophil%', 'Monocytes', 'Monocytes%', 'Basophilic granulocyte',
                  'Basophilic granulocyte%', 'Acidophilic granulocyte', 'Acidophilic granulocyte%', 'Total bilirubin',
                  'Chlorine', 'Lymphocyte', 'Lymphocyte%', 'Globular proteins', 'Albumin', 'ALP', 'Platelet',
                  'Hemoglobin',
                  'ALT', 'γ-GT', 'Natrium', 'Kalium', 'NLR', 'PLR', 'SII', 'MLR', 'Disease course', 'Age of admission']
    correlation_matrix = df.corr()
    plt.figure(figsize=(14, 11.2), dpi=600)
    plt.rcParams['font.family'] = 'Times New Roman'
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5, linecolor='white')
    plt.title('Correlation Matrix')
    plt.show()


def main():
    file_path = 'data1.csv'
    df = load_and_clean_data(file_path)

    print_label_distribution(df)

    describe_column_by_label(df, 1, 'xxx')
    describe_column_by_label(df, 0, 'xxx')

    feature_list = []  # fill the feature list here
    create_violin_plots(df, feature_list)

    create_count_plots(df)

    create_correlation_heatmap(df, feature_list)


if __name__ == "__main__":
    main()