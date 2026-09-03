

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

def load_data(path=config.DATA_PATH):
    df = pd.read_csv(path)
    print('Data Shape:', df.shape)
    print('Class Balance:\n')
    print(df['Class'].value_counts())
    print('Fraudulent rate: {:.4f}%'.format(df['Class'].mean() * 100))
    return df


def explore_data(df):
    print('Missing values:', df.isnull().sum().sum())

    print('Amount summary by class:\n')
    print(df.groupby('Class')['Amount'].describe())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, ax=axes[0], color='blue')
    axes[0].set_title('Non-Fraudulent Transactions Amount Distribution')
    axes[0].set_xlim(0, 2000)

    sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, ax=axes[1], color='red')
    axes[1].set_title('Fraudulent Transactions Amount Distribution')
    axes[1].set_xlim(0, 2000)

    plt.tight_layout()
    # out_path = output_path = os.path.join(os.getcwd(), 'output_distribution.png')
    # plt.savefig(out_path)
    # print(f'Distribution plots saved to {out_path}')

    plt.show()




if __name__ == "__main__":
    df = load_data()
    explore_data(df)
