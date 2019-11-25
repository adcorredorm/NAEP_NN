import pandas as pd
from datetime import datetime
from truncate import read_time

data = pd.read_csv('processed_data/data_a_train_30m.csv')
ids = data.STUDENTID.unique()

headers = ['ID', 'Time Spent', 'Total Questions', 'Total Helps', 'Average Time Spent']
result = []

for _id in ids:
    query = 'STUDENTID == {}'.format(_id)
    raw = data.query(query)

    questions = raw[raw['AccessionNumber'].str.startswith('VH') == True].AccessionNumber.unique()
    helps = raw[raw['AccessionNumber'].str.startswith('HELP') == True].AccessionNumber.unique()

    time_spent = read_time(raw.iloc[-1][6]) - read_time(raw.iloc[0][6])
    total_questions = len(questions)
    total_helps = len(helps)
    average = time_spent / total_questions

    result.append([
        _id, time_spent.total_seconds(), total_questions, total_helps, average.total_seconds()
    ])


df = pd.DataFrame(result, columns=headers)
df['Percentile Total Time Rank'] = df['Time Spent'].rank(pct=True)
df['Percentile Average Time Rank'] = df['Average Time Spent'].rank(pct=True)
df.to_csv('processed_data/features_30m.csv')
