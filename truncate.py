import csv
from datetime import datetime, timedelta

def read_time(str_date):
    try:
        return datetime.strptime(str_date, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        return datetime.today()

def truncate(file_path, result_path, time):
    delta = timedelta(minutes=time)
    with open(file_path, 'r+') as source, open(result_path, 'w+') as result:
        reader = list(csv.reader(source))
        writer = csv.writer(result)
        # Write headers
        writer.writerow(reader[0])
        
        current_id = reader[1][0]
        initial_time = read_time(reader[1][6])
        for line in reader[1:]:
            if current_id != line[0]:
                current_id = line[0]
                initial_time = read_time(line[6])

            if read_time(line[6]) - initial_time <= delta:
                writer.writerow(line)

if __name__ == '__main__':
    data_path = 'public_data/data_a_train.csv'
    result_path = 'processed_data/data_a_train_5m.csv'

    truncate(data_path, result_path, 5)