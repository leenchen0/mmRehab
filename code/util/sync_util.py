import csv

def read_csv_cell(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if len(row) > 0 and 'Capture end time' in row[0]:
                return row[0]