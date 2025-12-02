import csv
import json
def read_json(path):


    # Opening JSON file
    f = open(path)

    # returns JSON object as
    # a dictionary
    data = json.load(f)



    # Closing file
    f.close()

    return data

def read_csv(path):
    readed_rows=[]
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                line_count += 1
                readed_rows.append(row)
        print(f'Processed {line_count} lines.')
        return readed_rows