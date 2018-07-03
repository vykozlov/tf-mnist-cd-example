"""
Helper function to store data in CSV format
@author: valentin
"""

import csv

def store_data_in_csv(csv_file, param_entries):
    with open(csv_file, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow("")
        for param_entry in param_entries:
            writer.writerow(param_entry)
