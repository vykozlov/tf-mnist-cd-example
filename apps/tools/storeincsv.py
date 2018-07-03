"""
Helper functions to store data in file
@author: valentin
"""

import csv

def store_data_in_csv(csv_file, param_entries):
    """Helper function to store data in CSV format
    Args: 
        csv_file:      file where to store data
        param_entries: array of parameters to be stored
    """
    with open(csv_file, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow("")
        for param_entry in param_entries:
            writer.writerow(param_entry)
