import csv
import numpy as np
import matplotlib.pyplot as plt

'''
The purpose of this file is to read test results from CSV-files
created in /TestResultsContainer/TestResults.
This needs to be modified according to what data is in the file, 
but the loop that reads from the file (see below) should be 
pretty much the same. 
'''

'''
# This code is for testing convergence to outside temperature.
# Please do not delete, as we may need to redo this test.
num_time_steps = 43200
time = np.array(list(range(num_time_steps)))
temp_max = np.zeros(num_time_steps)
temp_min = np.zeros(num_time_steps)
temp_avg = np.zeros(num_time_steps)
outside_temp = 173.15 * np.ones(num_time_steps)-273.15

plt.plot(time, temp_max-273.15, color='red')
plt.plot(time, temp_min-273.15, color='green')
plt.plot(time, temp_avg-273.15, color='orange')
plt.plot(time, outside_temp, color='blue')
'''

# Todo: Modify num_time_steps here, when you know what this number is.
# Todo: 86402 is the go-to for 24hour-runs.
num_time_steps = 86402
time = np.array(list(range(num_time_steps)))
avg_temp = np.zeros(num_time_steps)
ideal_temp = np.zeros(num_time_steps)

_nonzero_line_count = 0
# Todo: The code for reading the CSV file is here. Modify after 'if row:' as you see fit.
with open('TestResultsContainer/TestResults', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    '''The CSV file will contain empty lines because of incompatibilities 
    with windows. We therefore ignore the empty row. A row of the 
    CSV file is on the form ['bla,blabla,lol'], so that
    row[0] = 'bla, blabla, lol', and 
    row[0].split(',') = ['bla', 'blabla', 'lol'].
    '''
    for row in csv_reader:
        if row:
            if _nonzero_line_count == num_time_steps:
                continue
            _, avg, ideal = row[0].split(',')
            avg_temp[_nonzero_line_count] = float(avg)
            ideal_temp[_nonzero_line_count] = float(ideal)
            # The next line must not be deleted: It keeps track of the nonzero CSV-file-lines.
            _nonzero_line_count += 1

# Set high resolution of the plot
plt.figure(dpi=200)
plt.plot(time, avg_temp)
plt.plot(time, ideal_temp)
plt.xlabel('Tidssteg')
plt.ylabel(r'Temperatur [C]')
plt.grid('minor')
plt.legend(['room avg. temp', 'ideal temp'])
# General format:
# plt.legend(['maks', 'min', 'gjennomsnitt', 'ute'])
plt.show()

