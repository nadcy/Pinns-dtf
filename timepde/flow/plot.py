# Open the text file for reading
import matplotlib.pyplot as plt
u1 = []
with open('timepde/flow/u1.txt', 'r') as f:
    # Loop through each line in the file
    for line in f:
        # Split the line into a list of values
        values = line.split()
        # Extract the second value and print it
        u1.append(float(values[2]))

u2 = []
with open('timepde/flow/u2.txt', 'r') as f:
    # Loop through each line in the file
    for line in f:
        # Split the line into a list of values
        values = line.split()
        # Extract the second value and print it
        u2.append(float(values[2]))

plt.plot(u1,label='udPINNs:new-method')
plt.plot(u2,label='XPINNs:baseline')
plt.legend()
plt.show()