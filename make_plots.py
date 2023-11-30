# Open the file and read its contents
with open('results.6.txt', 'r') as file:
    data = file.readlines()

# Filter the data
filtered_data = []
for line in data:
    if "lvl_type='scramble8'" in line and "win_reward=100.0" in line:
        filtered_data.append(line)

# Extract the required data for plotting
runs = []
percent_passed = []
config_ids = []

for line in filtered_data:
    split_line = line.split(',')
    for item in split_line:
        if "ep*batch" in item:
            runs.append(int(item.split(':')[1].strip()))
        if "%" in item:
            ppased = item.split('%')[0].strip()
            ppased = int(ppased.split(' ')[-1]) 
            percent_passed.append(ppased)
        if "config_" in item and "passed" in item:
            conid = item.split(':')[0].split('_')[1]
            conid = conid.split(' ')[0]
            config_ids.append(int(conid))


runs, percent_passed, config_ids


print(f"Runs: {runs}")
print(f"Percent passed: {percent_passed}")
print(f"Config ids: {config_ids}")



import matplotlib.pyplot as plt

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(runs, percent_passed)

# Add labels for each data point
for i in range(len(runs)):
    plt.annotate(config_ids[i], (runs[i], percent_passed[i]))

import matplotlib.ticker as mticker
# Format the x-axis labels to display full numbers
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))


# Add labels for the axes
plt.xlabel('Training Runs')
plt.ylabel('Tests Passed')

# Show the plot
plt.show()
