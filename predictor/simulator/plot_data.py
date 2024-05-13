import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data into a pandas DataFrame
data = pd.read_csv('download_times.csv')

# Plot the distribution of the time column
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='Time (microseconds)', kde=True)
plt.title('Distribution of Time (microseconds)')
plt.xlabel('Time (microseconds)')
plt.ylabel('Count')
plt.savefig('time_distribution_download.png', bbox_inches='tight')

# Plot a scatter plot of fraction vs time
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='MemSize', y='Time (microseconds)')
plt.title('Scatter Plot of MemSize vs Time')
plt.xlabel('MemSize')
plt.ylabel('Time (microseconds)')
plt.savefig('memsize_vs_time_download.png', bbox_inches='tight')