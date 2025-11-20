import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from pathlib import Path
import matplotlib.pyplot as plt

#Import file path
data = Path("/Users/reesecoleman/Desktop/UCAR Data/data/FormatCSVs/Pressure.csv")

#Export path
destination = Path("/Users/reesecoleman/Desktop/UCAR Data/data/FormatCSVs")

#Sensor target
sensor_column = 'lps1t'



# Read CSV file
dataframe = pd.read_csv(data)

# Convert 'time' column from string to datetime (let pandas infer format)
dataframe['time'] = pd.to_datetime(dataframe['time'], errors='coerce')

# Drop rows with missing or invalid 'time' or sensor_column
dataframe = dataframe.dropna(subset=['time', sensor_column])

# Save original datetime for axis labeling
datetime_col = dataframe['time']

# Convert 'time' to Unix timestamp for Datashader
dataframe['time_num'] = datetime_col.astype('int64') // 10**9

# Create a Datashader Canvas
canvas = ds.Canvas(plot_width=1000, plot_height=400)
agg = canvas.line(dataframe, 'time_num', sensor_column)

# Convert to image
img = tf.shade(agg)


# Display with matplotlib and add readable date/time labels
plt.figure(figsize=(12, 6))
plt.imshow(img.to_pil(), aspect='auto')

# Generate evenly spaced ticks for the x-axis
num_ticks = 10
min_time = datetime_col.min()
max_time = datetime_col.max()
tick_times = pd.date_range(start=min_time, end=max_time, periods=num_ticks)

# Convert to seconds since epoch
min_time_sec = min_time.timestamp()
max_time_sec = max_time.timestamp()
tick_times_sec = tick_times.astype('int64') // 10**9

# Map tick positions to image width
tick_pos = (tick_times_sec - min_time_sec) * (1000 / (max_time_sec - min_time_sec))

''' # 24hr format
plt.xticks(
    ticks=tick_pos,
    labels=[dt.strftime("%Y-%m-%d %H:%M") for dt in tick_times],
    rotation=45
)
'''

# 12hr format
plt.xticks(
    ticks=tick_pos,
    labels=[dt.strftime("%m/%d/%y %I:%M %p") for dt in tick_times],  
    rotation=45
)

plt.xlabel("Time")
plt.ylabel("Pressure (lps1t)")
plt.title("Pressure Over Time")
plt.legend(["Pressure"], loc="upper right")
plt.tight_layout()
plt.savefig(destination / "pressure_plot.png")
plt.show()
plt.close()