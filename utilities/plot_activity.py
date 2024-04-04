
import matplotlib.pyplot as plt
import pandas as pd

def plot_activities(filename):
  """
  Plots profit and loss values across time for each product in a file.

  Args:
    filename: The path to the activities log file.
  """
  # Read the data from the CSV file
  data = pd.read_csv(filename, delimiter=";")

  # Separate data by product
  products = data["product"].unique()

  # Create a plot for each product
  for product in products:
    product_data = data[data["product"] == product]

    # Convert timestamp to seconds
    product_data["timestamp"] = product_data["timestamp"] / 1000

    # Plot profit and loss vs time
    plt.plot(product_data["timestamp"], product_data["profit_and_loss"], label=product)

  # Add labels and title
  plt.xlabel("Time (seconds)")
  plt.ylabel("Profit and Loss")
  plt.title("Profit and Loss over Time")
  plt.legend()

  # Show the plot
  plt.show()

# Specify the filename
filename = "activities.csv"  # Replace with your actual filename

plot_activities(filename)
