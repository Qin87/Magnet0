import numpy as np

# Replace the following lines with your actual data
data = [
    [25.00, 23.21, 13.25],
    [24.56, 23.35, 13.58],
    [25.00, 24.01, 13.94],
    [26.32, 24.47, 13.99],
    [22.37, 20.00, 7.31],
    [22.37, 20.00, 7.31],
    [22.81, 20.00, 7.43],
    [21.27, 18.97, 9.99],
    [15.79, 17.18, 8.24],
    [24.56, 20.00, 7.89],
    [32.24, 29.74, 17.69],
    [24.56, 20.14, 9.03],
    [21.71, 19.32, 11.35],
    [24.56, 20.00, 7.89],
    [21.49, 20.00, 7.08],
    [26.54, 24.79, 14.87],
    [25.88, 24.20, 14.15],
    [26.97, 25.25, 15.55],
    [20.61, 19.24, 8.64],
    [19.30, 20.00, 6.47],
    [23.46, 20.00, 7.67],
    [21.93, 23.40, 16.19],
    [23.25, 20.00, 7.54],
    [23.46, 20.00, 7.62],
    [25.66, 20.65, 9.57],
    [25.22, 20.00, 8.06],
    [25.22, 20.00, 8.06],
    [27.63, 22.65, 12.55],
    [23.90, 19.57, 10.60],
    [20.18, 20.00, 6.72],
    [20.18, 20.00, 6.72],
    [20.18, 20.00, 6.72],
    [19.96, 19.78, 6.72],
    [20.18, 20.00, 6.72],
    [23.03, 22.43, 12.84],
    [21.27, 20.00, 7.02],
    [21.27, 20.00, 7.02],
    [21.71, 21.50, 11.22],
    [19.52, 20.65, 9.21],
    [23.25, 20.00, 7.54],
    [23.25, 20.00, 7.54],
    [27.41, 25.97, 16.22],
    [22.37, 19.43, 9.86],
    [23.25, 20.00, 7.54],
    [19.74, 20.00, 6.59],
    [22.15, 23.72, 17.95],
    [21.27, 21.16, 9.35],
    [24.34, 19.98, 8.74],
    [24.56, 20.13, 9.05]
]

# Convert data to a numpy array for easier calculations
data_array = np.array(data)

# Extract columns for test_Acc, test_bacc, and test_f1
test_acc = data_array[:, 0]
test_bacc = data_array[:, 1]
test_f1 = data_array[:, 2]

# Calculate the mean (average) and standard deviation for each metric
average_test_acc = np.mean(test_acc)
std_deviation_test_acc = np.std(test_acc)

average_test_bacc = np.mean(test_bacc)
std_deviation_test_bacc = np.std(test_bacc)

average_test_f1 = np.mean(test_f1)
std_deviation_test_f1 = np.std(test_f1)

# Display the results
print(f"Average test_Acc: {average_test_acc:.2f}, Standard deviation test_Acc: {std_deviation_test_acc:.2f}")
print(f"Average test_bacc: {average_test_bacc:.2f}, Standard deviation test_bacc: {std_deviation_test_bacc:.2f}")
print(f"Average test_f1: {average_test_f1:.2f}, Standard deviation test_f1: {std_deviation_test_f1:.2f}")
