import statistics

delivery_times = [
    22, 25, 27, 28, 29, 30, 30, 31,
    32, 33, 34, 35, 36, 38, 40,
    82, 95, 110
]

mean_value = statistics.mean(delivery_times)
median_value = statistics.median(delivery_times)
mode_value = statistics.mode(delivery_times)
variance_value = statistics.variance(delivery_times)
std_value = statistics.stdev(delivery_times)

print("=" * 45)
print("DELIVERY TIME STATISTICAL REPORT")
print("=" * 45)
print(f"{'Mean':<25}: {mean_value:.2f} minutes")
print(f"{'Median':<25}: {median_value:.2f} minutes")
print(f"{'Mode':<25}: {mode_value:.2f} minutes")
print(f"{'Variance':<25}: {variance_value:.2f}")
print(f"{'Standard Deviation':<25}: {std_value:.2f} minutes")
print("=" * 45)

if mean_value > median_value:
    print("Conclusion: The distribution is right-skewed.")
elif mean_value < median_value:
    print("Conclusion: The distribution is left-skewed.")
else:
    print("Conclusion: The distribution is approximately symmetrical.")