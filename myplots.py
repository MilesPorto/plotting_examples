import numpy as np
import matplotlib.pyplot as plt

def sample_inv_cdf(size=1):
    u = np.random.rand(size) 
    return 1.0 / (1 - u * (1 - 1/11))

# Parameters
mean = 100
sigma = 6
N = 10000
bins = 100
range_min, range_max = 50, 150


samples_gauss = np.random.normal(mean, sigma, N)

# Plot 1: Gaussian only
counts_g, bin_edges = np.histogram(samples_gauss, bins=bins, range=(range_min, range_max))
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
errors_g = np.sqrt(counts_g)

plt.errorbar(bin_centers, counts_g, yerr=errors_g, fmt='o', color='blue',
             markersize=3)

plt.xlabel("x")
plt.ylabel("Frequency")
plt.title("Random Gauss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Stats box
mean_g = np.mean(samples_gauss)
std_g = np.std(samples_gauss, ddof=1)
textstr = '\n'.join((
    f'Entries = {N}',
    f'Mean = {mean_g:.2f}',
    f'Std Dev = {std_g:.2f}',
))
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(0.70, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=props)

plt.savefig("canvas1_py.png", dpi=300)
plt.close() 

# Plot 2: 4 plots
uniform_samples = np.random.uniform(50, 150, N//3)
samples_base2 = sample_inv_cdf(N*30)*10+40
more_gauss = np.random.normal(mean,20,N//2)
samples_offset = np.concatenate([samples_gauss, uniform_samples])
samples_offset2 = np.concatenate([samples_gauss, samples_base2])
samples_doubleGauss = np.concatenate([samples_gauss, more_gauss])

# Histogram for offset
counts_o, _ = np.histogram(samples_offset, bins=bins, range=(range_min, range_max))
errors_o = np.sqrt(counts_o)
mean_o = np.mean(samples_offset)
std_o = np.std(samples_offset, ddof=1)

counts_o2, _ = np.histogram(samples_offset2, bins=bins, range=(range_min, range_max))
errors_o2 = np.sqrt(counts_o2)
mean_o2 = np.mean(samples_offset2)
std_o2 = np.std(samples_offset2, ddof=1)

counts_2g, _ = np.histogram(samples_doubleGauss, bins=bins, range=(range_min, range_max))
errors_2g = np.sqrt(counts_2g)
mean_2g = np.mean(samples_doubleGauss)
std_2g = np.std(samples_doubleGauss, ddof=1)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

titles = ["Random Gauss", "Gauss+offset1", "Gauss+offset2", "Double Gauss"]
axes_flat = axes.flatten()
for i, ax in enumerate(axes_flat):
    ax.set_xlabel("x")
    ax.set_ylabel("Frequency")
    ax.set_title(titles[i])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

axes[0,0].errorbar(bin_centers, counts_g, yerr=errors_g, fmt='o', color='blue', markersize=3)

textstr_g = '\n'.join((
    f'Entries = {N}',
    f'Mean = {mean_g:.2f}',
    f'Std Dev = {std_g:.2f}',
))
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
axes[0,0].text(0.70, 0.95, textstr_g, transform=axes[0,0].transAxes, fontsize=10, verticalalignment='top', bbox=props)

axes[0,1].errorbar(bin_centers, counts_o, yerr=errors_o, fmt='o', color='blue', markersize=3)

textstr_o = '\n'.join((
    f'Entries = {len(samples_offset)}',
    f'Mean = {mean_o:.2f}',
    f'Std Dev = {std_o:.2f}',
))
axes[0,1].text(0.70, 0.95, textstr_o, transform=axes[0,1].transAxes, fontsize=10, verticalalignment='top', bbox=props)

axes[1,0].errorbar(bin_centers, counts_o2, yerr=errors_o2, fmt='o', color='blue', markersize=3)
axes[1,0].set_yscale("log")
axes[1, 0].set_ylim(bottom=1e2) 

textstr_o2 = '\n'.join((
    f'Entries = {len(samples_offset2)}',
    f'Mean = {mean_o2:.2f}',
    f'Std Dev = {std_o2:.2f}',
))
axes[1,0].text(0.70, 0.95, textstr_o2, transform=axes[1,0].transAxes, fontsize=10, verticalalignment='top', bbox=props)

axes[1,1].errorbar(bin_centers, counts_2g, yerr=errors_2g, fmt='o', color='blue', markersize=3)

textstr_2g = '\n'.join((
    f'Entries = {len(samples_doubleGauss)}',
    f'Mean = {mean_2g:.2f}',
    f'Std Dev = {std_2g:.2f}',
))
axes[1,1].text(0.70, 0.95, textstr_2g, transform=axes[1,1].transAxes, fontsize=10, verticalalignment='top', bbox=props)

# Adjust layout and save
plt.tight_layout()

plt.savefig("canvas2_py.pdf")
plt.close()

