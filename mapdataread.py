import numpy as np
import matplotlib.pyplot as plt

omap = np.loadtxt('map.txt')
print(f"Map shape: {omap.shape}")
print(f"Map min/max values: {omap.min()}, {omap.max()}")

plt.imshow(omap, origin='lower')
plt.colorbar()
plt.title('Occupancy Map')
plt.savefig('map_visualization.png')  # Save to file instead of showing
print("Map saved as map_visualization.png")