import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 6))

# Function to draw a "stack" of layers
def draw_stack(x, y, width, height, num_layers, color, label):
    for i in range(num_layers):
        # Offset each layer to look 3D
        offset = i * 0.05
        rect = patches.Rectangle((x + offset, y + offset), width, height, 
                                 linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
    plt.text(x + 0.2, y - 0.5, label, fontsize=12, fontweight='bold')

# 1. Input (Represented as a thick bar for 64 channels)
draw_stack(1, 2, 0.5, 6, 1, 'lightsteelblue', "64 Input Channels")

# 2. Convolution Kernel (The Red Box)
kernel = patches.Rectangle((1, 6), 0.5, 1.5, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(kernel)

# 3. Arrow
ax.annotate('', xy=(3.5, 5), xytext=(1.6, 6.5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1))

# 4. Output Stack (Representing 256 channels with a deep stack)
draw_stack(4, 3, 1, 4, 8, 'cornflowerblue', "256 Output Channels")

# Formatting
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
plt.axis('off')
plt.title("Conv1D: 64 to 256 Transition", fontsize=15)
plt.show()
