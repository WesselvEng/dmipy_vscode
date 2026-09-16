import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis('off')

# Face
face = plt.Circle((0, 0), 1.2, facecolor='#f7d35c', edgecolor='black', linewidth=2)
ax.add_patch(face)

# Eyes
left_eye = plt.Circle((-0.42, 0.45), 0.12, facecolor='black')
right_eye = plt.Circle((0.42, 0.45), 0.12, facecolor='black')
ax.add_patch(left_eye)
ax.add_patch(right_eye)

# Smile
theta = np.linspace(np.pi * 0.25, np.pi * 0.75, 200)
x = 0.8 * np.cos(theta)
y = 0.55 * np.sin(theta) - 0.15
ax.plot(x, y, color='black', linewidth=3)

plt.show()
# plt.savefig('smiley_ggplot.png', dpi=200, bbox_inches='tight')

