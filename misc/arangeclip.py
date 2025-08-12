import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initial parameters
a = 1
sigma_init = 1
mu_init = 0.0

# Data generation function
def generate_data(alow, ahigh, sigma, mu):
    x = np.random.normal(mu, sigma, size=10000)
    x_clipped = np.clip(x, alow, ahigh)
    x_tanh = np.tanh(x) * (ahigh - alow) / 2 + (ahigh + alow) / 2
    return x, x_clipped, x_tanh

# Initial data
x, x_clipped, x_tanh = generate_data(-a, a, sigma_init, mu_init)

# Plot setup
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.4)
h1 = ax.hist(x, bins=100, alpha=0.4, label='$\mathcal{N}(\mu_\\theta, \sigma_\\theta)$')
h2 = ax.hist(x_clipped, bins=100, alpha=0.4, label='$\mathrm{clip}(\mathcal{N}(\mu_\\theta, \sigma_\\theta), \mathcal{A}_{low}, \mathcal{A}_{high})$')
h3 = ax.hist(x_tanh, bins=100, alpha=0.4, label='$\\tanh(\mathcal{N}(\mu_\\theta, \sigma_\\theta))\\frac{\mathcal{A}_{high} - \mathcal{A}_{low}}{2} + \\frac{\mathcal{A}_{high} + \mathcal{A}_{low}}{2}$')
ax.legend()
ax.set_title("Clipping vs Tanh on Normal Samples")

# Slider axes
ax_alow = plt.axes([0.1, 0.3, 0.8, 0.03])
ax_sigma = plt.axes([0.1, 0.2, 0.8, 0.03])
ax_mu = plt.axes([0.1, 0.1, 0.8, 0.03])

# Sliders
s_a = Slider(ax_alow, 'alow', 0, 5.0, valinit=a)
s_sigma = Slider(ax_sigma, 'sigma', 0.0, 2.0, valinit=sigma_init)
s_mu = Slider(ax_mu, 'mu', -2, 2, valinit=mu_init)

# Update function
def update(val):
    alow = -s_a.val
    ahigh = s_a.val
    sigma = s_sigma.val
    mu = s_mu.val
    if alow >= ahigh:
        return  # Prevent invalid range
    x, x_clipped, x_tanh = generate_data(alow, ahigh, sigma, mu)
    ax.cla()
    # Plot histograms and get colors
    h1 = ax.hist(x, bins=100, alpha=0.4, label='$\mathcal{N}(\mu_\\theta, \sigma_\\theta)$')
    h2 = ax.hist(x_clipped, bins=100, alpha=0.4, label='$\mathrm{clip}(\mathcal{N}(\mu_\\theta, \sigma_\\theta), \mathcal{A}_{low}, \mathcal{A}_{high})$')
    h3 = ax.hist(x_tanh, bins=100, alpha=0.4, label='$\\tanh(\mathcal{N}(\mu_\\theta, \sigma_\\theta))\\frac{\mathcal{A}_{high} - \mathcal{A}_{low}}{2} + \\frac{\mathcal{A}_{high} + \mathcal{A}_{low}}{2}$')
    
    ax.legend()
    ax.set_title("Clipping vs Tanh on Normal Samples")

    ax.plot([np.mean(x),np.mean(x)], [30,0], color='blue')
    ax.plot([np.mean(x_clipped),np.mean(x_clipped)], [30,0], color='orange')
    ax.plot([np.mean(x_tanh),np.mean(x_tanh)], [30,0], color='green')
    ax.set_yticks([])

    fig.canvas.draw_idle()

s_a.on_changed(update)
s_sigma.on_changed(update)
s_mu.on_changed(update)

plt.show()