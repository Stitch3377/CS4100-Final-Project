import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
old_epochs = range(1, 11)
old_loss = [0.8461, 0.6920, 0.6909, 0.6856, 0.6849, 0.6823, 0.6812, 0.6804, 0.6773, 0.6714]
old_f1 = [0.1819, 0.1817, 0.1818, 0.1678, 0.1497, 0.1533, 0.1218, 0.1229, 0.1544, 0.1175]

new_epochs = range(1, 21)
new_loss = [0.7472, 0.6865, 0.6404, 0.5894, 0.5232, 0.4220, 0.3286, 0.2578, 0.1856, 0.1519, 
            0.1130, 0.0953, 0.0774, 0.0733, 0.0664, 0.0584, 0.0564, 0.0523, 0.0526, 0.0532]
new_f1 = [0.2740, 0.3188, 0.3498, 0.3584, 0.3910, 0.4048, 0.4038, 0.4034, 0.4207, 0.4180, 
          0.4288, 0.4286, 0.4279, 0.4325, 0.4279, 0.4396, 0.4331, 0.4347, 0.4425, 0.4486]

plt.style.use('ggplot')

# --- Function to Draw Graph ---
def plot_arch(epochs, loss, f1, title, filename, color_loss='tab:red', color_f1='tab:blue'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Plot
    ax1.plot(epochs, loss, marker='o', color=color_loss, linewidth=2, label='Training Loss')
    ax1.set_title(f'{title}: Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # F1 Plot
    ax2.plot(epochs, f1, marker='o', color=color_f1, linewidth=2, label='F1 Score')
    ax2.set_title(f'{title}: F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Make layout tight
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"✅ Saved {filename}")
    plt.close()

# --- Generate Files ---
plot_arch(old_epochs, old_loss, old_f1, 
          "Old Architecture (Concat CNN)", 
          "old_arch_metrics.png", 
          color_loss='firebrick', color_f1='salmon')

plot_arch(new_epochs, new_loss, new_f1, 
          "New Architecture (ResNeXt-50)", 
          "new_arch_metrics.png", 
          color_loss='darkgreen', color_f1='mediumseagreen')