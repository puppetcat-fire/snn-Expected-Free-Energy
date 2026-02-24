import matplotlib.pyplot as plt
import numpy as np

methods = ['Baseline', 'FA (ours)', 'Exact EFE']
success = [3.33, 8.33, 5.00]
times = [0.19, 0.15, 35.50]
colors = ['#7f7f7f', '#2ca02c', '#d62728']

fig, ax1 = plt.subplots(figsize=(9, 5))
x = np.arange(len(methods))

bars = ax1.bar(x, success, color=colors, alpha=0.85, width=0.6)
ax1.set_ylabel('Delayed Reward Success Rate (%)', fontsize=12, color='#2ca02c')
ax1.set_ylim(0, 10)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=11)

ax2 = ax1.twinx()
ax2.plot(x, times, 'o--', color='#d62728', linewidth=2.5, markersize=8, label='Time per sequence (s)')
ax2.set_ylabel('Time per sequence (s)', fontsize=12, color='#d62728')
ax2.set_yscale('log')

for i, v in enumerate(success):
    ax1.text(i, v + 0.3, f'{v}%', ha='center', fontweight='bold')

plt.title('FA vs Exact EFE on 124M GPT-2 (4-8 step delayed rewards)', fontsize=14, pad=20)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('FA_vs_Exact_EFE_124M.png', dpi=300, bbox_inches='tight')
plt.show()