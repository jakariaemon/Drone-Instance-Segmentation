import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

file_paths = {
    'Total Loss': 'total_loss.csv',
    'Mask Loss': 'mask_loss.csv',
    'Loss Class': 'loss_class.csv',
    'Loss Box Reg': 'loss_box_reg.csv'
}

loss_patterns = {
    'Total Loss': 'output/total_loss',
    'Mask Loss': 'output/loss_mask',
    'Loss Class': 'output/loss_cls',
    'Loss Box Reg': 'output/loss_box_reg'
}

fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
axs = axs.flatten() 
subplot_labels = ['(a)', '(b)', '(c)', '(d)']  
plt.subplots_adjust(top=0.9)  
fig.suptitle('Model Loss Metrics Over Training Steps', fontsize=10, y=0.98) 
for i, (loss_type, file_path) in enumerate(file_paths.items()):
    data = pd.read_csv(file_path) 
    for col in data.columns:
        if loss_patterns[loss_type] in col and '__MAX' not in col and '__MIN' not in col:
            axs[i].plot(data['Step'], data[col], label=col.split(' - ')[0])
    axs[i].set_xlabel('Step')
    axs[i].set_ylabel(loss_type, fontsize=10)
    axs[i].set_title(f'{subplot_labels[i]} {loss_type} Over Steps', fontsize=10, loc='left')  
    axs[i].legend()
    axs[i].grid(True)
plt.show()
