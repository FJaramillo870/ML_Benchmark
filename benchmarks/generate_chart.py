import matplotlib.pyplot as plt
import numpy as np


def generate_hardware_chart():
    # 1. The Raw Data
    models = ['Anomaly (MLP)', 'Vision (CNN)', 'Forecaster (LSTM)']
    cpu_latency = [0.0428, 0.0788, 0.0516]
    gpu_latency = [0.2735, 0.2518, 0.4442]
    edge_latency = [0.4071, 1.8672, 0.5144]

    # 2. Chart Layout Configuration
    x = np.arange(len(models))  # Label locations
    width = 0.25  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Plotting the Bars (With distinct engineering colors)
    rects1 = ax.bar(x - width, cpu_latency, width, label='Desktop CPU', color='#4C72B0')
    rects2 = ax.bar(x, gpu_latency, width, label='Desktop GPU', color='#DD8452')
    rects3 = ax.bar(x + width, edge_latency, width, label='Edge ARM (Raspberry Pi)', color='#55A868')

    # 4. Labels and Formatting
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Inference Latency Comparison Across Hardware Platforms\n(Batch Size = 1)', fontsize=14,
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)

    # Add gridlines behind the bars for readability
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # 5. Auto-attach the exact numbers to the top of each bar
    ax.bar_label(rects1, padding=3, fmt='%.4f', fontsize=9)
    ax.bar_label(rects2, padding=3, fmt='%.4f', fontsize=9)
    ax.bar_label(rects3, padding=3, fmt='%.4f', fontsize=9)

    fig.tight_layout()

    # 6. Save and Display
    output_filename = 'hardware_comparison_chart.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"STATUS: [SUCCESS] Chart saved as {output_filename}")

    # Opens the chart in a window so you can see it immediately
    plt.show()


if __name__ == "__main__":
    generate_hardware_chart()