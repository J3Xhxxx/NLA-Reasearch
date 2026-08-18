import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_detection_curve(json_path, out_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    curve = data.get('summary', {}).get('detection_curve', {})
    if not curve:
        print("No detection curve found in", json_path)
        return
        
    alphas = []
    mean_abs_cos = []
    frac_gt_05 = []
    
    for alpha_str in sorted(curve.keys(), key=float):
        alphas.append(float(alpha_str))
        metrics = curve[alpha_str]
        mean_abs_cos.append(metrics['mean_abs_cos_sig'])
        frac_gt_05.append(metrics['frac_gt_0.5'])
        
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Alpha (Injection Strength)')
    ax1.set_ylabel('Mean Abs Cosine Similarity', color=color)
    ax1.plot(alphas, mean_abs_cos, marker='o', color=color, label='Mean Abs Cos')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Fraction > 0.5 Cosine', color=color)  
    ax2.plot(alphas, frac_gt_05, marker='x', linestyle='--', color=color, label='Frac > 0.5')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.1, 1.1)

    fig.tight_layout()  
    plt.title('Injection Pilot Detection Curve')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    json_path = 'injection_pilot.json'
    out_path = 'detection_curve.png'
    plot_detection_curve(json_path, out_path)
