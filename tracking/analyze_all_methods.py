#!/usr/bin/env python
"""Analyze GOT-10k Val Results - Compare Baseline vs V8 (Final)"""
import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_gt(path):
    """Load GT file with flexible delimiter"""
    with open(path) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if '\t' in line:
            vals = [float(x) for x in line.split('\t')]
        elif ',' in line:
            vals = [float(x) for x in line.split(',')]
        else:
            vals = [float(x) for x in line.split()]
        data.append(vals)
    return np.array(data)

def load_results(result_dir, gt_dir):
    """Load tracking results"""
    results = {}
    for f in sorted(os.listdir(result_dir)):
        if f.startswith('GOT-10k_Val_') and f.endswith('.txt') and '_time' not in f and '_scores' not in f:
            seq = f.replace('.txt', '')
            gt_file = os.path.join(gt_dir, seq, 'groundtruth.txt')
            if os.path.exists(gt_file):
                try:
                    pred = load_gt(os.path.join(result_dir, f))
                    gt = load_gt(gt_file)
                    n = min(len(pred), len(gt))
                    results[seq] = {'pred': pred[:n], 'gt': gt[:n]}
                except Exception as e:
                    print(f"Error loading {seq}: {e}")
    return results

def iou(b1, b2):
    """Calculate IoU between two bboxes"""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[0]+b1[2], b2[0]+b2[2]), min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter/union if union > 0 else 0

def center_dist(b1, b2):
    """Calculate center distance between two bboxes"""
    return np.sqrt((b1[0]+b1[2]/2-b2[0]-b2[2]/2)**2 + (b1[1]+b1[3]/2-b2[1]-b2[3]/2)**2)

def evaluate(results):
    """Evaluate tracking results"""
    ious, ces = [], []
    for d in results.values():
        for i in range(len(d['pred'])):
            ious.append(iou(d['pred'][i], d['gt'][i]))
            ces.append(center_dist(d['pred'][i], d['gt'][i]))
    return np.array(ious), np.array(ces)

def compute_metrics(ious, ces):
    """Compute AUC and P@20"""
    auc = np.mean(ious)
    p20 = np.mean(ces <= 20)
    return auc, p20

def main():
    base = '/home/m1n1ons/projects/dev/DUTrack'
    gt_dir = os.path.join(base, 'data/got10k/val')
    
    # Define result directories - Only Baseline and V8 (Final)
    methods = {
        'Baseline': os.path.join(base, 'output/test/tracking_results/dutrack/dutrack_256_got_baseline/got10k'),
        'V8': os.path.join(base, 'output/test/tracking_results/dutrack_v8/dutrack_256_got_v8/got10k'),
    }
    
    print("Loading results...")
    results = {}
    for name, path in methods.items():
        if os.path.exists(path):
            results[name] = load_results(path, gt_dir)
            print(f"  {name}: {len(results[name])} sequences")
        else:
            print(f"  {name}: NOT FOUND at {path}")
    
    if len(results) < 2:
        print("Error: Need both Baseline and V8 results")
        return
    
    # Evaluate each method
    metrics = {}
    for name, data in results.items():
        ious, ces = evaluate(data)
        auc, p20 = compute_metrics(ious, ces)
        metrics[name] = {
            'ious': ious, 'ces': ces, 
            'AUC': auc, 'P@20': p20,
            'frames': len(ious)
        }
    
    # Print results
    print(f"\n{'='*70}")
    print("GOT-10k Validation Results - Baseline vs V8 (Final)")
    print(f"{'='*70}")
    
    base_auc = metrics.get('Baseline', {}).get('AUC', 0)
    base_p20 = metrics.get('Baseline', {}).get('P@20', 0)
    
    print(f"\n{'Method':<15} {'Frames':>10} {'AUC':>12} {'P@20':>12} {'Δ AUC':>12} {'Δ P@20':>12}")
    print(f"{'-'*73}")
    
    for name in ['Baseline', 'V8']:
        if name in metrics:
            m = metrics[name]
            d_auc = (m['AUC'] - base_auc) * 100 if name != 'Baseline' else 0
            d_p20 = (m['P@20'] - base_p20) * 100 if name != 'Baseline' else 0
            print(f"{name:<15} {m['frames']:>10} {m['AUC']*100:>11.2f}% {m['P@20']*100:>11.2f}% {d_auc:>+11.2f}% {d_p20:>+11.2f}%")
    
    print(f"{'='*70}")
    
    # V8 improvement summary
    if 'V8' in metrics and 'Baseline' in metrics:
        print(f"\n📊 V8 Improvement over Baseline:")
        d_auc = (metrics['V8']['AUC'] - metrics['Baseline']['AUC']) * 100
        d_p20 = (metrics['V8']['P@20'] - metrics['Baseline']['P@20']) * 100
        print(f"   AUC: {d_auc:+.2f}%")
        print(f"   P@20: {d_p20:+.2f}%")
        print(f"   ✅ V8 is the final optimized version!")
    
    # Plot comparison
    save_dir = os.path.join(base, 'output/got10k_analysis/all_methods')
    os.makedirs(save_dir, exist_ok=True)
    
    # Success Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    t = np.linspace(0, 1, 101)
    colors = {'Baseline': 'blue', 'V8': 'orange'}
    styles = {'Baseline': '--', 'V8': '-'}
    
    for name in ['Baseline', 'V8']:
        if name in metrics:
            m = metrics[name]
            success = [np.mean(m['ious'] >= x) for x in t]
            ax.plot(t, np.array(success)*100, 
                   color=colors[name], linestyle=styles[name], 
                   lw=2.5, label=f'{name} [AUC={m["AUC"]*100:.2f}%]')
    
    ax.set_xlabel('Overlap Threshold', fontsize=14)
    ax.set_ylabel('Success Rate [%]', fontsize=14)
    ax.set_title('GOT-10k Validation - Success Plot (Baseline vs V8)', fontsize=16)
    ax.legend(fontsize=12, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'success_plot.png'), dpi=150)
    print(f"\n✅ Success plot saved to: {save_dir}/success_plot.png")
    
    # Precision Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    t = np.linspace(0, 50, 101)
    
    for name in ['Baseline', 'V8']:
        if name in metrics:
            m = metrics[name]
            precision = [np.mean(m['ces'] <= x) for x in t]
            ax.plot(t, np.array(precision)*100,
                   color=colors[name], linestyle=styles[name],
                   lw=2.5, label=f'{name} [P@20={m["P@20"]*100:.2f}%]')
    
    ax.set_xlabel('Center Location Threshold [pixels]', fontsize=14)
    ax.set_ylabel('Precision [%]', fontsize=14)
    ax.set_title('GOT-10k Validation - Precision Plot (Baseline vs V8)', fontsize=16)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=20, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'precision_plot.png'), dpi=150)
    print(f"✅ Precision plot saved to: {save_dir}/precision_plot.png")
    
    # Save results to text file
    with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GOT-10k Validation Results - Baseline vs V8 (Final Version)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Method':<15} {'Frames':>10} {'AUC':>12} {'P@20':>12} {'Δ AUC':>12} {'Δ P@20':>12}\n")
        f.write("-" * 70 + "\n")
        for name in ['Baseline', 'V8']:
            if name in metrics:
                m = metrics[name]
                d_auc = (m['AUC'] - base_auc) * 100 if name != 'Baseline' else 0
                d_p20 = (m['P@20'] - base_p20) * 100 if name != 'Baseline' else 0
                f.write(f"{name:<15} {m['frames']:>10} {m['AUC']*100:>11.2f}% {m['P@20']*100:>11.2f}% {d_auc:>+11.2f}% {d_p20:>+11.2f}%\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("\nV8 Core Improvements:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Quality-Gated Template Storage (confidence >= 0.5)\n")
        f.write("2. Peak Sharpness Quality Assessment\n")
        f.write("3. Response Spatial Concentration Metric\n")
        f.write("4. Size-Matched Template Selection\n")
        f.write("\nV8 is the FINAL optimized version.\n")
    print(f"✅ Results saved to: {save_dir}/results.txt")

if __name__ == '__main__':
    main()
