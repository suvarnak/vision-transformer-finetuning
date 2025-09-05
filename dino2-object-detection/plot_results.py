import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_performance_charts():
    # Load results
    with open('validation_results.json', 'r') as f:
        results = json.load(f)
    
    # Extract data
    models = [r['model_name'] for r in results]
    metrics = {
        'Accuracy': [r['accuracy'] for r in results],
        'Precision': [r['precision'] for r in results],
        'Recall': [r['recall'] for r in results],
        'F1-Score': [r['f1_score'] for r in results],
        'Mean IoU': [r['mean_iou'] for r in results],
        'mAP@0.5': [r['map_50'] for r in results]
    }
    
    # Set style
    plt.style.use('seaborn-v0_8')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Bar chart comparison
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(models))
    width = 0.12
    
    for i, (metric, values) in enumerate(metrics.items()):
        ax1.bar(x + i*width, values, width, label=metric, color=colors[i%3], alpha=0.8)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('All Metrics Comparison')
    ax1.set_xticks(x + width*2.5)
    ax1.set_xticklabels([m.replace(' Detector', '') for m in models], rotation=45)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy comparison
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(models, metrics['Accuracy'], color=colors[:len(models)], alpha=0.8)
    ax2.set_title('Classification Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics['Accuracy']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. IoU vs mAP scatter plot
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(metrics['Mean IoU'], metrics['mAP@0.5'], 
                         c=colors[:len(models)], s=200, alpha=0.8, edgecolors='black')
    
    for i, model in enumerate(models):
        ax3.annotate(model.replace(' Detector', ''), 
                    (metrics['Mean IoU'][i], metrics['mAP@0.5'][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Mean IoU')
    ax3.set_ylabel('mAP@0.5')
    ax3.set_title('IoU vs mAP Performance')
    ax3.grid(True, alpha=0.3)
    
    # 4. Radar chart
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    
    # Normalize metrics to 0-1 scale for radar chart
    normalized_metrics = {}
    for metric, values in metrics.items():
        max_val = max(values) if max(values) > 0 else 1
        normalized_metrics[metric] = [v/max_val for v in values]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, model in enumerate(models):
        values = [normalized_metrics[metric][i] for metric in metrics.keys()]
        values += values[:1]  # Complete the circle
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax4.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics.keys())
    ax4.set_ylim(0, 1)
    ax4.set_title('Normalized Performance Radar')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 5. Precision vs Recall
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(metrics['Recall'], metrics['Precision'], 
                         c=colors[:len(models)], s=200, alpha=0.8, edgecolors='black')
    
    for i, model in enumerate(models):
        ax5.annotate(model.replace(' Detector', ''), 
                    (metrics['Recall'][i], metrics['Precision'][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title('Precision vs Recall')
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance improvement
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate improvement over simple DINO
    simple_idx = 0  # Simple DINO is first
    improvements = {}
    
    for metric in ['Accuracy', 'Mean IoU', 'mAP@0.5']:
        baseline = metrics[metric][simple_idx]
        improvements[metric] = [
            ((v - baseline) / baseline * 100) if baseline > 0 else 0 
            for v in metrics[metric]
        ]
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, (metric, values) in enumerate(improvements.items()):
        ax6.bar(x + i*width, values, width, label=metric, alpha=0.8)
    
    ax6.set_xlabel('Models')
    ax6.set_ylabel('Improvement over Simple DINO (%)')
    ax6.set_title('Performance Improvement')
    ax6.set_xticks(x + width)
    ax6.set_xticklabels([m.replace(' Detector', '') for m in models], rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('model_comparison_charts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary table
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Dynamic table header based on number of models
    header = f"{'Metric':<15}"
    for model in models:
        header += f" {model.replace(' Detector', ''):<15}"
    print(header)
    print("-" * (15 + 16 * len(models)))
    
    for metric in metrics.keys():
        values = metrics[metric]
        row = f"{metric:<15}"
        for value in values:
            row += f" {value:<15.4f}"
        print(row)
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    best_accuracy = max(metrics['Accuracy'])
    best_iou = max(metrics['Mean IoU'])
    best_map = max(metrics['mAP@0.5'])
    
    acc_winner = models[metrics['Accuracy'].index(best_accuracy)]
    iou_winner = models[metrics['Mean IoU'].index(best_iou)]
    map_winner = models[metrics['mAP@0.5'].index(best_map)]
    
    print(f"🏆 Best Accuracy: {acc_winner} ({best_accuracy:.3f})")
    print(f"🏆 Best IoU: {iou_winner} ({best_iou:.3f})")
    print(f"🏆 Best mAP@0.5: {map_winner} ({best_map:.3f})")
    
    # Calculate improvements
    simple_acc = metrics['Accuracy'][0]
    improved_acc = metrics['Accuracy'][1]
    improvement = ((improved_acc - simple_acc) / simple_acc) * 100
    
    print(f"\n📈 Improved DINO vs Simple DINO:")
    print(f"   • Accuracy improvement: +{improvement:.1f}%")
    print(f"   • IoU improvement: +{((metrics['Mean IoU'][1] - metrics['Mean IoU'][0]) / metrics['Mean IoU'][0]) * 100:.1f}%")
    print(f"   • mAP improvement: +{((metrics['mAP@0.5'][1] - metrics['mAP@0.5'][0]) / metrics['mAP@0.5'][0]) * 100:.1f}%")

if __name__ == "__main__":
    create_performance_charts()