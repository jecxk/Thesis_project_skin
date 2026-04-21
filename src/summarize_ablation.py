import os
import json
import pandas as pd

def main():
    print("==========================================================")
    print("  Ablation Study Summary")
    print("==========================================================")
    
    # Define baseline
    baseline_path = "outputs/efficientnet_b0_v3(main_best)/experiment_summary.json"
    if not os.path.exists(baseline_path):
        print(f"Baseline file not found at {baseline_path}")
        return
        
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)
        
    base_acc = baseline_data['test_results']['accuracy']
    base_f1 = baseline_data['test_results']['macro_f1']
    base_auc = baseline_data['test_results']['auc_macro']
    base_kappa = baseline_data['test_results']['cohen_kappa']
    
    results = []
    results.append({
        'Configuration': 'Full Pipeline (Baseline)',
        'Accuracy': base_acc,
        'Macro F1': base_f1,
        'AUC': base_auc,
        'Kappa': base_kappa,
        'Delta F1': 0.0
    })
    
    # Define ablations to check
    ablations = [
        ('No Mixup/CutMix', 'outputs/ablation_no_mixup/experiment_summary.json'),
        ('No Weighted Sampler', 'outputs/ablation_no_sampler/experiment_summary.json'),
        ('No Advanced Augmentation', 'outputs/ablation_no_adv_aug/experiment_summary.json'),
        ('No Label Smoothing', 'outputs/ablation_no_smooth/experiment_summary.json'),
    ]
    
    for name, path in ablations:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            acc = data['test_results']['accuracy']
            f1 = data['test_results']['macro_f1']
            auc = data['test_results']['auc_macro']
            kappa = data['test_results']['cohen_kappa']
            delta_f1 = f1 - base_f1
            
            results.append({
                'Configuration': name,
                'Accuracy': acc,
                'Macro F1': f1,
                'AUC': auc,
                'Kappa': kappa,
                'Delta F1': delta_f1
            })
        else:
            results.append({
                'Configuration': name,
                'Accuracy': None,
                'Macro F1': None,
                'AUC': None,
                'Kappa': None,
                'Delta F1': None
            })
            
    df = pd.DataFrame(results)
    
    # Format the dataframe for pretty printing
    formatted_df = df.copy()
    for col in ['Accuracy', 'Macro F1', 'AUC', 'Kappa']:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "Running...")
    
    formatted_df['Delta F1'] = formatted_df['Delta F1'].apply(lambda x: f"{x:+.4f}" if pd.notnull(x) and x != 0.0 else ("-" if x == 0.0 else ""))
    
    print(formatted_df.to_string(index=False))
    print("==========================================================")
    
    # Save to CSV
    os.makedirs('outputs/ablations_summary', exist_ok=True)
    df.to_csv('outputs/ablations_summary/ablation_results.csv', index=False)
    print("Saved to outputs/ablations_summary/ablation_results.csv")

if __name__ == "__main__":
    main()
