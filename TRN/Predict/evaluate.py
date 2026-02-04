import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from model import TRN
from utils import Standardizer
from graphs import GraphData, collate_graph_dataset
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, 
                             precision_recall_curve, auc, accuracy_score)
import argparse
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

class Config:
    def __init__(self):
        self.max_atoms = 200
        self.node_vec_len = 80
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
config = Config()


def load_model_components(model_path, threshold_path, standardizer_path):
    
    model_params = {
        'node_vec_len': config.node_vec_len,
        'node_fea_len': 250,
        'hidden_fea_len': 250,
        'n_conv': 4,
        'n_hidden': 2,
        'n_outputs': 1,
        'p_dropout': 0.302544
    }
    
    
    model = TRN(**model_params).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()  
    
    best_threshold = np.load(threshold_path).item()
    standardizer = Standardizer.load(standardizer_path)
    
    return model, best_threshold, standardizer

def process_external_data(data_path):
    data = pd.read_csv(data_path)
    
    required_cols = ['smiles', 'label']
    if not set(required_cols).issubset(data.columns):
        raise ValueError(f"Input data must include columns.: {required_cols}")
    
    dataset = GraphData(
        dataset_path=Path(data_path),
        max_atoms=config.max_atoms,
        node_vec_len=config.node_vec_len
    )
    
    valid_indices = []
    for i in range(len(dataset)):
        try:
            data = dataset[i]
            if not any(torch.isnan(tensor).any() for tensor in data[0] if isinstance(tensor, torch.Tensor)):
                valid_indices.append(i)
        except:
            continue
    
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
    data_loader = DataLoader(
        valid_dataset,
        batch_size=128,
        collate_fn=collate_graph_dataset
    )
    
    valid_labels = []
    valid_smiles = []
    for idx in valid_indices:
        (_, _), label, smile = dataset[idx]
        valid_labels.append(label.item())
        valid_smiles.append(smile)
    
    
    return data_loader, np.array(valid_labels), valid_smiles, valid_indices


def prepare_batch(batch_data, standardizer):
    (node_mats, adj_mats), labels, smiles = batch_data
    
    node_mats = standardizer.standardize(node_mats)
    
    batch_size = labels.size(0)
    node_mats = node_mats.view(batch_size, config.max_atoms, config.node_vec_len)
    adj_mats = adj_mats.view(batch_size, config.max_atoms, config.max_atoms)
    
    return (node_mats.to(config.device), adj_mats.to(config.device), 
            labels.to(config.device), smiles)


def evaluate(model, data_loader, true_labels, valid_smiles, valid_indices, threshold):
    all_preds = []
    all_probs = []
    
    with torch.no_grad(): 
        for batch_data in data_loader:
            node_mats, adj_mats, labels, smiles = prepare_batch(batch_data, standardizer)
            
            
            outputs = model(node_mats, adj_mats)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten() 
            preds = (probs > threshold).astype(int)  
            
            all_probs.extend(probs)
            all_preds.extend(preds)
    
    metrics = {}
    metrics['ACC'] = accuracy_score(true_labels, all_preds)
    metrics['Recall'] = recall_score(true_labels, all_preds)
    metrics['F1'] = f1_score(true_labels, all_preds)
    
    if len(np.unique(true_labels)) >= 2:
        metrics['AUROC'] = roc_auc_score(true_labels, all_probs)
        
        precision, recall, _ = precision_recall_curve(true_labels, all_probs)
        metrics['AUPRC'] = auc(recall, precision)
    else:
        metrics['AUROC'] = "N/A"
        metrics['AUPRC'] = "N/A"
    
    result_df = pd.DataFrame({
        'original_index': valid_indices,
        'smiles': valid_smiles,
        'true_label': true_labels,
        'predicted_probability': all_probs,
        'predicted_label': all_preds
    })
    
    return result_df, metrics

def save_results(result_df, metrics, output_path):
    result_df.to_csv(output_path, index=False)
    print(f"\nDetailed results have been saved to: {output_path}")
    
    metrics_path = output_path.replace('.csv', '_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("===== Summary of Model Evaluation Metrics =====\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"Evaluation metrics have been saved to: {metrics_path}")

def main():
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR / "model"
    parser = argparse.ArgumentParser(description='Evaluating the TRN model on labeled external test sets')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the labeled external test set CSV file')
    parser.add_argument('--model_path', type=str, 
                      default=MODEL_DIR / 'final_model_best.pt', 
                      help='Model .pt file path')
    parser.add_argument('--threshold_path', type=str, 
                      default=MODEL_DIR / 'best_threshold.npy', 
                      help='Optimal Threshold.npy File Path')
    parser.add_argument('--standardizer_path', type=str, 
                      default=MODEL_DIR / 'standardizer.pth', 
                      help='Path to the Standardizer Parameters .pth File')
    parser.add_argument('--output_path', type=str, 
                      default=MODEL_DIR / 'TRN_evaluation_results.csv', 
                      help='Prediction Output Path')
    
    args = parser.parse_args()
    
    
    global standardizer 
    model, best_threshold, standardizer = load_model_components(
        args.model_path, args.threshold_path, args.standardizer_path
    )
    
    
    
    data_loader, true_labels, valid_smiles, valid_indices = process_external_data(args.data_path)
    
    if not data_loader.dataset:
        print("No valid samples available for evaluation; program exits.")
        return
    
    
    result_df, metrics = evaluate(
        model, data_loader, true_labels, valid_smiles, valid_indices, best_threshold
    )
    

    print("\n===== Key Performance Indicators =====")
    print(f"ACC: {metrics['ACC']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1: {metrics['F1']:.4f}")
    print(f"AUROC: {metrics['AUROC']:.4f}" if isinstance(metrics['AUROC'], float) else f"AUROC: {metrics['AUROC']}")
    print(f"AUPRC: {metrics['AUPRC']:.4f}" if isinstance(metrics['AUPRC'], float) else f"AUPRC: {metrics['AUPRC']}")
    
    save_results(result_df, metrics, args.output_path)

if __name__ == "__main__":
    torch.manual_seed(42)
    main()