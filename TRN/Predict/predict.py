import json
import torch
from torch.utils.data import Dataset, DataLoader
from Model.model import TRN
from Model.utils import Standardizer
from Model.graphs import Graph, collate_graph_dataset_predict
import numpy as np
import pandas as pd
import argparse
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")


class Config:
    def __init__(self):
        self.max_atoms = 200
        self.node_vec_len = 80
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()



def load_model_components(model_path, threshold_path, standardizer_path):
    model_params = {
        "node_vec_len": config.node_vec_len,
        "node_fea_len": 250,
        "hidden_fea_len": 250,
        "n_conv": 4,
        "n_hidden": 2,
        "n_outputs": 1,
        "p_dropout": 0.302544,
    }

    model = TRN(**model_params).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    best_threshold = np.load(threshold_path).item()
    standardizer = Standardizer.load(standardizer_path)

    return model, best_threshold, standardizer



class GraphDataPredictStreaming(Dataset):
    """
    Dataset for streaming prediction: build graph on-the-fly.
    """
    def __init__(self, jsonl_path, node_vec_len, max_atoms):
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.smiles_list = []
        self.ids = []

        with open(jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                self.smiles_list.append(data["smiles"])
                self.ids.append(data.get("id", ""))

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        graph = Graph(smi, self.node_vec_len, self.max_atoms)
        node_mat = torch.tensor(graph.node_mat, dtype=torch.float32)
        adj_mat = torch.tensor(graph.adj_mat, dtype=torch.float32)
        return (node_mat, adj_mat), smi



def predict(model, data_loader, standardizer, threshold, log_every=500):
    all_probs = []
    all_preds = []
    all_smiles = []

    total = len(data_loader.dataset)
    processed = 0

    with torch.no_grad():
        for batch_data in data_loader:
            (node_mats, adj_mats), smiles = batch_data

            node_mats = standardizer.standardize(node_mats)
            node_mats = node_mats.view(len(smiles), config.max_atoms, config.node_vec_len)
            adj_mats = adj_mats.view(len(smiles), config.max_atoms, config.max_atoms)

            node_mats = node_mats.to(config.device)
            adj_mats = adj_mats.to(config.device)

            outputs = model(node_mats, adj_mats)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > threshold).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_smiles.extend(smiles)

            processed += len(smiles)
            if processed % log_every <= len(smiles):
                print(f"Processed {processed}/{total} ({processed/total*100:.2f}%) ")

    return all_smiles, all_probs, all_preds


def save_results(smiles, ids, probs, preds, output_path):
    df = pd.DataFrame(
        {
            "id": ids,
            "smiles": smiles,
            "predicted_probability": probs,
            "predicted_label": preds,
        }
    )
    df.to_csv(output_path, index=False)
    print(f"The prediction results have been saved to: {output_path}")



def main():
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR / "Model"
    parser = argparse.ArgumentParser(description="TRN Large-Scale Prediction Script")
    parser.add_argument("--jsonl_path", type=str, required=True, help="JSONL File path")
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_DIR / 'final_model_best.pt',
    )
    parser.add_argument(
        "--threshold_path",
        type=str,
        default=MODEL_DIR / 'best_threshold.npy',
    )
    parser.add_argument(
        "--standardizer_path",
        type=str,
        default=MODEL_DIR / 'standardizer.pth',
    )
    parser.add_argument("--output_path", type=str, default="prediction_results.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--log_every",
        type=int,
        default=50000,
        help="Print progress every X molecules processed",
    )

    args = parser.parse_args()

    global standardizer
    model, best_threshold, standardizer = load_model_components(
        args.model_path, args.threshold_path, args.standardizer_path
    )
    
    dataset = GraphDataPredictStreaming(
        args.jsonl_path, node_vec_len=config.node_vec_len, max_atoms=config.max_atoms
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_graph_dataset_predict,
    )


    all_smiles, all_probs, all_preds = predict(
        model, data_loader, standardizer, best_threshold, log_every=args.log_every
    )


    save_results(all_smiles, dataset.ids, all_probs, all_preds, args.output_path)
    print("Done")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
