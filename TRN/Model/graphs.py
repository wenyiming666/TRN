import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch.utils.data import Dataset

def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        mol_frags = Chem.GetMolFrags(mol, asMols=True)
        largest_frag = max(mol_frags, default=None, key=lambda x: x.GetNumAtoms())
        
        if largest_frag is None:
            return None
            
        canonical_smiles = Chem.MolToSmiles(largest_frag, isomericSmiles=True)
        return canonical_smiles
        
    except Exception as e:
        print(f"SMILES standardize failed: {smiles}, error: {e}")
        return None

class Graph:
    def __init__(self, molecule_smiles: str, node_vec_len: int, max_atoms: int = None):
        self.smiles = molecule_smiles
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        
        self.mol = Chem.MolFromSmiles(molecule_smiles)
        if self.mol is not None:
            self.mol = Chem.AddHs(self.mol)
            self.smiles_to_graph()
        else:
            print(f"Unable to resolve SMILES: {molecule_smiles}")

    def smiles_to_graph(self):
        # Get list of atoms in molecule
        atoms = self.mol.GetAtoms()

        # Create empty node matrix
        if self.max_atoms is None:
            n_atoms = len(list(atoms))
        else:
            n_atoms = self.max_atoms
        node_mat = np.zeros((n_atoms, self.node_vec_len))

        # Iterate over atoms and add to node matrix
        for atom in atoms:
            # Get atom index and atomic number
            atom_index = atom.GetIdx()
            atom_no = atom.GetAtomicNum()

            # Assign to node matrix
            node_mat[atom_index, atom_no] = 1

        # Create empty adjacency matrix
        adj_mat = np.zeros((n_atoms, n_atoms))

        # Create adjacency matrix
        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol)
        self.std_adj_mat = np.copy(adj_mat)

        # Pad the adjacency matrix
        dim_add = n_atoms - adj_mat.shape[0]
        adj_mat = np.pad(
            adj_mat, pad_width=((0, dim_add), (0, dim_add)), mode="constant"
        )

        # Add an identity matrix to adjacency matrix
        # This will make an atom its own neighbor
        adj_mat = adj_mat + np.eye(n_atoms)

        # Save both matrices
        self.node_mat = node_mat
        self.adj_mat = adj_mat


class GraphData(Dataset):
    def __init__(self, dataset_path: str, node_vec_len: int, max_atoms: int):
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        df = pd.read_csv(dataset_path)
        self.smiles = []
        self.outputs = []
        
        for _, row in df.iterrows():
            smi = row["smiles"]
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                self.smiles.append(smi)
                self.outputs.append(row["label"])
        self.outputs = np.array(self.outputs, dtype=np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        mol_graph = Graph(smile, self.node_vec_len, self.max_atoms)
        node_mat = torch.tensor(mol_graph.node_mat, dtype=torch.float32)
        adj_mat = torch.tensor(mol_graph.adj_mat, dtype=torch.float32)
        output = torch.tensor([self.outputs[idx]], dtype=torch.float32)
        
        return (node_mat, adj_mat), output, smile 

    def get_atom_no_sum(self, i):
        # Get smile
        smile = self.smiles[i]

        # Create MolGraph object
        mol = Graph(smile, self.node_vec_len, self.max_atoms)

        # Get matrices
        node_mat = mol.node_mat

        # Get atomic number sum
        one_pos_mat = np.argwhere(node_mat == 1)
        atomic_no_sum = one_pos_mat[:, -1].sum()
        return atomic_no_sum


def collate_graph_dataset(dataset: Dataset):
    # Create empty lists of node and adjacency matrices, outputs, and smiles
    node_mats = []
    adj_mats = []
    outputs = []
    smiles = []

    # Iterate over list and assign each component to the correct list
    for i in range(len(dataset)):
        (node_mat, adj_mat), output, smile = dataset[i]
        node_mats.append(node_mat)
        adj_mats.append(adj_mat)
        outputs.append(output)
        smiles.append(smile)

    # Create tensors
    node_mats_tensor = torch.cat(node_mats, dim=0)
    adj_mats_tensor = torch.cat(adj_mats, dim=0)
    outputs_tensor = torch.stack(outputs, dim=0)

    # Return tensors
    return (node_mats_tensor, adj_mats_tensor), outputs_tensor, smiles

def collate_graph_dataset_predict(dataset: Dataset):
    node_mats = []
    adj_mats = []
    smiles = []

    for i in range(len(dataset)):
        # prediction dataset returns only two elements
        (node_mat, adj_mat), smile = dataset[i]
        node_mats.append(node_mat)
        adj_mats.append(adj_mat)
        smiles.append(smile)

    node_mats_tensor = torch.cat(node_mats, dim=0)
    adj_mats_tensor = torch.cat(adj_mats, dim=0)

    return (node_mats_tensor, adj_mats_tensor), smiles

class GraphDataPredict(Dataset):
    def __init__(self, smiles_list: list, node_vec_len: int, max_atoms: int):
        self.smiles = smiles_list
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms
        self.node_mats = []
        self.adj_mats = []

        for smi in smiles_list:
            mol_graph = Graph(smi, node_vec_len, max_atoms)
            self.node_mats.append(torch.tensor(mol_graph.node_mat, dtype=torch.float32))
            self.adj_mats.append(torch.tensor(mol_graph.adj_mat, dtype=torch.float32))

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
       
        return (self.node_mats[idx], self.adj_mats[idx]), self.smiles[idx]

