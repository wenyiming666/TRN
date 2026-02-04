import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, SubsetRandomSampler
from Model.model import TRN
from Model.utils import Standardizer, confusion_matrix_plot
from Model.graphs import GraphData, collate_graph_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_recall_curve, auc
from torch.utils.tensorboard import SummaryWriter
import optuna
from torch.cuda import amp
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class Config:
    def __init__(self):
        self.max_atoms = 200
        self.node_vec_len = 80
        BASE_DIR = Path(__file__).resolve().parent
        self.train_data_path = BASE_DIR / "data" / "Train_ROBIN_and_decoys_remaining_data.csv"
        self.test_data_path = BASE_DIR / "data" / "Decoys_Robin_test_set.csv"
        self.n_epochs = 100
        self.patience = 10
        self.optuna_trials = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
config = Config()

class StableFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        inputs = inputs.float() if inputs.dtype == torch.half else inputs
        targets = targets.float() if targets.dtype == torch.half else targets
        
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss).clamp(self.eps, 1-self.eps)
        loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return loss.mean()


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.standardizer = None

    def load_datasets(self):
        train_set = GraphData(dataset_path=config.train_data_path, 
                            max_atoms=config.max_atoms, 
                            node_vec_len=config.node_vec_len)
        test_set = GraphData(dataset_path=config.test_data_path,
                           max_atoms=config.max_atoms,
                           node_vec_len=config.node_vec_len)
        return self._filter_invalid_samples(train_set), self._filter_invalid_samples(test_set)

    def _filter_invalid_samples(self, dataset):
        valid_indices = []
        for i in range(len(dataset)):
            data = dataset[i]
            if not any(torch.isnan(tensor).any() for tensor in data if isinstance(tensor, torch.Tensor)):
                valid_indices.append(i)
        return torch.utils.data.Subset(dataset, valid_indices)

    def create_loaders(self, train_set, test_set, batch_size):
        train_node_mats = []
        for i in range(len(train_set)):
            (node_mat, adj_mat), output, smile = train_set[i]
            train_node_mats.append(node_mat)
        train_node_mats = torch.cat(train_node_mats)
        self.standardizer = Standardizer(train_node_mats)
        train_labels = [train_set[i][1][0].item() for i in range(len(train_set))]  
        if sum(train_labels) == 0:  
            pos_weight = torch.tensor([1.0])
        else:
            pos_weight = torch.tensor([(len(train_labels)-sum(train_labels))/sum(train_labels)])
        
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=self._get_balanced_sampler(train_labels),
            collate_fn=collate_graph_dataset
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            collate_fn=collate_graph_dataset
        )
        return train_loader, test_loader, pos_weight

    def _get_balanced_sampler(self, labels):
        class_counts = np.bincount([int(label) for label in labels])
        if len(class_counts) < 2: 
            class_counts = np.array([1, 1]) if class_counts[0] > 0 else np.array([1, 1])
        weights = 1. / torch.tensor([class_counts[int(label)] for label in labels], dtype=torch.float)
        return torch.utils.data.WeightedRandomSampler(weights, len(weights))

class Trainer:
    def __init__(self, config, model, optimizer, loss_fn, scheduler, writer, data_pipeline):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.writer = writer
        self.data_pipeline = data_pipeline
        self.scaler = amp.GradScaler(enabled=config.device.type == 'cuda')  
        self.best_metrics = {'f1': 0, 'auroc': 0, 'auprc': 0}
        
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        for batch_data in train_loader:
            (node_mats, adj_mats), labels, smiles = batch_data
            node_mats, adj_mats, labels = self._prepare_batch(batch_data)
            
            self.optimizer.zero_grad()
            
            with amp.autocast(enabled=self.config.device.type == 'cuda'):
                outputs = self.model(node_mats, adj_mats)  
                loss = self.loss_fn(outputs, labels.float())  
            
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy()) 
            all_labels.extend(labels.cpu().numpy())
        
        return total_loss / len(train_loader), np.array(all_preds), np.array(all_labels)

    
    def evaluate(self, test_loader, epoch):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_data in test_loader:
                (node_mats, adj_mats), labels, smiles = batch_data
                
                node_mats, adj_mats, labels = self._prepare_batch(batch_data)
                
                with amp.autocast(enabled=self.config.device.type == 'cuda'):
                    outputs = self.model(node_mats, adj_mats).squeeze() 
                    labels = labels.squeeze() 
                    loss = self.loss_fn(outputs, labels.float())
                
                total_loss += loss.item()
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return total_loss / len(test_loader), np.array(all_preds), np.array(all_labels)
    
   
    def _prepare_batch(self, batch_data, labels=None, smiles=None):
        
        if isinstance(batch_data, tuple) and len(batch_data) == 3:
            (node_mats, adj_mats), labels, smiles = batch_data
        
        node_mats = self.data_pipeline.standardizer.standardize(node_mats)
        
        batch_size = labels.size(0)
        node_mats = node_mats.view(batch_size, self.config.max_atoms, self.config.node_vec_len)
        adj_mats = adj_mats.view(batch_size, self.config.max_atoms, self.config.max_atoms)
        device = self.config.device
        return node_mats.to(device), adj_mats.to(device), labels.to(device)
    
    def _compute_metrics(self, true, pred):
        if len(np.unique(true)) == 1: 
            return {
                'loss': 0,
                'acc': 0,
                'f1': 0,
                'auroc': 0.5,
                'auprc': 0,
                'recall': 0,
                'threshold': 0.5
            }
        
        precision, recall, thresholds = precision_recall_curve(true, pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_thresh = thresholds[np.argmax(f1_scores)]
        pred_class = (pred > best_thresh).astype(int)
        
        
        with torch.no_grad():
            loss = self.loss_fn(
                torch.tensor(pred, dtype=torch.float32),
                torch.tensor(true, dtype=torch.float32)
            ).item()
        
        return {
            'loss': loss,
            'acc': np.mean(pred_class == true),
            'f1': f1_score(true, pred_class),
            'auroc': roc_auc_score(true, pred),
            'auprc': auc(recall, precision),
            'recall': recall_score(true, pred_class),
            'threshold': best_thresh
        }
    
    def run(self, train_loader, test_loader):
        history = {'train': [], 'test': []}
        no_improve = 0
        
        for epoch in range(config.n_epochs):
            train_loss, train_preds, train_labels = self.train_epoch(train_loader, epoch)
            test_loss, test_preds, test_labels = self.evaluate(test_loader, epoch)
            
            train_metrics = self._compute_metrics(train_labels, train_preds)
            test_metrics = self._compute_metrics(test_labels, test_preds)
            
            history['train'].append(train_metrics)
            history['test'].append(test_metrics)
            
            self.scheduler.step(test_metrics['f1'])
            
            if test_metrics['f1'] > self.best_metrics['f1']:
                self.best_metrics = test_metrics
                no_improve = 0
                torch.save(self.model.state_dict(), self.config.save_dir / 'best_model.pt')
                np.save(self.config.save_dir / 'best_threshold.npy', np.array(test_metrics['threshold']))
            else:
                no_improve += 1
                if no_improve >= config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            self._log_metrics(epoch, train_metrics, test_metrics)
            
        return history
    
    def _log_metrics(self, epoch, train_metrics, test_metrics):
        log_msg = f"Epoch {epoch+1}/{config.n_epochs} | "
        log_msg += f"Train Loss: {train_metrics['loss']:.4f} | "
        log_msg += f"Test Loss: {test_metrics['loss']:.4f}\n"
        log_msg += f"Train F1: {train_metrics['f1']:.4f} | "
        log_msg += f"Test F1: {test_metrics['f1']:.4f} | "
        log_msg += f"Train Acc: {train_metrics['acc']:.4f} | "  
        log_msg += f"Test Acc: {test_metrics['acc']:.4f} | "    
        log_msg += f"Train Recall: {train_metrics['recall']:.4f} | "  
        log_msg += f"Test Recall: {test_metrics['recall']:.4f} | "   
        log_msg += f"Best Thresh: {test_metrics['threshold']:.3f}"
        print(log_msg)
        
        for metric in ['loss', 'f1', 'auroc', 'auprc', 'acc', 'recall']:
            self.writer.add_scalars(metric, {
                'train': train_metrics[metric],
                'test': test_metrics[metric]
            }, epoch)

def objective(trial):
    params = {
        'hidden_nodes': trial.suggest_int('hidden_nodes', 64, 256),
        'n_conv_layers': trial.suggest_int('n_conv_layers', 2, 4),
        'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.3, 0.6),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2),
        'gamma': trial.suggest_float('gamma', 1.0, 3.0)
    }
    
    data_pipeline = DataPipeline(config)
    train_set, test_set = data_pipeline.load_datasets()
    train_loader, test_loader, pos_weight = data_pipeline.create_loaders(
        train_set, test_set, params['batch_size'])
    
    model = TRN(
        node_vec_len=config.node_vec_len,
        node_fea_len=params['hidden_nodes'],
        hidden_fea_len=params['hidden_nodes'],
        n_conv=params['n_conv_layers'],
        n_hidden=params['n_hidden_layers'],
        n_outputs=1,
        p_dropout=params['dropout_rate']
    ).to(config.device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    loss_fn = StableFocalLoss(alpha=pos_weight.item(), gamma=params['gamma'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )
    
    trainer = Trainer(config, model, optimizer, loss_fn, scheduler, writer, data_pipeline)
    history = trainer.run(train_loader, test_loader)
    
    best_f1 = max([m['f1'] for m in history['test']])
    best_loss = min([m['loss'] for m in history['test']])
    return 0.7 * best_f1 - 0.3 * best_loss

def plot_results(history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ['loss', 'f1', 'auroc', 'auprc']
    
    for ax, metric in zip(axes.flatten(), metrics):
        ax.plot([m[metric] for m in history['train']], label='Train')
        ax.plot([m[metric] for m in history['test']], label='Test')
        ax.set_title(metric.upper())
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(config.save_dir / 'training_curves.png')
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    writer = SummaryWriter()
    
    # ---------------------- running Optuna ----------------------
    # study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    # study.optimize(objective, n_trials=config.optuna_trials)
    # print("Best trial:")
    # trial = study.best_trial
    # print(f"  Value: {trial.value}")
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print(f"    {key}: {value}")
    # best_params = trial.params

    # ---------------------- Use the optimal parameters directly ----------------------
    best_params = {
        'hidden_nodes': 250,         
        'n_conv_layers': 4,          
        'n_hidden_layers': 2,        
        'learning_rate': 0.00039057, 
        'batch_size': 256,           
        'dropout_rate': 0.302544,    
        'weight_decay': 0.00974258,  
        'gamma': 2.9817595           
    }

    data_pipeline = DataPipeline(config)
    train_set, test_set = data_pipeline.load_datasets()
    
    
    train_loader, test_loader, pos_weight = data_pipeline.create_loaders(
        train_set, test_set, batch_size=best_params['batch_size']
    )
    
    
    final_model = TRN(
        node_vec_len=config.node_vec_len,
        node_fea_len=best_params['hidden_nodes'],
        hidden_fea_len=best_params['hidden_nodes'],
        n_conv=best_params['n_conv_layers'],        
        n_hidden=best_params['n_hidden_layers'],    
        n_outputs=1,
        p_dropout=best_params['dropout_rate']
    ).to(config.device)
    
    
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    
    loss_fn = StableFocalLoss(
        alpha=pos_weight.item(),  
        gamma=best_params['gamma']
    )
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5
    )
    
    print("\n=== Final Training with Best Parameters ===")
    trainer = Trainer(
        config=config,
        model=final_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        writer=writer,
        data_pipeline=data_pipeline
    )
    history = trainer.run(train_loader, test_loader)
    
    torch.save(final_model.state_dict(), config.save_dir / 'final_model_best.pt')
    final_threshold = trainer.best_metrics['threshold']
    np.save(config.save_dir / 'final_threshold.npy', np.array(final_threshold))
    print(f"Best threshold: {final_threshold:.4f}")
    data_pipeline.standardizer.save(config.save_dir / 'standardizer.pth')
    print("Standardizer parameters have been saved.")
    writer.close()
    plot_results(history)