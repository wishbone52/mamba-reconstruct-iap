import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import ApDataset
from data.utils import *
from model.mamba import Mamba_model
from plotters import *
import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class PIAPMambaTrainer():
    """
    Unified trainer class for Mamba model on intracellular action potential prediction task with optional distillation and physics constraints.
    """

    def __init__(self, data_loc, args):
        """
        Initialize trainer: load data, build model, set up optimizer and teacher (if distillation).
        
        Args:
            data_loc: Path to the pickle file containing raw data.
            args: Namespace with train/model configs.
        """

        self.train_loader = None
        self.test_dataloaders = []
        self.args = args
        self.best_val_loss = float('inf')
        self.patience = self.args.train.patience          
        self.patience_counter = 0
        self.best_model_path = os.path.join("models", self.args.train.run_name, "best_model.pt")

        # Dataset
        data = prepare_raw_data(data_loc)
        
        intras_train, extras_train = data_prep_test(data,trains,raw = False,max_samples=10000,limit_keys=trains_with_limits)
        apd_train =  get_all_apds_multiprocessing(intras_train.reshape(-1, 8000))
        full_train_dataset = ApDataset(extras_train,intras_train,apd_train)
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        generator = torch.Generator().manual_seed(self.args.train.seed)
        train_subset, val_subset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size], generator=generator
        )
        self.train_loader = DataLoader(train_subset, batch_size=args.train.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.val_loader = DataLoader(val_subset, batch_size=args.train.batch_size, shuffle=False, num_workers=4)

        for test_number, test_data_name in enumerate(tests):  
            extras_test = data[(test_data_name, 'extra')]
            intras_test = data[(test_data_name, 'intra')]
            apd_test = get_all_apds_multiprocessing((data[(test_data_name, 'intra')]).reshape(-1, 8000))

            test_dataset = ApDataset(extras_test, intras_test, apd_test)
            test_dataloader = DataLoader(test_dataset, batch_size=args.train.batch_size, shuffle=False, num_workers=4)
            self.test_dataloaders.append(test_dataloader)

        # model
        self.model = Mamba_model(
            seq_len=args.model.seq_len, 
            n_layers=args.model.n_layers, 
            dim=args.model.dim, 
            d_state=args.model.d_state, 
            expand=args.model.expand,
            use_cond=args.train.use_cond, 
            physics=args.train.physics ).to(args.train.device)

        # if distilled:
        if args.train.distill and args.train.teacher_model_path is not None:
            self.teacher_model = Mamba_model(
                seq_len=8000, 
                n_layers=4, 
                dim=512, 
                d_state=32, 
                expand=2,
                use_cond=False, 
                physics=False).to(args.train.device)
            checkpoint = torch.load(args.train.teacher_model_path, map_location=self.args.train.device, weights_only=True)
            self.teacher_model.load_state_dict(checkpoint, strict=True)

            for param in self.teacher_model.parameters():
                param.requires_grad = False
        
        # training setting
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name.lower() or "bias" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.001},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=args.train.lr
        )
        self.clip_value = args.train.clip_value
        self.mae = nn.L1Loss()
    
    def train(self):
        """
        Main training loop with validation, early stopping, and optional distillation.
        """

        setup_logging(self.args.train.run_name)
        logger = SummaryWriter(os.path.join("runs",self.args.train.run_name))
        l = len(self.train_loader)

        for epoch in range(1,self.args.train.epochs+1):
            self.model.train()
            self.teacher_model.eval() if self.args.train.distill else None
            logging.info(f"Starting epoch {epoch}/{self.args.train.epochs}:")

            pbar = tqdm(self.train_loader)
            epoch_loss = 0.0
            device = self.args.train.device
            for i, (extras, intras, _) in enumerate(pbar):
                extras = extras.to(device).float()    
                intras = intras.to(device).float()

                pred_intras, pred_dv_out = self.model(extras)

                if self.args.train.distill:
                    teacher_pred_intras, _ = self.teacher_model(extras)
        
                intras_loss = self.mae(pred_intras,intras)
                
                loss = intras_loss
                if self.args.train.physics:
                    dv_out_groud_truth = torch.zeros(pred_intras.shape[0], 7700, device=pred_intras.device)
                    physics_loss = self.mae(pred_dv_out,dv_out_groud_truth)
                    loss = ( intras_loss * self.args.w_data + physics_loss * self.args.w_physics ) / (self.args.w_data + self.args.w_physics)
                
                if self.args.train.distill:
                    distill_loss = self.mae(pred_intras, teacher_pred_intras.detach())
                    loss = self.args.train.w_distill * loss + (1-self.args.train.w_distill) * distill_loss

                self.optimizer.zero_grad()
                loss.backward()

                if self.clip_value > 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.clip_value
                    )
                    logger.add_scalar("Gradient_Norm", total_norm, global_step=epoch * l + i)

                self.optimizer.step()
                
                logger.add_scalar("MAE_batch", loss.item(), global_step=epoch * l + i)
                epoch_loss += loss.item()
            
            epoch_avg_loss = epoch_loss / len(self.train_loader)
            logger.add_scalar("MAE_epoch", epoch_avg_loss, global_step=epoch)

            # ---------- Validation ----------
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for extras, intras, _ in self.val_loader:
                    extras = extras.to(self.args.train.device).float()
                    intras = intras.to(self.args.train.device).float()
                    pred_intras, _ = self.model(extras)
                    loss = self.mae(pred_intras, intras)   
                    val_loss += loss.item() * extras.size(0)
            val_loss /= len(self.val_loader.dataset)

            logger.add_scalar("Val_Loss", val_loss, global_step=epoch)

            # ---------- Early Stopping & Model Saving ----------
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
    
                torch.save(self.model.state_dict(), self.best_model_path)
                logging.info(f"New best model saved with val_loss={val_loss:.4f}")
            else:
                self.patience_counter += 1
                logging.info(f"Validation loss did not improve, patience {self.patience_counter}/{self.patience}")
                if self.patience_counter >= self.patience:
                    logging.info(f"Early stopping triggered after epoch {epoch}")
                    break          
    
    def evaluate(self, model_ckpt_path=None):
        """
        Evaluate model on all test datasets.
        
        Args:
            model_ckpt_path: Optional path to a checkpoint; if None, use current model.
        """
        
        if model_ckpt_path:
            print(f"Starting test for {model_ckpt_path}:")
            model = Mamba_model(
                seq_len=self.args.model.seq_len, 
                n_layers=self.args.model.n_layers, 
                dim=self.args.model.dim,
                d_state=self.args.model.d_state, 
                expand=self.args.model.expand,
                use_cond=self.args.train.use_cond, 
                physics=self.args.train.physics).to(self.args.train.device)
            checkpoint = torch.load(model_ckpt_path, map_location=self.args.train.device, weights_only=True)
            model.load_state_dict(checkpoint, strict=True)
        else:
            model = self.model
        
        model.eval()

        intras_list_all = []
        preds_list_all = []
        preds_apd_list_all = []
        actual_apds_list = []

        for test_number, test_data_name in enumerate(tests): 
            
            test_loader = self.test_dataloaders[test_number]
                    
            intras_list = []
            apds_list = []
            preds_list = []
            preds_apd_list = []

            pbar = tqdm(test_loader,desc=f"Evaluating [{test_data_name}]",total=len(test_loader))
            for extras_t, intras_t, apd_t in pbar:     
                extras_t = extras_t.to(self.args.train.device).float()  
                intras_t = intras_t.to(self.args.train.device).float() 
                
                with torch.no_grad():
                    pred_intras, _ = model(extras_t)

                    pred_intras = pred_intras.reshape(-1,8000).cpu().numpy()
                    gt_intras = intras_t.reshape(-1,8000).cpu().numpy()

                pred_intras_new, intras_t_new = autro_correct(pred_intras, gt_intras)

                pred_intras_apd = get_all_apds_multiprocessing(pred_intras_new)
                        
                intras_list.append(intras_t_new)
                apds_list.append(apd_t)
                preds_list.append(pred_intras_new)
                preds_apd_list.append(pred_intras_apd)
                    
            intras_list_ = np.vstack(intras_list)
            apds_list_ = np.vstack(apds_list)
            preds_list_ = np.vstack(preds_list)
            preds_apd_list_ = np.vstack(preds_apd_list)

            intras_list_all.append(intras_list_)
            actual_apds_list.append(apds_list_)
            preds_list_all.append(preds_list_)
            preds_apd_list_all.append(preds_apd_list_)

            sum_error_test = np.mean(np.abs(preds_list_ - intras_list_),axis=1)

            print(f"Test dataset {test_data_name} finished.")
            print(f"MAE:{sum_error_test.mean():.4f}, MAE-STD:{sum_error_test.std():.4f}, MAE-MAX:{sum_error_test.max():.4f}")