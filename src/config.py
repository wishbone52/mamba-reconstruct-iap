from dataclasses import dataclass, field
from typing import Optional
import yaml

@dataclass
class DataConfig:
    data_loc: str

@dataclass
class TrainConfig:
    run_name: str
    epochs: int
    batch_size: int
    lr: float
    clip_value: float
    w_data: float
    w_physics: float
    w_distill: float
    distill: bool
    physics: bool
    use_cond: bool
    seed: int
    patience: int
    teacher_model_path: str
    device: str = "cuda" 
    
@dataclass
class ModelConfig:
    n_layers: int
    dim: int
    d_state: int
    expand: int
    seq_len: int

@dataclass
class Config:
    train: TrainConfig
    model: ModelConfig
    data: DataConfig

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    train_cfg = TrainConfig(**raw['train'])
    model_cfg = ModelConfig(**raw['model'])
    data_cfg = DataConfig(**raw['data'])
    return Config(train=train_cfg, model=model_cfg, data=data_cfg)