import yaml

def read_train_config(path: str) -> dict:
    """读取并解析训练配置，返回字典。"""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg.get('train', {})
    return {
        'data_dir': tcfg.get('data_dir'),
        'continue_model': tcfg.get('continue_model'),
        'batch_size': int(tcfg.get('batch_size')),
        'num_workers': int(tcfg.get('num_workers')),
        'epochs': int(tcfg.get('epochs')),
        'save_interval': int(tcfg.get('save_interval', 1)),
        'device': str(tcfg.get('device', 'cuda:0')),
    }

def main():
    config = read_train_config('train_config.yaml')
    device = torch.device(config['device'])

    
