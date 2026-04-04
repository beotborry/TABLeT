import torch.optim as optim

def get_optimizer(config, args, model):
    # Set up the optimizer
    optimizer_dict = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
    }
    optimizer_cls = optimizer_dict[config.train.optim.name]
    optimizer = optimizer_cls(model.parameters(), 
                              lr=args.lr,
                              weight_decay=args.weight_decay)
    
    return optimizer