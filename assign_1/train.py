import torch
import torch.nn as nn
import tempfile
import torch.nn.functional as F
import ray.train.torch
from ray.train import RunConfig
import os
from tqdm import tqdm
from dataset import (
    mnist_train_loader,
    mnist_test_loader,
    cifar100_test_loader,
    cifar100_train_loader,
    cifar10_test_loader,
    cifar10_train_loader
)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader_dict = {
    "mnist": (mnist_train_loader, mnist_test_loader),
    "cifar100": (cifar100_train_loader, cifar100_test_loader),
    "cifar10": (cifar10_train_loader, cifar10_test_loader)
}

def calculate_accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))

class MLP(nn.Module):
    def __init__(
        self, 
        n_layers, 
        hidden_size, 
        input_size, 
        output_size, 
        activation_fn: nn.Module,
        dropout: bool = False
    ):
        super().__init__()
        self.model_layers = []
        self.input_dim = input_size
        self.input = [nn.Linear(input_size, hidden_size)]
        self.n_layers = [nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            activation_fn
        ) for _ in range(n_layers)]
        self.output = [nn.Linear(hidden_size, output_size)]
        self.model_layers.extend(self.input)
        self.model_layers.extend(self.n_layers)
        self.drop = dropout

        if dropout:
            self.model_layers.extend([nn.Dropout(p=0.5)])

        self.model_layers.extend(self.output)
        self.model = nn.Sequential(*self.model_layers)
    
    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        out = self.model(x)
        out = F.softmax(out, dim=1)
        return out

    def training_step(self, x, y):
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        return loss

def train_func(config):
    lr = config['lr']
    dataset = config['dataset']
    num_epochs = config['num_epochs']
    act_fn = config['act_fn']
    in_size = config['in_size']
    hidden_size = config['hidden_size']
    out_size = config['out_size']
    optimizer_select = config['optimizer_select']
    dropout = config['dropout']

    act_fn_dict = {
        "leaky_relu": nn.LeakyReLU(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid()
    }

    model = MLP(
        n_layers=2,
        hidden_size=hidden_size,
        input_size=in_size,
        output_size=out_size,
        activation_fn=act_fn_dict[act_fn],
        dropout=dropout
    )

    model = ray.train.torch.prepare_model(model)

    if optimizer_select == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_select == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    optimizer = ray.train.torch.prepare_optimizer(optimizer=optimizer)

    train_loader, test_loader = dataloader_dict[dataset]
    train_loader, test_loader = ray.train.torch.prepare_data_loader(train_loader), ray.train.torch.prepare_data_loader(test_loader)

    for epoch in tqdm(
        range(num_epochs), 
        leave=False,
        position=0, 
        desc="EPOCHS"
    ):
        model.train()
        train_acc_total = 0
        for x, y in tqdm(
            train_loader, 
            desc="Training",
            leave=False, 
            position=0
        ):
            # x = x.to(device)
            # y = y.to(device)
            out = model.forward(x)
            train_loss = F.cross_entropy(out, y)
            train_loss.backward()
            train_acc = calculate_accuracy(out, y)
            optimizer.step()
            optimizer.zero_grad()
            train_acc_total += train_acc.item()

        test_acc_total = 0
        model.eval()
        for x, y in tqdm(
            test_loader,
            desc="Testing",
            leave=False, 
            position=0
        ):
            # x = x.to(device)
            # y = y.to(device)
            out = model.forward(x)
            test_acc = calculate_accuracy(out, y)
            test_acc_total += test_acc.item()
        
        # print(
        #     f"Epoch {epoch} | Test Accuracy {(test_acc_total/len(test_loader))*100} | Train Accuracy {(train_acc_total/len(train_loader))*100}"
        # )

        metrics = {
            "train_acc": (train_acc_total/len(train_loader))*100,
            "epoch": epoch,
            "test_acc": (test_acc_total/len(test_loader))*100, 
            "dataset": f"{dataset}", 
            "act_fn": act_fn, 
            "optimizer": optimizer_select
        }
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.module.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pt")
            )
            ray.train.report(
                metrics,
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
        if ray.train.get_context().get_world_rank() == 0:
            print(metrics)

scaling_config = ray.train.ScalingConfig(num_workers=4, use_gpu=False)
config = {
    "lr": 1e-4, 
    "num_epochs": 10,
    "dataset": "cifar100",
    "act_fn": "relu",
    "in_size": 3*32*32,
    "out_size": 100,
    "hidden_size": 128,
    "optimizer_select": "adam",
    "dropout": True
}
run_config = RunConfig(storage_path="./results", name="MNIST")

trainer = ray.train.torch.TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    train_loop_config=config
)
result = trainer.fit()