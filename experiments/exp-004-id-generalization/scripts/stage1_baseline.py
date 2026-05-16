"""
Exp-004 Stage 1: ID 与泛化的相关性 Baseline
不同 weight decay 训练 ResNet-18 on CIFAR-10，测最后一层 TwoNN ID vs test accuracy

对应: experiments/exp-004-id-generalization/plan.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

EXP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHT_DECAYS = [0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]


def parse_weight_decays(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def build_run_grid(weight_decays: list[float], n_seeds: int) -> list[tuple[float, int]]:
    return [(wd, seed) for wd in weight_decays for seed in range(n_seeds)]


def select_shard(grid: list[tuple[float, int]], shard_idx: int, n_shards: int) -> list[tuple[float, int]]:
    if n_shards < 1:
        raise ValueError("--n-shards must be >= 1")
    if shard_idx < 0 or shard_idx >= n_shards:
        raise ValueError("--shard-idx must satisfy 0 <= shard_idx < n_shards")
    return [job for i, job in enumerate(grid) if i % n_shards == shard_idx]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def twonn_id(X: np.ndarray) -> float:
    """TwoNN intrinsic dimension estimator (Facco et al. 2017)."""
    from scipy.spatial.distance import cdist
    N = len(X)
    dists = cdist(X, X)
    np.fill_diagonal(dists, np.inf)

    # For each point, get distances to 1st and 2nd nearest neighbor
    sorted_dists = np.sort(dists, axis=1)
    r1 = sorted_dists[:, 0]
    r2 = sorted_dists[:, 1]

    # Avoid division by zero
    mask = r1 > 1e-10
    mu = r2[mask] / r1[mask]

    # MLE estimator
    n = len(mu)
    mu_sorted = np.sort(mu)
    # Empirical CDF
    F_emp = np.arange(1, n + 1) / n

    # Fit: log(1 - F) = -(d-1) * log(mu)
    # Use median estimator for robustness
    d_hat = np.mean(np.log(n / np.arange(n, 0, -1)) / np.log(mu_sorted))

    return float(d_hat)


def get_cifar10(batch_size=128, data_root=None, num_workers=2):
    """Load CIFAR-10 with standard augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_root = Path(data_root) if data_root is not None else EXP_DIR / "data"
    train_ds = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32x32 input)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(weights=None, num_classes=num_classes)
        # Adapt for 32x32: smaller initial conv, no maxpool
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)

    def features(self, x):
        """Extract penultimate features (before FC)."""
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def train_and_evaluate(weight_decay, seed, n_epochs=100, device="mps", data_root=None, batch_size=128, num_workers=2):
    """Train ResNet-18 and return test accuracy + penultimate ID."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ResNet18CIFAR().to(device)
    train_loader, test_loader = get_cifar10(batch_size=batch_size, data_root=data_root, num_workers=num_workers)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Train
    model.train()
    for epoch in range(n_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate test accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    test_acc = correct / total

    # Extract penultimate features for ID estimation
    features_list = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            feat = model.features(x)
            features_list.append(feat.cpu().numpy())
            if sum(f.shape[0] for f in features_list) >= 4096:
                break

    features = np.concatenate(features_list)[:4096]
    id_estimate = twonn_id(features)

    # Param norm
    param_norm = sum(p.data.norm().item()**2 for p in model.parameters())**0.5

    # Stable rank (of last layer weight)
    W = model.model.fc.weight.data.cpu()
    s = torch.linalg.svdvals(W)
    stable_rank = (s.sum()**2 / (s**2).sum()).item()

    return {
        "weight_decay": weight_decay,
        "seed": seed,
        "test_acc": test_acc,
        "twonn_id": id_estimate,
        "param_norm": param_norm,
        "stable_rank": stable_rank,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--weight-decays", default=",".join(str(v) for v in DEFAULT_WEIGHT_DECAYS))
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", default="/nvmessd/lifanhong/LR-env/data/cifar10")
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    weight_decays = parse_weight_decays(args.weight_decays)
    run_grid = build_run_grid(weight_decays, args.n_seeds)
    jobs = select_shard(run_grid, args.shard_idx, args.n_shards)
    print(f"Shard {args.shard_idx}/{args.n_shards}: {len(jobs)} runs")

    results = []
    for wd, seed in jobs:
        print(f"\n{'='*50}")
        print(f"Weight decay={wd}, seed={seed}")
        print(f"{'='*50}")
        r = train_and_evaluate(
            wd,
            seed,
            n_epochs=args.n_epochs,
            device=device,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"  Test acc: {r['test_acc']:.4f}")
        print(f"  TwoNN ID: {r['twonn_id']:.1f}")
        print(f"  Param norm: {r['param_norm']:.1f}")
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'WD':>8} {'Acc':>8} {'ID':>8} {'‖θ‖':>8} {'srank':>8}")
    print("-" * 42)

    # Average over seeds
    for wd in sorted(set(r["weight_decay"] for r in results)):
        wd_results = [r for r in results if r["weight_decay"] == wd]
        acc_mean = np.mean([r["test_acc"] for r in wd_results])
        id_mean = np.mean([r["twonn_id"] for r in wd_results])
        norm_mean = np.mean([r["param_norm"] for r in wd_results])
        srank_mean = np.mean([r["stable_rank"] for r in wd_results])
        print(f"{wd:>8.0e} {acc_mean:>8.4f} {id_mean:>8.1f} {norm_mean:>8.1f} {srank_mean:>8.1f}")

    # Correlation
    accs = [r["test_acc"] for r in results]
    ids = [r["twonn_id"] for r in results]
    from scipy.stats import pearsonr
    r, p = pearsonr(ids, accs)
    print(f"\nPearson r(ID, acc) = {r:.3f}, p = {p:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(ids, accs, c=[np.log10(r["weight_decay"]+1e-10) for r in results], cmap="viridis")
    axes[0].set_xlabel("TwoNN ID (penultimate layer)")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title(f"ID vs Accuracy (r={r:.3f}, p={p:.4f})")

    wd_log = [np.log10(r["weight_decay"]+1e-10) for r in results]
    axes[1].scatter(wd_log, ids)
    axes[1].set_xlabel("log10(weight_decay)")
    axes[1].set_ylabel("TwoNN ID")
    axes[1].set_title("Weight Decay → ID")

    axes[2].scatter(wd_log, accs)
    axes[2].set_xlabel("log10(weight_decay)")
    axes[2].set_ylabel("Test Accuracy")
    axes[2].set_title("Weight Decay → Accuracy")

    plt.tight_layout()
    fig_dir = EXP_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "stage1_correlation.png", dpi=150)
    print(f"\nFigure saved to {fig_dir / 'stage1_correlation.png'}")

    # Save
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    with open(EXP_DIR / f"results_stage1{suffix}.json", "w") as f:
        json.dump({"results": results, "pearson_r": r, "pearson_p": p}, f, indent=2)
    print(f"Results saved to {EXP_DIR / f'results_stage1{suffix}.json'}")


if __name__ == "__main__":
    main()
