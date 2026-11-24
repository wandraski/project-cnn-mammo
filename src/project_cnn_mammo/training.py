"""Treinamento PyTorch preguiçoso e determinístico.

Importar este módulo não exige torch; só `fit`/`build_classifier` exigem, e
falham com instrução clara quando o extra `ml` não está instalado.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def require_torch() -> tuple[Any, Any]:
    try:
        import torch  # type: ignore
        import torchvision  # type: ignore
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise RuntimeError(
            "instale o extra `ml` (torch/torchvision) para treinar; o núcleo stdlib segue útil"
        ) from exc
    return torch, torchvision


def set_determinism(seed: int, *, deterministic: bool = True) -> None:
    """Seed global + flags de determinismo; idempotente."""
    torch, _ = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_classifier(name: str, *, pretrained: bool = True) -> Any:
    """Instancia uma CNN do escopo; pesos ImageNet quando disponíveis."""
    from .models import classifier

    spec = classifier(name)
    torch, torchvision = require_torch()
    weights = "DEFAULT" if pretrained else None
    builders = {
        "resnet50": lambda: torchvision.models.resnet50(weights=weights),
        "efficientnet_b0": lambda: torchvision.models.efficientnet_b0(weights=weights),
        "efficientnet_b3": lambda: torchvision.models.efficientnet_b3(weights=weights),
        "convnext_tiny": lambda: torchvision.models.convnext_tiny(weights=weights),
    }
    model = builders[spec.name]()
    head_size = model.fc.in_features if hasattr(model, "fc") else model.classifier[-1].in_features
    if hasattr(model, "fc"):
        model.fc = torch.nn.Linear(head_size, 1)
    else:
        model.classifier[-1] = torch.nn.Linear(head_size, 1)
    return model


def build_segmenter(name: str, *, pretrained: bool = True) -> Any:
    """U-Net do zero ou com encoder ResNet34 (segmentation_models_pytorch)."""
    from .models import segmenter

    spec = segmenter(name)
    require_torch()
    try:
        import segmentation_models_pytorch as smp  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("instale segmentation-models-pytorch para a U-Net") from exc
    if spec.name == "unet":
        return smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=1)
    weights = "imagenet" if pretrained else None
    return smp.Unet(encoder_name="resnet34", encoder_weights=weights, classes=1)


def fit(
    model: Any,
    train_loader: Iterable[tuple[Any, Any]],
    val_loader: Iterable[tuple[Any, Any]],
    *,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    linear_probe: bool = False,
) -> dict[str, list[float]]:
    """Loop mínimo com validação, BCEWithLogits e registro por época."""
    torch, _ = require_torch()
    set_determinism(seed)
    model.to(device)
    if linear_probe:
        model.requires_grad_(False)
        head = model.fc if hasattr(model, "fc") else model.classifier[-1]
        head.requires_grad_(True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        model.train(not linear_probe)  # mantém BatchNorm do backbone congelado
        losses: list[float] = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history["train_loss"].append(sum(losses) / len(losses) if losses else 0.0)

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float().view(-1, 1)
                val_losses.append(float(criterion(model(images), labels).cpu()))
        history["val_loss"].append(sum(val_losses) / len(val_losses) if val_losses else 0.0)
    return history


def predict_proba(model: Any, loader: Iterable[Any], *, device: str = "cpu") -> list[float]:
    """Probabilidades de malignidade em ordem de loader."""
    torch, _ = require_torch()
    model.to(device).eval()
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            logits = model(images.to(device))
            probabilities.extend(torch.sigmoid(logits).reshape(-1).cpu().tolist())
    return probabilities
