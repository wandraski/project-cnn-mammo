"""Registry de arquiteturas do escopo (CNNs + U-Net baseline).

O factory devolve especificação, não instância: o treinamento com PyTorch vive
no extra `ml`, e o registry permanece testável sem GPU.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Architecture:
    name: str
    family: str
    input_size: int
    pretrained_options: tuple[str, ...]
    heads: tuple[str, ...] = ("classification",)


_CNN_REGISTRY: dict[str, Architecture] = {
    "resnet50": Architecture("resnet50", "cnn", 224, ("imagenet", "radimagenet")),
    "efficientnet_b0": Architecture("efficientnet_b0", "cnn", 224, ("imagenet",)),
    "efficientnet_b3": Architecture("efficientnet_b3", "cnn", 300, ("imagenet",)),
    "convnext_tiny": Architecture("convnext_tiny", "cnn", 224, ("imagenet",)),
}

_SEGMENTER_REGISTRY: dict[str, Architecture] = {
    "unet": Architecture("unet", "segmenter", 512, (), ("segmentation",)),
    "unet_resnet34": Architecture(
        "unet_resnet34", "segmenter", 512, ("imagenet",), ("segmentation",)
    ),
}


def classifier(name: str) -> Architecture:
    """Arquitetura de classificação; recusa nomes fora do escopo CNN."""
    try:
        return _CNN_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"arquitetura fora do escopo: {name}; opções: {sorted(_CNN_REGISTRY)}"
        ) from exc


def segmenter(name: str) -> Architecture:
    try:
        return _SEGMENTER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"segmentador inválido: {name}; opções: {sorted(_SEGMENTER_REGISTRY)}"
        ) from exc


def architectures(family: str | None = None) -> tuple[Architecture, ...]:
    """Lista registro opcional por família (`cnn`/`segmenter`)."""
    items: Iterable[Architecture] = (
        *_CNN_REGISTRY.values(),
        *_SEGMENTER_REGISTRY.values(),
    )
    if family is None:
        return tuple(items)
    return tuple(item for item in items if item.family == family)


def supports_pretraining(name: str, option: str) -> bool:
    return option in classifier(name).pretrained_options
