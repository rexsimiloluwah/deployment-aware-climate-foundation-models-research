from __future__ import annotations

import importlib
import importlib.util
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AuroraReadiness:
    model: str
    display_name: str
    package_import: str
    package_available: bool
    parameter_count_estimate: int | None
    checkpoint_size_estimate: str | None
    import_error: str | None
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_aurora_readiness(model_cfg: dict) -> AuroraReadiness:
    package_import = model_cfg.get("package_import", "aurora")
    package_available = importlib.util.find_spec(package_import) is not None
    import_error = None
    notes = "Aurora package import is available. Next step: construct model and load checkpoints."
    if package_available:
        try:
            importlib.import_module(package_import)
        except Exception as exc:
            package_available = False
            import_error = f"{type(exc).__name__}: {exc}"
            notes = "Aurora package was found but failed during import."
    else:
        notes = (
            "Aurora package is not importable in this environment. Install/provide the Aurora code and checkpoints "
            "before running real Aurora forecasts."
        )
    return AuroraReadiness(
        model=model_cfg["model"],
        display_name=model_cfg["display_name"],
        package_import=package_import,
        package_available=package_available,
        parameter_count_estimate=model_cfg.get("parameter_count_estimate"),
        checkpoint_size_estimate=model_cfg.get("checkpoint_size_estimate"),
        import_error=import_error,
        notes=notes,
    )


def count_parameters(model: Any) -> dict[str, int]:
    params = list(model.parameters())
    return {
        "total_params": int(sum(p.numel() for p in params)),
        "trainable_params": int(sum(p.numel() for p in params if getattr(p, "requires_grad", False))),
    }
