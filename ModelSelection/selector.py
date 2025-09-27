from dataclasses import dataclass

@dataclass
class ModelChoice:
    name: str  # e.g., "linear"

class ModelSelector:
    """Selects among candidate model families (fixed to linear for this lab)."""
    def choose(self, model_type: str = "linear") -> ModelChoice:
        if model_type != "linear":
            raise NotImplementedError("Only 'linear' is implemented for this assignment.")
        return ModelChoice(name="linear")
