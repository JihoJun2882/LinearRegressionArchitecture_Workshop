# Implemented a configuration-driven design where experiments can be reproduced by toggling model.type in config.yaml.
# In the previous codebase, model selection and training happened in a single step, so I couldn’t swap the model independently.

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
