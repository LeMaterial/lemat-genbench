"""Model registry for easy access to different MLIPs.

This is a refactored version that eliminates redundancy and standardizes
model handling using a configuration-driven approach.

This implementation assumes a consistent naming convention for model packages
and their contents:

- The model package must have an ``INFO`` dictionary with the following keys:
  - ``name``: str
  - ``description``: str
  - ``default_model``: str
  - ``supports_embeddings``: bool
  - ``supports_relaxation``: bool

- The model package must have a calculator class called ``{name}Calculator``
  - e.g. ``MACECalculator``
- The model package must have a factory function called ``create_{name.lower()}_calculator``
  - e.g. ``create_mace_calculator``
- The model package must have an attribute called ``AVAILABLE_{name.upper()}_MODELS``
  - e.g. ``AVAILABLE_MACE_MODELS``
- The model package _may_ have an attribute called ``AVAILABLE_{name.upper()}_TASKS``
  - e.g. ``AVAILABLE_MACE_TASKS``

Available models are automatically discovered by looking for directories with
``__init__.py`` in the ``models/`` package.
"""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, Type

from lematerial_forgebench.models.base import BaseMLIPCalculator
from lematerial_forgebench.utils.logging import logger


@dataclass
class ModelConfig:
    """Data class to describe a model's configuration."""

    description: str
    name: str
    supports_embeddings: bool
    supports_relaxation: bool

    available_models: Optional[list[str]] = None
    available_tasks: Optional[list[str]] = None
    calculator_class: str = ""
    default_model: Optional[str] = None
    default_task: Optional[str] = None
    factory_function: str = ""
    is_available: bool = False
    module_path: str = ""

    def __post_init__(self):
        self.module_path = (
            f"lematerial_forgebench.models.{self.name.lower()}.calculator"
        )

        try:
            self.module = import_module(self.module_path)
        except ImportError as e:
            logger.debug(f"Module {self.module_path} unavailable: {e}")
            return
        self.is_available = True

        self.calculator_class = f"{self.name}Calculator"
        if not hasattr(self.module, self.calculator_class):
            raise ImportError(
                f"Calculator class {self.calculator_class} not found in {self.module_path}"
            )

        self.factory_function = f"create_{self.name.lower()}_calculator"
        if not hasattr(self.module, self.factory_function):
            raise ImportError(
                f"Factory function {self.factory_function} not found in {self.module_path}"
            )

        self.available_models = f"AVAILABLE_{self.name.upper()}_MODELS"
        if not hasattr(self.module, self.available_models):
            raise ImportError(
                f"Available models attribute {self.available_models} not found in {self.module_path}"
            )

        self.available_tasks_attr = f"AVAILABLE_{self.name.upper()}_TASKS"
        if not hasattr(self.module, self.available_tasks_attr):
            self.available_tasks_attr = None


class ModelRegistry:
    """Registry for managing available MLIP calculators.

    This version uses a configuration-driven approach to eliminate redundancy
    and standardize model handling.

    Models are discovered by looking for directories with ``__init__.py`` in the
    ``models/`` package.
    """

    def __init__(self):
        self._calculators: Dict[str, Type[BaseMLIPCalculator]] = {}
        self._factory_functions: Dict[str, callable] = {}
        self._model_info: Dict[str, Dict] = {}
        self._model_availability: Dict[str, bool] = {}
        self._model_configs: Dict[str, ModelConfig] = {}
        self._discover_and_register_models()

    def _discover_and_register_models(self):
        """Automatically discover and register available models.

        Models are discovered by looking for directories with ``__init__.py`` in
        the ``models/`` package.

        If a model's calculator module can be imported, its calculator class and
        factory function are registered and a success message is printed.

        Otherwise, a warning message is printed and the model is skipped.
        """

        # Discover models in the models package by looking for directories with __init__.py
        candidates = [
            p.name
            for p in Path(__file__).parent.iterdir()
            if p.is_dir() and (p / "__init__.py").exists()
        ]
        for model_name in candidates:
            self._model_availability[model_name] = False
            try:
                # Import the model package
                package = import_module(f"lematerial_forgebench.models.{model_name}")
                if not hasattr(package, "INFO"):
                    logger.warning(f"Info for {model_name} not found")
                    continue

                # Create a model config from the INFO dictionary
                config = ModelConfig(**package.INFO)
                self._model_configs[model_name] = config

                if not config.is_available:
                    logger.warning(f"{model_name} not available")
                    continue

                # Get calculator class and factory function
                calculator_class = getattr(package, config.calculator_class)
                factory_function = getattr(package, config.factory_function)

                # Register the model
                self._calculators[model_name] = calculator_class
                self._factory_functions[model_name] = factory_function

                # Build model info
                self._model_info[model_name] = self._build_model_info(config, package)

                self._model_availability[model_name] = True
                logger.info(f"Successfully registered {model_name} model")

            except ImportError as e:
                logger.warning(f"{model_name} not available: {e}")
            except AttributeError as e:
                logger.warning(f"Missing required components for {model_name}: {e}")

    def _build_model_info(self, config: ModelConfig, package) -> Dict[str, Any]:
        """Build model information dictionary."""
        info = {
            "class": config.calculator_class,
            "description": config.description,
            "supports_embeddings": config.supports_embeddings,
            "supports_relaxation": config.supports_relaxation,
            "default_model": config.default_model,
            "default_task": config.default_task,
            "is_available": config.is_available,
            "available_models": [],
            "available_tasks": [],
        }

        # Add available models if attribute exists
        if config.available_models and hasattr(package, config.available_models):
            info["available_models"] = getattr(package, config.available_models)

        # Add available tasks if attribute exists
        if config.available_tasks_attr and hasattr(
            package, config.available_tasks_attr
        ):
            info["available_tasks"] = getattr(package, config.available_tasks_attr)

        return info

    def get_available_models(self) -> list[str]:
        """Get list of available model types.

        Returns
        -------
        list[str]
            List of available model names
        """
        return list(self._calculators.keys())

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available.

        Parameters
        ----------
        model_name : str
            Name of the model to check

        Returns
        -------
        bool
            True if model is available, False otherwise
        """
        return self._model_availability.get(model_name, False)

    def create_calculator(self, model_name: str, **kwargs) -> BaseMLIPCalculator:
        """Create a calculator for the specified model type.

        You can see which models are available in your current version of ``forgebench``
        using :func:`~lematerial_forgebench.models.registry.get_available_models`

        Parameters
        ----------
        model_name : str
            Name of the model (e.g., "orb", "mace", "equiformer", "uma").
            Use :func:`~lematerial_forgebench.models.registry.get_available_models`
            to see which models are available.
        **kwargs
            Model-specific arguments

        Returns
        -------
        BaseMLIPCalculator
            Configured calculator instance

        Raises
        ------
        ValueError
            If model_name is not available
        """
        if model_name not in self._factory_functions:
            available = self.get_available_models()
            raise ValueError(
                f"Model type '{model_name}' not available. "
                f"Available models: {available}"
            )

        factory_func = self._factory_functions[model_name]
        return factory_func(**kwargs)

    def get_model_info(self) -> Dict[str, Dict]:
        """Get information about available models.

        Returns
        -------
        Dict[str, Dict]
            Information about each available model
        """
        return {
            name: info
            for name, info in self._model_info.items()
            if self.is_model_available(name)
        }

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        Optional[ModelConfig]
            Model configuration if available
        """
        return self._model_configs.get(model_name)


# Global registry instance
_registry = ModelRegistry()


def get_calculator(model_name: str, **kwargs) -> BaseMLIPCalculator:
    """Get a calculator for the specified model type.

    This is the main entry point for creating calculators.

    Parameters
    ----------
    model_name : str
        Name of the model to create
    **kwargs
        Model-specific configuration

    Returns
    -------
    BaseMLIPCalculator
        Configured calculator

    Examples
    --------
    >>> calc = get_calculator("orb", model_type="orb_v3_conservative_inf_omat")
    >>> calc = get_calculator("mace", model_type="mp")
    >>> calc = get_calculator("uma", model_name="uma-s-1", task="omat")
    """
    return _registry.create_calculator(model_name, **kwargs)


def list_available_models() -> list[str]:
    """List all available model types.

    Returns
    -------
    list[str]
        Available model types
    """
    return _registry.get_available_models()


def get_model_info() -> Dict[str, Dict]:
    """Get information about all available models.

    Returns
    -------
    Dict[str, Dict]
        Model information
    """
    return _registry.get_model_info()


def print_model_info():
    """Print information about available models."""
    from rich import print
    from rich.console import Group
    from rich.table import Table

    info = get_model_info()
    tables = []
    max_width = 0

    for model_type, details in info.items():
        if not max_width:
            max_width = max(len(key) for key in details.keys())

        table = Table(
            title=f"\n[bold blue]{model_type.upper()}[/bold blue]",
            show_header=False,
            title_justify="left",
        )
        table.add_column(width=max_width, justify="left")
        table.add_column(justify="left")

        for key, value in details.items():
            key = " ".join(word.capitalize() for word in key.split("_")).strip()
            table.add_row(key, str(value))

        tables.append(table)

    print(Group(*tables))


def get_equiformer_calculator(**kwargs) -> BaseMLIPCalculator:
    """Get an Equiformer calculator."""
    if not _registry.is_model_available("equiformer"):
        raise ValueError("Equiformer is not available")
    return _registry.create_calculator("equiformer", **kwargs)


def get_mace_calculator(**kwargs) -> BaseMLIPCalculator:
    """Get a MACE calculator."""
    if not _registry.is_model_available("mace"):
        raise ValueError("MACE is not available")
    return _registry.create_calculator("mace", **kwargs)


def get_orb_calculator(**kwargs) -> BaseMLIPCalculator:
    """Get an ORB calculator."""
    if not _registry.is_model_available("orb"):
        raise ValueError("ORB is not available")
    return _registry.create_calculator("orb", **kwargs)


def get_uma_calculator(**kwargs) -> BaseMLIPCalculator:
    """Get a UMA calculator."""
    if not _registry.is_model_available("uma"):
        raise ValueError("UMA is not available")
