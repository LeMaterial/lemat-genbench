"""Registry for relaxer implementations and base classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type

from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry


@dataclass
class RelaxationResult:
    """Result of a structure relaxation.

    Parameters
    ----------
    success : bool
        Whether the relaxation was successful.
    energy : float | None
        Final energy of the relaxed structure.
    structure : Structure | None
        Final relaxed structure.
    message : str | None
        Error message if relaxation failed.
    """

    success: bool
    energy: float | None = None
    structure: Structure | None = None
    message: str | None = None


class BaseRelaxer(ABC):
    """Base class for structure relaxation implementations.

    All relaxer implementations should inherit from this class and implement
    the required methods.
    """

    @abstractmethod
    def relax(self, structure: Structure, relax: bool = False) -> RelaxationResult:
        """Relax a structure and return the result.

        Parameters
        ----------
        structure : Structure
            Structure to relax.
        relax : bool
            Whether to actually perform relaxation.

        Returns
        -------
        RelaxationResult
            Result of the relaxation.
        """
        pass

    def get_computed_entry(
        self, structure: Structure, energy: float
    ) -> ComputedStructureEntry:
        """Create a ComputedStructureEntry from a relaxed structure.

        Parameters
        ----------
        structure : Structure
            The relaxed structure.
        energy : float
            The energy of the relaxed structure.

        Returns
        -------
        ComputedStructureEntry
            The computed structure entry with appropriate corrections applied.
        """
        # Default implementation - can be overridden by subclasses
        return ComputedStructureEntry(structure, energy, correction=0.0)


_RELAXER_REGISTRY: Dict[str, Type[BaseRelaxer]] = {}


def register_relaxer(name: str):
    """Register a relaxer implementation.

    Parameters
    ----------
    name : str
        Name of the relaxer.

    Returns
    -------
    callable
        Decorator function.
    """

    def decorator(cls: Type[BaseRelaxer]) -> Type[BaseRelaxer]:
        if not issubclass(cls, BaseRelaxer):
            raise TypeError(f"{cls.__name__} must inherit from BaseRelaxer")
        _RELAXER_REGISTRY[name] = cls
        return cls

    return decorator


def get_relaxer(relaxer_type: str, **kwargs) -> BaseRelaxer:
    """Get a relaxer implementation by name.

    This function first tries to use the new MLIP-based relaxers,
    and falls back to the legacy system if needed.

    Parameters
    ----------
    relaxer_type : str
        Name of the relaxer.
    **kwargs
        Additional arguments to pass to the relaxer constructor.

    Returns
    -------
    BaseRelaxer
        Relaxer instance.

    Raises
    ------
    ValueError
        If relaxer_type is not registered.
    """
    # Try new MLIP-based relaxers first
    if relaxer_type in _RELAXER_REGISTRY:
        return _RELAXER_REGISTRY[relaxer_type](**kwargs)

    # Fallback: try to create MLIP relaxer directly
    try:
        from lemat_genbench.models.registry import (
            get_calculator,
            list_available_models,
        )

        if relaxer_type in list_available_models():
            # Create calculator and wrap it in MLIPRelaxer
            calculator = get_calculator(relaxer_type, **kwargs)

            # Import here to avoid circular imports
            from lemat_genbench.utils.relaxers.relaxers import MLIPRelaxer

            return MLIPRelaxer(calculator, **kwargs)

    except ImportError:
        pass

    # If not found, raise error
    available_relaxers = list(_RELAXER_REGISTRY.keys())
    try:
        from lemat_genbench.models.registry import list_available_models

        available_relaxers.extend(list_available_models())
    except ImportError:
        pass

    raise ValueError(
        f"Unknown relaxer type: {relaxer_type}. Available types: {available_relaxers}"
    )


def list_available_relaxers() -> list[str]:
    """List all available relaxer types.

    Returns
    -------
    list[str]
        List of available relaxer names.
    """
    relaxers = list(_RELAXER_REGISTRY.keys())

    # Add MLIP calculators as available relaxers
    try:
        from lemat_genbench.models.registry import list_available_models

        relaxers.extend(list_available_models())
    except ImportError:
        pass

    return list(set(relaxers))  # Remove duplicates


def print_available_relaxers():
    """Print information about available relaxers."""
    relaxers = list_available_relaxers()
    print("Available Relaxers:")
    print("=" * 30)

    for relaxer in relaxers:
        print(f"- {relaxer}")

        # Add description if available
        if relaxer in _RELAXER_REGISTRY:
            cls = _RELAXER_REGISTRY[relaxer]
            if hasattr(cls, "__doc__") and cls.__doc__:
                print(f"  {cls.__doc__.strip()}")

    print(f"\nTotal: {len(relaxers)} relaxers available")
