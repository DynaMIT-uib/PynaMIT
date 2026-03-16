"""Helpers for tagging and validating preconditioners on LinearMap space metadata."""

from __future__ import annotations

from typing import Literal, Optional, TypeAlias

from .linear_map import LinearMap, as_linear_map

PreconditionerSpaceKind: TypeAlias = Literal["full", "reduced"]


def preconditioner_space_name(system_id: Optional[str], space_kind: PreconditionerSpaceKind) -> str:
    """Return the canonical LinearMap space label for one preconditioner space."""
    if system_id is None or not str(system_id).strip():
        return space_kind
    return f"{system_id}:{space_kind}"


def make_preconditioner(
    linear_map: LinearMap | object,
    *,
    system_id: Optional[str],
    space_kind: PreconditionerSpaceKind,
) -> LinearMap:
    """Return a LinearMap tagged as a preconditioner in one coordinate space."""
    base = as_linear_map(linear_map)
    space_name = preconditioner_space_name(system_id, space_kind)
    return base.with_spaces(domain_space=space_name, codomain_space=space_name)


def validate_preconditioner(
    preconditioner: LinearMap | None,
    *,
    system_id: Optional[str] = None,
    allowed_space_kinds: tuple[PreconditionerSpaceKind, ...] = ("full", "reduced"),
) -> Optional[LinearMap]:
    """Validate that a LinearMap is a tagged preconditioner in an allowed space."""
    if preconditioner is None:
        return None
    pre = as_linear_map(preconditioner)
    if pre.domain_space is None or pre.codomain_space is None:
        raise ValueError("Preconditioner LinearMap must carry domain/codomain space metadata.")
    if pre.domain_space != pre.codomain_space:
        raise ValueError(
            "Preconditioner LinearMap must be an endomorphism with matching domain/codomain "
            f"spaces, got {pre.codomain_space!r} <- {pre.domain_space!r}."
        )

    if system_id is not None:
        allowed_names = {
            preconditioner_space_name(system_id, space_kind) for space_kind in allowed_space_kinds
        }
        if pre.domain_space not in allowed_names:
            raise ValueError(
                f"Invalid preconditioner space {pre.domain_space!r}; expected one of "
                f"{sorted(allowed_names)!r}."
            )
    return pre


def preconditioner_space_kind(
    preconditioner: LinearMap, *, system_id: Optional[str]
) -> PreconditionerSpaceKind:
    """Return whether a validated preconditioner lives in full or reduced coordinates."""
    pre = validate_preconditioner(preconditioner, system_id=system_id)
    assert pre is not None
    full_name = preconditioner_space_name(system_id, "full")
    return "full" if pre.domain_space == full_name else "reduced"
