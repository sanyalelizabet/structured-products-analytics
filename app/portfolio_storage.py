"""Public, named portfolio storage with owner-key write protection.

Design summary
--------------
Saved portfolios live as JSON files in ``data/user_portfolios/``.
Reads are public: anyone using the app can list and load every saved
portfolio.  Writes (overwrite or delete) require the **owner key** that
was shown to the user once at save time and stored only as a SHA-256
hash inside the portfolio file.

This is the "pastebin / gist edit-token" pattern:

* No accounts, no auth machinery.
* Public read, gated write.
* Key is shown to the user once and is unrecoverable — losing it means
  you can't update that portfolio (but you can always Save As under a
  new name).

The 32-character URL-safe owner key carries ~190 bits of entropy, well
beyond brute-force range.  Per-request rate limiting is out of scope
for v1.

File schema
-----------
``<slug>.json`` contains::

    {
      "name":            "human readable name",
      "slug":            "human-readable-name",
      "created_at":      "2026-05-18T...Z",
      "updated_at":      "2026-05-18T...Z",
      "owner_key_hash":  "sha256:<hex>",
      "reference_currency": "CHF",  // analytics roll-up currency
      "products":        [<row dict>, ...]
    }
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import secrets
from pathlib import Path
from typing import Any, NamedTuple


# Default storage location.  Tests inject a different path via the
# ``portfolios_dir=`` parameter on every public function.
DEFAULT_PORTFOLIOS_DIR = Path("data") / "user_portfolios"

_HASH_PREFIX = "sha256:"
_KEY_BYTES = 32   # secrets.token_urlsafe(32) → 43 chars, ~256 bits.


# ──────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────

class StorageError(Exception):
    """Base class for storage errors raised to the UI layer."""


class NotFoundError(StorageError):
    """Raised when the requested portfolio name doesn't exist."""


class NameExistsError(StorageError):
    """Raised when ``save_new`` is called on a name that already exists."""


class AuthError(StorageError):
    """Raised when an owner-key check fails on an overwrite or delete."""


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Convert a display name to a filesystem-safe slug.

    Lowercase, non-alphanumerics collapsed to single hyphens, leading
    and trailing hyphens stripped.  Empty input raises ``ValueError``
    so the caller can surface a friendly message.
    """
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"Name {name!r} produced an empty slug.")
    return slug


def generate_owner_key() -> str:
    """Return a fresh URL-safe owner key.  Show to the user once."""
    return secrets.token_urlsafe(_KEY_BYTES)


def _hash_key(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"{_HASH_PREFIX}{digest}"


def _verify_key(key: str, stored_hash: str) -> bool:
    if not stored_hash.startswith(_HASH_PREFIX):
        return False
    return secrets.compare_digest(_hash_key(key), stored_hash)


def _now_iso() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _path_for(slug: str, portfolios_dir: Path) -> Path:
    return portfolios_dir / f"{slug}.json"


def _ensure_dir(portfolios_dir: Path) -> None:
    portfolios_dir.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

class PortfolioSummary(NamedTuple):
    """Lightweight metadata for the saved-portfolios listing."""
    name: str
    slug: str
    created_at: str
    updated_at: str
    n_products: int


def list_saved(portfolios_dir: Path = DEFAULT_PORTFOLIOS_DIR) -> list[PortfolioSummary]:
    """List every saved portfolio (read-only metadata, no key needed)."""
    if not portfolios_dir.exists():
        return []
    out: list[PortfolioSummary] = []
    for p in sorted(portfolios_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Skip unreadable / corrupt files instead of crashing the UI.
            continue
        out.append(PortfolioSummary(
            name=payload.get("name", p.stem),
            slug=payload.get("slug", p.stem),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
            n_products=len(payload.get("products", []) or []),
        ))
    return out


def exists(name: str, portfolios_dir: Path = DEFAULT_PORTFOLIOS_DIR) -> bool:
    return _path_for(slugify(name), portfolios_dir).exists()


def load(name: str, portfolios_dir: Path = DEFAULT_PORTFOLIOS_DIR) -> dict[str, Any]:
    """Load a saved portfolio by name.  Public read — no key required."""
    path = _path_for(slugify(name), portfolios_dir)
    if not path.exists():
        raise NotFoundError(f"No portfolio called {name!r}.")
    return json.loads(path.read_text(encoding="utf-8"))


def save_new(
    name: str,
    products: list[dict],
    portfolios_dir: Path = DEFAULT_PORTFOLIOS_DIR,
    reference_currency: str = "CHF",
) -> tuple[str, str]:
    """Save a new portfolio.

    Returns ``(slug, plaintext_owner_key)``.  The plaintext key is the
    *only* opportunity to capture it — the caller must show it to the
    user and never log it.  Raises :class:`NameExistsError` if a
    portfolio with this slug already exists.
    """
    _ensure_dir(portfolios_dir)
    slug = slugify(name)
    path = _path_for(slug, portfolios_dir)
    if path.exists():
        raise NameExistsError(f"A portfolio called {name!r} already exists.")

    key = generate_owner_key()
    now = _now_iso()
    payload = {
        "name":               name.strip(),
        "slug":               slug,
        "created_at":         now,
        "updated_at":         now,
        "owner_key_hash":     _hash_key(key),
        "reference_currency": reference_currency,
        "products":           list(products),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return slug, key


def overwrite(
    name: str,
    owner_key: str,
    products: list[dict],
    portfolios_dir: Path = DEFAULT_PORTFOLIOS_DIR,
    reference_currency: str | None = None,
) -> None:
    """Replace ``products`` (and optionally the reference currency) in an
    existing portfolio.

    Requires the plaintext owner key shown at original save time.
    Raises :class:`NotFoundError` if the name doesn't exist, or
    :class:`AuthError` if the key doesn't match.
    """
    path = _path_for(slugify(name), portfolios_dir)
    if not path.exists():
        raise NotFoundError(f"No portfolio called {name!r}.")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _verify_key(owner_key, payload.get("owner_key_hash", "")):
        raise AuthError("Owner key does not match.")

    payload["products"] = list(products)
    if reference_currency is not None:
        payload["reference_currency"] = reference_currency
    payload["updated_at"] = _now_iso()
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def delete(
    name: str,
    owner_key: str,
    portfolios_dir: Path = DEFAULT_PORTFOLIOS_DIR,
) -> None:
    """Remove a saved portfolio.  Requires the owner key."""
    path = _path_for(slugify(name), portfolios_dir)
    if not path.exists():
        raise NotFoundError(f"No portfolio called {name!r}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _verify_key(owner_key, payload.get("owner_key_hash", "")):
        raise AuthError("Owner key does not match.")
    path.unlink()
