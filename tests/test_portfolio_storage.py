"""Tests for the public-storage / owner-key portfolio module.

Each test uses ``tmp_path`` so nothing touches the real
``data/user_portfolios/`` directory.
"""
from __future__ import annotations

import json

import pytest

from app import portfolio_storage as ps


# ──────────────────────────────────────────────────────────────────────────
# Slugify
# ──────────────────────────────────────────────────────────────────────────

class TestSlugify:

    @pytest.mark.parametrize("name, expected", [
        ("My Portfolio",       "my-portfolio"),
        ("MY-PORTFOLIO",       "my-portfolio"),
        ("  spaced  out  ",    "spaced-out"),
        ("CH1537766565",       "ch1537766565"),
        ("foo!!!bar",          "foo-bar"),
        ("über portfolio 2",   "ber-portfolio-2"),   # non-ASCII collapsed
    ])
    def test_canonical_forms(self, name, expected):
        assert ps.slugify(name) == expected

    @pytest.mark.parametrize("name", ["", "   ", "!!!", "---"])
    def test_empty_or_garbage_raises(self, name):
        with pytest.raises(ValueError):
            ps.slugify(name)


# ──────────────────────────────────────────────────────────────────────────
# Key generation & hashing
# ──────────────────────────────────────────────────────────────────────────

class TestKeyMechanics:

    def test_generated_keys_are_unique(self):
        keys = {ps.generate_owner_key() for _ in range(100)}
        assert len(keys) == 100   # no collisions in 100 draws

    def test_key_length_reasonable(self):
        # token_urlsafe(32) → 43 chars (URL-safe base64 of 32 bytes).
        assert len(ps.generate_owner_key()) >= 32

    def test_hash_is_deterministic(self):
        k = "abcd"
        assert ps._hash_key(k) == ps._hash_key(k)

    def test_verify_accepts_correct_key(self):
        k = ps.generate_owner_key()
        assert ps._verify_key(k, ps._hash_key(k))

    def test_verify_rejects_wrong_key(self):
        k = ps.generate_owner_key()
        assert not ps._verify_key("not-the-key", ps._hash_key(k))

    def test_verify_rejects_unprefixed_hash(self):
        k = "x"
        bare_hex = ps._hash_key(k)[len(ps._HASH_PREFIX):]
        assert not ps._verify_key(k, bare_hex)


# ──────────────────────────────────────────────────────────────────────────
# Save / load / list — happy path
# ──────────────────────────────────────────────────────────────────────────

class TestSaveLoadList:

    def test_save_new_returns_slug_and_key(self, tmp_path):
        slug, key = ps.save_new(
            "Vontobel Book",
            [{"product_id": "X"}],
            portfolios_dir=tmp_path,
        )
        assert slug == "vontobel-book"
        assert isinstance(key, str) and len(key) > 30

    def test_saved_file_contains_expected_fields(self, tmp_path):
        ps.save_new("Vontobel Book",
                    [{"product_id": "X"}, {"product_id": "Y"}],
                    portfolios_dir=tmp_path)
        payload = json.loads(
            (tmp_path / "vontobel-book.json").read_text(encoding="utf-8")
        )
        assert payload["name"] == "Vontobel Book"
        assert payload["slug"] == "vontobel-book"
        assert payload["owner_key_hash"].startswith(ps._HASH_PREFIX)
        assert len(payload["products"]) == 2
        assert "created_at" in payload and "updated_at" in payload

    def test_load_returns_products(self, tmp_path):
        ps.save_new("p", [{"product_id": "X"}], portfolios_dir=tmp_path)
        loaded = ps.load("p", portfolios_dir=tmp_path)
        assert loaded["products"] == [{"product_id": "X"}]

    def test_load_is_case_insensitive_via_slug(self, tmp_path):
        ps.save_new("My Portfolio", [], portfolios_dir=tmp_path)
        # Reading via different surface forms hits the same slug
        for variant in ("MY PORTFOLIO", "my portfolio", " my  portfolio  "):
            ps.load(variant, portfolios_dir=tmp_path)

    def test_list_returns_summary(self, tmp_path):
        ps.save_new("A", [{}, {}], portfolios_dir=tmp_path)
        ps.save_new("B", [{}],     portfolios_dir=tmp_path)
        summary = ps.list_saved(portfolios_dir=tmp_path)
        names = {s.name for s in summary}
        assert names == {"A", "B"}
        counts = {s.name: s.n_products for s in summary}
        assert counts == {"A": 2, "B": 1}

    def test_list_empty_dir_returns_empty(self, tmp_path):
        assert ps.list_saved(portfolios_dir=tmp_path / "does-not-exist") == []

    def test_exists_round_trip(self, tmp_path):
        assert not ps.exists("p", portfolios_dir=tmp_path)
        ps.save_new("p", [], portfolios_dir=tmp_path)
        assert ps.exists("p", portfolios_dir=tmp_path)
        assert ps.exists("P", portfolios_dir=tmp_path)   # slug equivalence


# ──────────────────────────────────────────────────────────────────────────
# Save / load — error paths
# ──────────────────────────────────────────────────────────────────────────

class TestSaveLoadErrors:

    def test_save_new_collision_raises(self, tmp_path):
        ps.save_new("p", [], portfolios_dir=tmp_path)
        with pytest.raises(ps.NameExistsError):
            ps.save_new("p", [], portfolios_dir=tmp_path)

    def test_load_unknown_raises(self, tmp_path):
        with pytest.raises(ps.NotFoundError):
            ps.load("nope", portfolios_dir=tmp_path)

    def test_list_skips_corrupt_files(self, tmp_path):
        ps.save_new("ok", [{"x": 1}], portfolios_dir=tmp_path)
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        summary = ps.list_saved(portfolios_dir=tmp_path)
        assert [s.name for s in summary] == ["ok"]


# ──────────────────────────────────────────────────────────────────────────
# Overwrite (owner-key gated)
# ──────────────────────────────────────────────────────────────────────────

class TestOverwrite:

    def test_overwrite_with_correct_key_replaces_products(self, tmp_path):
        _, key = ps.save_new("p", [{"v": 1}], portfolios_dir=tmp_path)
        ps.overwrite("p", key, [{"v": 2}, {"v": 3}], portfolios_dir=tmp_path)
        loaded = ps.load("p", portfolios_dir=tmp_path)
        assert loaded["products"] == [{"v": 2}, {"v": 3}]

    def test_overwrite_with_wrong_key_raises(self, tmp_path):
        ps.save_new("p", [{"v": 1}], portfolios_dir=tmp_path)
        with pytest.raises(ps.AuthError):
            ps.overwrite("p", "fake-key", [{"v": 2}], portfolios_dir=tmp_path)
        # Original content untouched
        assert ps.load("p", portfolios_dir=tmp_path)["products"] == [{"v": 1}]

    def test_overwrite_unknown_name_raises(self, tmp_path):
        with pytest.raises(ps.NotFoundError):
            ps.overwrite("nope", "k", [], portfolios_dir=tmp_path)

    def test_overwrite_updates_timestamp(self, tmp_path):
        _, key = ps.save_new("p", [], portfolios_dir=tmp_path)
        original_updated = ps.load("p", portfolios_dir=tmp_path)["updated_at"]
        ps.overwrite("p", key, [], portfolios_dir=tmp_path)
        new_updated = ps.load("p", portfolios_dir=tmp_path)["updated_at"]
        # updated_at moves forward (or at minimum doesn't regress)
        assert new_updated >= original_updated

    def test_overwrite_updates_reference_currency(self, tmp_path):
        _, key = ps.save_new(
            "p", [], portfolios_dir=tmp_path, reference_currency="CHF",
        )
        ps.overwrite(
            "p", key, [{"x": 1}],
            portfolios_dir=tmp_path, reference_currency="USD",
        )
        loaded = ps.load("p", portfolios_dir=tmp_path)
        assert loaded["reference_currency"] == "USD"

    def test_overwrite_preserves_currency_when_not_specified(self, tmp_path):
        _, key = ps.save_new(
            "p", [], portfolios_dir=tmp_path, reference_currency="EUR",
        )
        ps.overwrite("p", key, [{"x": 1}], portfolios_dir=tmp_path)
        loaded = ps.load("p", portfolios_dir=tmp_path)
        assert loaded["reference_currency"] == "EUR"


# ──────────────────────────────────────────────────────────────────────────
# Reference currency persistence
# ──────────────────────────────────────────────────────────────────────────

class TestReferenceCurrency:

    def test_save_persists_reference_currency(self, tmp_path):
        ps.save_new(
            "p", [{"v": 1}],
            portfolios_dir=tmp_path, reference_currency="USD",
        )
        loaded = ps.load("p", portfolios_dir=tmp_path)
        assert loaded["reference_currency"] == "USD"

    def test_save_defaults_to_chf(self, tmp_path):
        ps.save_new("p", [], portfolios_dir=tmp_path)
        loaded = ps.load("p", portfolios_dir=tmp_path)
        assert loaded["reference_currency"] == "CHF"


# ──────────────────────────────────────────────────────────────────────────
# Delete (owner-key gated)
# ──────────────────────────────────────────────────────────────────────────

class TestDelete:

    def test_delete_with_correct_key_removes_file(self, tmp_path):
        _, key = ps.save_new("p", [], portfolios_dir=tmp_path)
        ps.delete("p", key, portfolios_dir=tmp_path)
        assert not ps.exists("p", portfolios_dir=tmp_path)

    def test_delete_with_wrong_key_raises(self, tmp_path):
        ps.save_new("p", [], portfolios_dir=tmp_path)
        with pytest.raises(ps.AuthError):
            ps.delete("p", "fake", portfolios_dir=tmp_path)
        assert ps.exists("p", portfolios_dir=tmp_path)

    def test_delete_unknown_name_raises(self, tmp_path):
        with pytest.raises(ps.NotFoundError):
            ps.delete("nope", "k", portfolios_dir=tmp_path)
