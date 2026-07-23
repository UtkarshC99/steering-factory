"""Tests for the quantization preset manifests (manifests/quantization/*.yaml).

Each preset crosses a precision level with parameters chosen for that
precision's specific, stated hypothesis (see docs/quantization_matrix.md and
each file's own header comment) -- these tests check the structural
invariants that make the presets safe to run and comparable to each other:
every model `id` is globally unique (so runs from different presets never
collide on artifact keys -- see runner.py's `id`-based vector/model-json/
qlora-dir naming), each preset's steering-side and QLoRA-side precision
match per model entry (the "matched precision" invariant _bnb_config_for/
resolve_dtype makes possible, see test_finetune_quantization.py), and every
file loads through the real manifest validator.
"""
from pathlib import Path

import pytest
import yaml

from steering_factory.manifest import load_manifest
from steering_factory.model_utils import bnb_config_for, resolve_dtype
from steering_factory.runner import _resolve_layers

PRESETS_DIR = Path(__file__).resolve().parent.parent / "manifests" / "quantization"
PRESET_FILES = sorted(PRESETS_DIR.glob("*.yaml"))

EXPECTED_PRECISION_BY_FILE = {
    "preset_4bit_baseline.yaml": ("4bit", "bfloat16"),
    "preset_8bit_midpoint.yaml": ("8bit", "bfloat16"),
    "preset_bf16_control.yaml": ("none", "bfloat16"),
    "preset_low_bit_layer_scan.yaml": ("4bit", "bfloat16"),
}


def test_all_four_presets_exist():
    names = {p.name for p in PRESET_FILES}
    assert names == set(EXPECTED_PRECISION_BY_FILE)


@pytest.mark.parametrize("path", PRESET_FILES, ids=lambda p: p.name)
def test_preset_loads_and_validates(path):
    manifest = load_manifest(str(path), [])  # raises on any structural problem
    assert manifest["models"]
    assert manifest["recipes"]


@pytest.mark.parametrize("path", PRESET_FILES, ids=lambda p: p.name)
def test_preset_declares_its_documented_precision(path):
    quantization, dtype = EXPECTED_PRECISION_BY_FILE[path.name]
    manifest = load_manifest(str(path), [])
    for model in manifest["models"]:
        assert model.get("quantization") == quantization, f"{path.name}: {model['id']}"
        assert model.get("dtype") == dtype, f"{path.name}: {model['id']}"


def test_every_model_id_is_globally_unique_across_all_presets():
    """The collision hazard confirmed during exploration: runner.py
    namespaces every vector filename, model json, qlora adapter dir, and
    result row's model_id purely by the manifest model entry's `id` -- NOT
    by name_or_path or quantization. If a user runs several presets against
    the same artifacts root, colliding ids would silently overwrite each
    other's vectors/results. Each preset's own ids must be internally
    unique too, and (looking across all four files together) must never
    repeat -- so a real multi-preset run stays safe even if a user reuses
    one artifacts.root across presets."""
    all_ids = []
    for path in PRESET_FILES:
        manifest = load_manifest(str(path), [])
        for model in manifest["models"]:
            all_ids.append(model["id"])
    assert len(all_ids) == len(set(all_ids)), f"Duplicate model ids across presets: {all_ids}"


@pytest.mark.parametrize("path", PRESET_FILES, ids=lambda p: p.name)
def test_preset_matches_steering_and_qlora_precision_per_model_entry(path):
    """The "matched precision" invariant this whole quantization design
    depends on (decision from planning: parameterize BOTH arms): a preset's
    steering side (model_utils.bnb_config_for) and QLoRA side
    (finetune.py's now-parameterized train/eval loaders, driven by the same
    model.get("quantization")/model.get("dtype") fields via runner.py) must
    resolve to the IDENTICAL BitsAndBytesConfig shape for the same model
    entry -- there is only one `quantization`/`dtype` per model entry, and
    both arms read it, so this is really a test that the fields exist and
    resolve consistently, not that two different paths could diverge."""
    manifest = load_manifest(str(path), [])
    for model in manifest["models"]:
        dtype = resolve_dtype(model.get("dtype"))
        steering_side = bnb_config_for(model.get("quantization"), dtype)
        qlora_side = bnb_config_for(model.get("quantization", "4bit"), dtype)  # runner.py's qlora default
        if steering_side is None:
            assert qlora_side is None
        else:
            assert qlora_side is not None
            assert type(steering_side) is type(qlora_side)
            assert steering_side.load_in_4bit == qlora_side.load_in_4bit
            assert steering_side.load_in_8bit == qlora_side.load_in_8bit


def test_bf16_control_widens_layer_grid_relative_to_baseline():
    """The bf16 control preset is documented as deliberately widening the
    layer grid (cheapest preset to widen safely) to establish which layer
    is robust with no quantization noise -- confirms the manifest actually
    does this, not just the header comment."""
    baseline = load_manifest(str(PRESETS_DIR / "preset_4bit_baseline.yaml"), [])
    control = load_manifest(str(PRESETS_DIR / "preset_bf16_control.yaml"), [])
    baseline_layers = next(r for r in baseline["recipes"] if r["id"] == "harmful_instruction_compliance")["application"]["layers"]
    control_layers = next(r for r in control["recipes"] if r["id"] == "harmful_instruction_compliance")["application"]["layers"]
    assert len(control_layers) > len(baseline_layers)
    # And they must resolve to genuinely distinct layer indices, not all
    # collapse to the same default -- this is exactly the bug the
    # runner._resolve_layers fix (adding explicit "early"/"mid"/"late"
    # entries) addresses; a 20-layer stand-in model is enough to check
    # three fractions land on three different integers.
    resolved = _resolve_layers(control_layers, num_layers=20)
    assert len(resolved) == len(set(resolved)) == 3


def test_low_bit_layer_scan_matches_baseline_precision_not_a_new_level():
    """preset_low_bit_layer_scan.yaml is documented as the SAME 4bit
    precision as the baseline, with the experiment design (not the
    precision) changed -- confirms it isn't accidentally a 5th precision
    level."""
    baseline = load_manifest(str(PRESETS_DIR / "preset_4bit_baseline.yaml"), [])
    layer_scan = load_manifest(str(PRESETS_DIR / "preset_low_bit_layer_scan.yaml"), [])
    baseline_precisions = {(m["quantization"], m["dtype"]) for m in baseline["models"]}
    layer_scan_precisions = {(m["quantization"], m["dtype"]) for m in layer_scan["models"]}
    assert baseline_precisions == layer_scan_precisions


@pytest.mark.parametrize("path", PRESET_FILES, ids=lambda p: p.name)
def test_batch_size_does_not_exceed_the_documented_safe_ceiling(path):
    """Sanity pin on the memory-sizing math in each file's header comment --
    not a full memory simulation, just a guard against a future edit
    silently raising batch_size past what that preset's own precision
    budget was sized for. Ceilings taken directly from the derivation in
    each file's header (4bit~33, 8bit~28, bf16~18 max safe batch at a 6.5
    GiB margin on a 22.03 GiB L4)."""
    ceilings = {
        "preset_4bit_baseline.yaml": 33,
        "preset_8bit_midpoint.yaml": 28,
        "preset_bf16_control.yaml": 18,
        "preset_low_bit_layer_scan.yaml": 33,
    }
    manifest = load_manifest(str(path), [])
    assert manifest["decoding"]["batch_size"] <= ceilings[path.name]
    assert manifest["extraction"]["batch_size"] <= ceilings[path.name]


def test_yaml_files_are_plain_yaml_not_just_manifest_loadable():
    # Defense-in-depth: confirms these are also valid standalone YAML (not
    # relying on load_manifest's override-merging to paper over a syntax
    # issue), since a user may `cat`/copy these directly.
    for path in PRESET_FILES:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        assert isinstance(data, dict)
        assert "models" in data and "recipes" in data
