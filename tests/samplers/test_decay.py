from random import Random
from unittest.mock import Mock

import pytest

from dynamicprompts.commands import LiteralCommand
from dynamicprompts.enums import SamplingMethod
from dynamicprompts.generators.randomprompt import RandomPromptGenerator
from dynamicprompts.samplers import DecayRandomSampler, RandomSampler
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.wildcards import WildcardManager
from dynamicprompts.wildcards.item import WildcardItem
from dynamicprompts.wildcards.values import WildcardValues


@pytest.fixture
def context(tmp_path):
    (tmp_path / "color.txt").write_text("1::A\n2::B\n4::C\n", encoding="utf-8")
    (tmp_path / "pose.txt").write_text("A\nB\nC\n", encoding="utf-8")
    manager = WildcardManager(tmp_path)
    return SamplingContext(SamplingMethod.DECAY_RANDOM, manager, rand=Random(12))


def sample(context, prompt="__color__", count=1):
    return [str(value) for value in context.sample_prompts(prompt, count)]


def test_method_and_context_lifetime(context):
    assert SamplingMethod.DECAY_RANDOM.is_nonfinite()
    assert not SamplingMethod.COMBINATORIAL.is_nonfinite()
    assert SamplingMethod.RANDOM.value == "random"
    assert SamplingMethod.CYCLICAL.value == "cycle"
    assert isinstance(context.default_sampler, DecayRandomSampler)
    derived = context.with_sampling_method(SamplingMethod.RANDOM).with_variables({"x": LiteralCommand("value")})
    assert derived.samplers[SamplingMethod.DECAY_RANDOM] is context.default_sampler
    fresh = SamplingContext(SamplingMethod.DECAY_RANDOM, context.wildcard_manager)
    assert fresh.default_sampler is not context.default_sampler


def test_weight_recovery_reselection_and_original_weights(context, monkeypatch):
    sampler = context.default_sampler
    values = context.wildcard_manager.get_values("color")
    original = values.items
    state = sampler._get_state(context.wildcard_manager, "color", values)
    assert sampler._effective_weights(state) == [1, 2, 4]
    choices = Mock(side_effect=[[1], [0], [0], [0], [1], [0], [1]])
    monkeypatch.setattr(context.rand, "choices", choices)
    expected = [0.4, 0.7, 0.9, 1, 0.4, 0.7, 0.4]
    for factor in expected:
        sample(context)
        assert sampler.RECOVERY_FACTORS[state.stages[1]] == factor
    assert choices.call_args_list[1].kwargs["weights"] == [1, 0.8, 4]
    assert choices.call_args_list[2].kwargs["weights"][1] == 1.4
    assert choices.call_args_list[3].kwargs["weights"][1] == 1.8
    assert all(call.kwargs["k"] == 1 for call in choices.call_args_list)
    assert values.items == original
    assert [item.weight for item in values] == [1, 2, 4]


def test_same_wildcard_occurrences_share_history(context, monkeypatch):
    choices = Mock(side_effect=[[1], [1]])
    monkeypatch.setattr(context.rand, "choices", choices)
    assert sample(context, "__color__, __color__") == ["B, B"]
    assert choices.call_args_list[1].kwargs["weights"] == [1, 0.8, 4]
    assert len(context.default_sampler._states) == 1


def test_different_wildcards_and_managers_are_independent(context, monkeypatch):
    monkeypatch.setattr(context.rand, "choices", Mock(return_value=[0]))
    sample(context, "__color__ __pose__")
    sampler = context.default_sampler
    a = sampler._get_state(context.wildcard_manager, "color", context.wildcard_manager.get_values("color"))
    b = sampler._get_state(context.wildcard_manager, "pose", context.wildcard_manager.get_values("pose"))
    assert a is not b
    assert b.stages == [0, 3, 3]
    other = sampler._get_state(WildcardManager(), "color", WildcardValues.from_items(["A", "B"]))
    assert other.stages == [3, 3]


def test_duplicate_rows_keep_separate_indices(context, monkeypatch):
    manager = context.wildcard_manager
    manager.dedup_wildcards = False
    (manager.path / "dupes.txt").write_text("red\nred\nblue\n", encoding="utf-8")
    manager.sort_wildcards = False
    choices = Mock(return_value=[0])
    monkeypatch.setattr(context.rand, "choices", choices)
    sample(context, "__dupes__", 2)
    assert choices.call_args_list[0].kwargs["weights"] == [1, 1, 1]
    assert choices.call_args_list[1].kwargs["weights"] == [0.4, 1, 1]
    assert list(choices.call_args.args[0]) == [0, 1, 2]


def test_changed_candidates_reset_and_normalized_names_share(context):
    sampler = context.default_sampler
    manager = context.wildcard_manager
    values = WildcardValues.from_items(["A", WildcardItem("B", 2)])
    first = sampler._get_state(manager, "color", values)
    first.stages[0] = 0
    assert sampler._get_state(manager, "/color/", values) is first
    changed = sampler._get_state(manager, "color", WildcardValues.from_items(["B", "A"]))
    assert changed is not first
    assert changed.stages == [3, 3]


def test_cache_reload_is_seen_by_existing_generator(context, monkeypatch):
    monkeypatch.setattr(context.rand, "choices", Mock(return_value=[0]))
    gen = iter(context.sample_prompts("__color__", 2))
    assert str(next(gen)) == "A"
    (context.wildcard_manager.path / "color.txt").write_text("new\n", encoding="utf-8")
    context.wildcard_manager.clear_cache()
    assert str(next(gen)) == "new"
    assert len(context.default_sampler._states[(context.wildcard_manager, "color")].stages) == 1


def test_nested_and_derived_context_share(context, monkeypatch):
    (context.wildcard_manager.path / "outer.txt").write_text("__color__", encoding="utf-8")
    choices = Mock(return_value=[1])
    # Select the only outer row; color selection chooses B.
    choices.side_effect = [[0], [1], [1]]
    monkeypatch.setattr(context.rand, "choices", choices)
    assert sample(context, "__outer__ __color__") == ["B B"]
    assert choices.call_args_list[2].kwargs["weights"] == [1, 0.8, 4]


@pytest.mark.parametrize("template", ["{2$$A|B|C}", "{2$$__pose__}"])
def test_variants_and_multiple_choice_match_random(context, template):
    manager = context.wildcard_manager
    normal = RandomPromptGenerator(manager, seed=45)
    decay = RandomPromptGenerator(manager, seed=45, default_sampling_method=SamplingMethod.DECAY_RANDOM)
    assert decay.generate(template, 20) == normal.generate(template, 20)
    assert decay._context.rand.getstate() == normal._context.rand.getstate()
    assert not decay._context.default_sampler._states


def test_seed_reproducibility_and_generator_state_lifetime(context):
    manager = context.wildcard_manager
    a = RandomPromptGenerator(manager, seed=19, default_sampling_method=SamplingMethod.DECAY_RANDOM)
    b = RandomPromptGenerator(manager, seed=19, default_sampling_method=SamplingMethod.DECAY_RANDOM)
    template = "__color__ __pose__ {x|y}"
    assert a.generate(template, 15) == b.generate(template, 15)
    assert a.generate(template, 10) == b.generate(template, 10)
    assert a._context.rand.getstate() == b._context.rand.getstate()


def test_explicit_random_still_uses_original_sampler(context):
    normal = RandomPromptGenerator(context.wildcard_manager, seed=13)
    decay = RandomPromptGenerator(context.wildcard_manager, seed=13, default_sampling_method=SamplingMethod.DECAY_RANDOM)
    assert decay.generate("__~color__", 20) == normal.generate("__color__", 20)
    assert normal._context.rand.getstate() == decay._context.rand.getstate()
    assert not decay._context.default_sampler._states
    assert type(normal._context.default_sampler) is RandomSampler


def test_missing_wildcard_and_single_candidate(context):
    assert sample(context, "__missing__", 2) == ["__missing__"] * 2
    (context.wildcard_manager.path / "one.txt").write_text("only", encoding="utf-8")
    context.wildcard_manager.clear_cache()
    assert sample(context, "__one__", 4) == ["only"] * 4
