from random import Random
from unittest.mock import Mock

import pytest

from dynamicprompts.enums import SamplingMethod
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.wildcards import WildcardManager


@pytest.fixture
def context(tmp_path):
    (tmp_path / "color.txt").write_text("red\nblue\ngreen\n", encoding="utf-8")
    (tmp_path / "nested.txt").write_text("{red|blue|green}\n", encoding="utf-8")
    return SamplingContext(SamplingMethod.DECAY_RANDOM, WildcardManager(tmp_path), rand=Random(12))


def sample(context, prompt, count=1):
    return [str(result) for result in context.sample_prompts(prompt, count)]


def test_recovery_and_reselection(context, monkeypatch):
    choices = Mock(side_effect=[[0], [1], [1], [1], [0], [1], [0]])
    monkeypatch.setattr(context.rand, "choices", choices)
    sampler = context.default_sampler
    for expected in [0.4, 0.7, 0.9, 1.0, 0.4, 0.7, 0.4]:
        sample(context, "{red|blue|green}")
        state = next(iter(sampler._variant_states.values()))
        assert sampler.RECOVERY_FACTORS[state.stages[0]] == expected
    assert choices.call_args_list[0].kwargs["weights"] == [1, 1, 1]
    assert choices.call_args_list[1].kwargs["weights"] == [0.4, 1, 1]
    assert all(call.kwargs["k"] == 1 for call in choices.call_args_list)


def test_same_expression_shares_across_occurrences_and_calls(context, monkeypatch):
    choices = Mock(return_value=[0])
    monkeypatch.setattr(context.rand, "choices", choices)
    assert sample(context, "{red|blue|green}, {red|blue|green}") == ["red, red"]
    sample(context, "{red|blue|green}")
    assert choices.call_args_list[1].kwargs["weights"] == [0.4, 1, 1]
    assert choices.call_args_list[2].kwargs["weights"] == [0.4, 1, 1]
    assert len(context.default_sampler._variant_states) == 1


def test_structure_order_and_weights_separate_history(context, monkeypatch):
    choices = Mock(return_value=[0])
    monkeypatch.setattr(context.rand, "choices", choices)
    for prompt in ["{red|blue|green}", "{red|blue|yellow}", "{green|red|blue}", "{2::red|blue|green}"]:
        sample(context, prompt)
    assert len(context.default_sampler._variant_states) == 4
    assert [call.kwargs["weights"] for call in choices.call_args_list] == [[1, 1, 1]] * 3 + [[2, 1, 1]]


def test_weighted_base_weights_are_immutable(context, monkeypatch):
    choices = Mock(return_value=[0])
    monkeypatch.setattr(context.rand, "choices", choices)
    sample(context, "{2::red|1::blue|1::green}", 2)
    state = next(iter(context.default_sampler._variant_states.values()))
    assert [weight for _, weight in state.candidates] == [2, 1, 1]
    assert choices.call_args_list[1].kwargs["weights"] == [0.8, 1, 1]


def test_duplicate_candidates_keep_indices(context, monkeypatch):
    choices = Mock(side_effect=[[0], [1]])
    monkeypatch.setattr(context.rand, "choices", choices)
    assert sample(context, "{red|red|blue}", 2) == ["red", "red"]
    assert choices.call_args_list[1].kwargs["weights"] == [0.4, 1, 1]
    state = next(iter(context.default_sampler._variant_states.values()))
    assert state.stages == [1, 0, 3]


def test_nested_variants_have_separate_history(context, monkeypatch):
    choices = Mock(side_effect=[[1], [0], [1], [1]])
    monkeypatch.setattr(context.rand, "choices", choices)
    assert sample(context, "{red|{blue|green}}", 2) == ["blue", "green"]
    assert len(context.default_sampler._variant_states) == 2
    assert choices.call_args_list[2].kwargs["weights"] == [1, 0.4]
    assert choices.call_args_list[3].kwargs["weights"] == [0.4, 1]


@pytest.mark.parametrize("prompt", ["__color__, {red|blue|green}", "{__color__|black}", "__nested__"])
def test_wildcards_and_variants_are_independent(context, monkeypatch, prompt):
    monkeypatch.setattr(context.rand, "choices", Mock(return_value=[0]))
    sample(context, prompt)
    assert len(context.default_sampler._states) == 1
    assert len(context.default_sampler._variant_states) == 1
    assert next(iter(context.default_sampler._states.values())) is not next(iter(context.default_sampler._variant_states.values()))


@pytest.mark.parametrize("prompt", ["{2$$red|blue|green}", "{1-2$$red|blue|green}", "{0$$red|blue|green}", "{red}"])
def test_multiselect_ranged_and_singleton_preserve_random(context, prompt):
    normal = RandomPromptGenerator(context.wildcard_manager, seed=45)
    decay = RandomPromptGenerator(context.wildcard_manager, seed=45, default_sampling_method=SamplingMethod.DECAY_RANDOM)
    assert decay.generate(prompt, 30) == normal.generate(prompt, 30)
    assert decay._context.rand.getstate() == normal._context.rand.getstate()
    assert not decay._context.default_sampler._variant_states


@pytest.mark.parametrize("prompt", ["{~2::red|blue|green}", "{@red|blue|green}"])
def test_explicit_methods_preserve_routing(context, prompt):
    normal = RandomPromptGenerator(context.wildcard_manager, seed=45)
    decay = RandomPromptGenerator(context.wildcard_manager, seed=45, default_sampling_method=SamplingMethod.DECAY_RANDOM)
    assert decay.generate(prompt, 30) == normal.generate(prompt, 30)
    assert decay._context.rand.getstate() == normal._context.rand.getstate()
    assert not decay._context.default_sampler._variant_states


def test_generator_lifetime_and_reproducibility(context, monkeypatch):
    def create():
        return RandomPromptGenerator(context.wildcard_manager, seed=19, default_sampling_method=SamplingMethod.DECAY_RANDOM)
    a, b = create(), create()
    prompt = "{red|blue|green} {red|{blue|green}}"
    assert a.generate(prompt, 20) == b.generate(prompt, 20)
    assert a.generate(prompt, 10) == b.generate(prompt, 10)
    assert a._context.rand.getstate() == b._context.rand.getstate()
    c = create()
    choices = Mock(return_value=[0])
    monkeypatch.setattr(c._context.rand, "choices", choices)
    sampler = c._context.default_sampler
    c.generate("{red|blue|green}", 1)
    c.generate("{red|blue|green}", 1)
    assert c._context.default_sampler is sampler
    assert choices.call_args_list[1].kwargs["weights"] == [0.4, 1, 1]
    fresh = create()
    other = Mock(return_value=[0])
    monkeypatch.setattr(fresh._context.rand, "choices", other)
    fresh.generate("{red|blue|green}", 1)
    assert fresh._context.default_sampler is not sampler
    assert other.call_args.kwargs["weights"] == [1, 1, 1]
