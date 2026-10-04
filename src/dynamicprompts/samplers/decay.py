from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from random import Random
from typing import Hashable

from dynamicprompts.commands import Command, VariantCommand, WildcardCommand
from dynamicprompts.samplers.random import RandomSampler
from dynamicprompts.samplers.utils import get_wildcard_not_found_fallback
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.types import ResultGen
from dynamicprompts.wildcards import WildcardManager
from dynamicprompts.wildcards.utils import clean_wildcard
from dynamicprompts.wildcards.values import WildcardValues


@dataclass
class _DecayState:
    candidates: tuple[tuple[Hashable, float], ...]
    stages: list[int]


class DecayRandomSampler(RandomSampler):
    """Decay for ordinary wildcard and fixed single-choice variant selection.

    History belongs to this sampler and is shared by derived SamplingContexts.
    Multi-choice/ranged variants use ordinary Random selection. Instances are
    intended for one generation session, not concurrent use by multiple threads.
    """

    RECOVERY_FACTORS = (0.40, 0.70, 0.90, 1.00)

    def __init__(self) -> None:
        self._states: dict[tuple[WildcardManager, str], _DecayState] = {}
        self._variant_states: dict[tuple, _DecayState] = {}
        self._random_variant_sampler = RandomSampler()

    def _get_state(
        self,
        manager: WildcardManager,
        wildcard: str,
        values: WildcardValues,
    ) -> _DecayState:
        key = (manager, clean_wildcard(wildcard, wildcard_wrap=manager.wildcard_wrap))
        candidates = tuple((str(item), getattr(item, "weight", 1.0)) for item in values)
        state = self._states.get(key)
        # Reset if rows, their order, or original weights changed. Equal duplicate
        # rows remain separate positions; no content-based deduplication occurs.
        if state is None or state.candidates != candidates:
            state = _DecayState(candidates, [len(self.RECOVERY_FACTORS) - 1] * len(values))
            self._states[key] = state
        return state

    def _effective_weights(self, state: _DecayState) -> list[float]:
        return [
            weight * self.RECOVERY_FACTORS[stage]
            for (_, weight), stage in zip(state.candidates, state.stages)
        ]

    def _choose_index(self, context: SamplingContext, state: _DecayState) -> int:
        return self._draw_index(context.rand, state)

    def _draw_index(self, rand: Random, state: _DecayState) -> int:
        # Do not prefetch: each draw must see the preceding draw's penalty.
        index = rand.choices(
            range(len(state.candidates)), weights=self._effective_weights(state), k=1,
        )[0]
        maximum = len(self.RECOVERY_FACTORS) - 1
        state.stages[:] = [min(stage + 1, maximum) for stage in state.stages]
        state.stages[index] = 0
        return index

    @staticmethod
    def _structure(value) -> Hashable:
        """Immutable structural key for built-in AST nodes, without node identity."""
        if is_dataclass(value):
            return (type(value), tuple(
                (field.name, DecayRandomSampler._structure(getattr(value, field.name)))
                for field in fields(value)
            ))
        if isinstance(value, (list, tuple)):
            return tuple(DecayRandomSampler._structure(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted(
                (key, DecayRandomSampler._structure(item)) for key, item in value.items()
            ))
        return value

    def _get_variant_choices(
        self,
        values: list[Command],
        weights: list[float],
        num_choices: int,
        rand: Random,
    ) -> list[Command]:
        if num_choices != 1:
            return super()._get_variant_choices(values, weights, num_choices, rand)
        candidates = tuple((self._structure(value), weight) for value, weight in zip(values, weights))
        state = self._variant_states.get(candidates)
        if state is None:
            state = _DecayState(candidates, [len(self.RECOVERY_FACTORS) - 1] * len(values))
            self._variant_states[candidates] = state
        return [values[self._draw_index(rand, state)]]

    def _get_variant(self, command: VariantCommand, context: SamplingContext) -> ResultGen:
        # Keep ranged and multiple-choice selection on the original stateless
        # RandomSampler. Nested commands still route through the original context.
        if command.min_bound != 1 or command.max_bound != 1:
            yield from self._random_variant_sampler._get_variant(command, context)
        else:
            yield from super()._get_variant(command, context)

    def _get_wildcard(
        self,
        command: WildcardCommand,
        context: SamplingContext,
    ) -> ResultGen:
        wildcard = next(iter(context.sample_prompts(command.wildcard, 1))).text
        context = context.with_variables(command.variables)
        while True:
            # Usually a cached immutable collection. Re-read on each evaluation so
            # cache clearing or a changed collection safely resets row identities.
            values = context.wildcard_manager.get_values(wildcard)
            if not values:
                yield from get_wildcard_not_found_fallback(command, context)
                return
            state = self._get_state(context.wildcard_manager, wildcard, values)
            index = self._choose_index(context, state)
            yield from context.sample_prompts(str(values[index]), 1)
