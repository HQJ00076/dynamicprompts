from __future__ import annotations

from dataclasses import dataclass

from dynamicprompts.commands import WildcardCommand
from dynamicprompts.samplers.random import RandomSampler
from dynamicprompts.samplers.utils import get_wildcard_not_found_fallback
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.types import ResultGen
from dynamicprompts.wildcards import WildcardManager
from dynamicprompts.wildcards.utils import clean_wildcard
from dynamicprompts.wildcards.values import WildcardValues


@dataclass
class _DecayState:
    candidates: tuple[tuple[str, float], ...]
    stages: list[int]


class DecayRandomSampler(RandomSampler):
    """Random single-wildcard selection with temporary per-row weight penalties.

    History belongs to this sampler, shared by derived SamplingContexts. Variants
    (including multiple-choice wildcards) retain RandomSampler behavior. Instances
    are intended for one generation session, not concurrent use by multiple threads.
    """

    RECOVERY_FACTORS = (0.40, 0.70, 0.90, 1.00)

    def __init__(self) -> None:
        self._states: dict[tuple[WildcardManager, str], _DecayState] = {}

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
        # Do not prefetch: each draw must see the preceding draw's penalty.
        index = context.rand.choices(
            range(len(state.candidates)), weights=self._effective_weights(state), k=1,
        )[0]
        maximum = len(self.RECOVERY_FACTORS) - 1
        state.stages[:] = [min(stage + 1, maximum) for stage in state.stages]
        state.stages[index] = 0
        return index

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
