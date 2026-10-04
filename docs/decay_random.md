# Decay Random (library API)

```python
from dynamicprompts.enums import SamplingMethod
from dynamicprompts.generators import RandomPromptGenerator
from dynamicprompts.wildcards import WildcardManager

generator = RandomPromptGenerator(
    wildcard_manager=WildcardManager("wildcards"),
    seed=123,
    default_sampling_method=SamplingMethod.DECAY_RANDOM,
)
print(generator.generate("__pose__, __color__", 10))
```

Ordinary wildcard draws and fixed single-select variants such as
`{red|blue|green}` use decay, including weighted variants. Multiple-choice and
ranged variants retain ordinary Random selection; nested commands still use
the context routing. No new syntax is added. Explicit `~`, `!`,
and `@` keep their existing meanings. Jinja-specific sampling is unchanged.

A draw uses `original_weight * factor`, then advances all recovering rows one
stage, then resets the selected row to 0.40. Stages are 0.40, 0.70, 0.90, 1.00.
Original weights are never mutated. Each draw calls the context RNG once with
`choices(..., k=1)`; no selections are prefetched.

History belongs to the sampler in the SamplingContext, shared by derived
contexts and successive generator calls. Create a new generator/context for a
new session. Reseeding an existing generator does not reset history. Reproduction
requires the same initial history, seed, candidate order and evaluation order.
Instances should not be shared across concurrent threads.

History keys use the manager instance and normalized evaluated wildcard name,
not filesystem paths: wildcard expressions may resolve to multiple collections.
Different names (including distinct glob patterns or recursive lookup aliases)
keep separate histories even if they happen to match the same file. Rows are
identified by index, including duplicate content if the manager's deduplication
is disabled. The manager's existing sorting/deduplication rules are preserved.

Candidates are fetched on each draw. Cache clearing and reloads are observed on
the next draw; changed row content, weight or order resets that wildcard history.
Reloading an identical list preserves history. Shuffled order changes can reset
history, so stable candidate order is recommended for this first version.
Variant history uses the ordered candidate AST structures and base weights.
Identical expressions share history across positions and generator calls;
changed content, order or weights get separate history. Candidate indices keep
duplicate options distinct. Nested variants have their own structural histories.
Wildcard and variant histories are separate, including when one contains the other.
Single literal options require no selection; the existing single-wildcard variant
conversion continues to use the variant selection path.
History is not stored in WildcardManager caches and is not process global.
