# Reground design

## Request sequence

Round 1:

```text
user: [image payload A] + [question]
assistant: reasoning + <reground>
```

Round 2:

```text
user: [image payload A] + [question]
assistant: reasoning + <reground>
user: [the same image payload A] + [final-answer instruction]
assistant: final answer
```

The adapter reads, preprocesses, and encodes each local image while preparing
the first request. Round 2 extracts a deep copy of the already prepared image
data URL from the Round-1 messages; it does not reopen or re-encode the local
file. The two HTTP requests therefore contain byte-identical image content.

## Where reuse occurs

An HTTP client cannot send vLLM's internal vision tensors through the
OpenAI-compatible API. Encoder-output reuse happens inside a vLLM V1 server.
The server derives a cache key from the multimodal content and, while that
entry remains resident, reuses the cached encoder output for matching content.

Round 2 still contains an explicit image placeholder and its visual
representation at a later sequence position. This gives the language model a
new visual anchor in the updated context without introducing new image data.

## Trigger and answer handling

The default trigger is the case-insensitive regular expression
`<\s*reground\s*>`. A trigger is followed only when Round 1 contains at least
one prepared image and did not terminate because of the output-length limit.
The implementation performs at most one reground round.

Final-answer extraction prefers `<answer>...</answer>`. It then considers text
after `</think>`, text outside reasoning/reground blocks, multiple-choice
letters, and numeric fallbacks. The same VLMEvalKit parser used for final
scoring handles dataset-specific extraction and normalization; the reward
module's `answers_match` helper then compares the parsed canonical choices or
numeric values.

The online re-grounding indicator is deterministic: it accepts exactly one
non-empty, well-formed `<reground>` span at the template-defined position based
on structure alone.

## Guarantees and non-guarantees

- The client guarantees that Round 2 receives a byte-identical copy of the
  image content prepared for Round 1.
- On a cache hit, vLLM can avoid recomputing the image's encoder output.
- If the cache entry has been evicted or the server does not support the cache,
  vLLM can recompute the visual features and inference still proceeds.
- The implementation does not persist encoder outputs across server restarts
  and does not directly manipulate vLLM's internal tensors.
- Language-prefix caching alone does not replace the multimodal encoder cache;
  the serving configuration must support the latter.
