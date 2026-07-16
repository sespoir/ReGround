# Validation run

Validation was repeated on 2026-07-16 with vLLM 0.11.2 V1 and one NVIDIA
A100 80 GB GPU. The selected SFT checkpoint is represented by the portable
placeholder path below:

```text
/tmp/reground/models/qwen2.5-vl-reground
```

The checkpoint was served as `qwen2_5-vl-reground` through an isolated vLLM
endpoint. A focused integration run and the full benchmark used separate
server jobs, and both were stopped after validation to release the GPU.

## Static and offline contract validation

- Shell and Python 3.10 syntax checks passed.
- The secret scan and `git diff --check` passed.
- Two offline tests passed both with the standalone source loader and with the
  pinned VLMEvalKit commit `2c25371d602909ae3d6d395185aff1bc9493262d`.
- The tests verify that `<reground>` causes a second request and that Round 2
  receives a byte-identical copy of the Round-1 image payload.

## Natural-trigger probe

Five training-set examples labeled with `<reground>` were replayed with their
original images and questions. All five completed normally with a direct
`<answer>` and none emitted `<reground>`. An additional easy prompt that
explicitly requested the marker also answered directly. The later full
HallusionBench run did trigger naturally, so this small probe was not
representative of the checkpoint's overall trigger behavior.

A pre-release audit of the local training JSONL found 13,149 `<reground>`
opening tags, no matching closing tags, and 13,148 legacy closing tags. The
inference trigger depends only on the opening marker, but future dataset
releases and training runs should normalize the closing tags.

## Forced two-round integration test

The explicit `--force-reground` diagnostic completed two real HTTP requests
with the Hungary insurance-market chart:

- Round-1 answer: `36`
- Round-2 answer: `36`
- API requests: 2
- Prompt tokens: 2,186
- Completion tokens: 546
- Total tokens: 2,732
- vLLM reported a multimodal cache hit rate of 57.1% after the request pair.

The forced test proves that the checkpoint, two-round conversation, repeated
image placement, answer extraction, and vLLM multimodal cache path work
together. It does not count as a natural-trigger success.

## Full HallusionBench benchmark

The evaluation completed all 951 HallusionBench records with 32 API workers and
the `exact_matching` judge. Inference took 6 minutes 4 seconds; the complete
scheduled job took 6 minutes 16 seconds and exited with code 0.

The adapter observed 154 natural `<reground>` triggers (16.19% of the 951
benchmark records), and all 154 second-round calls returned non-empty outputs.
There were 952 first-round calls because one length-truncated empty answer was
automatically retried once by VLMEvalKit. This produced 1,106 HTTP requests in
total. Token usage was 1,424,856 prompt tokens and 333,477 completion tokens,
for 1,758,333 tokens overall. At the end of the run, vLLM reported a 73.8%
multimodal cache hit rate.

The official HallusionBench aggregate scores were:

| split | aAcc | fAcc | qAcc |
| --- | ---: | ---: | ---: |
| Overall | 72.8707 | 51.4451 | 52.7473 |
| VD | 68.6971 | 49.1304 | 44.0433 |
| VS | 79.7222 | 56.0345 | 66.2921 |

Exact matching extracted 693 correct answers out of 951. Nineteen predictions
could not be normalized to Yes/No and were scored as `Unknown`; there were no
unrecovered API failures. The full score CSV, prediction workbook, auxiliary
matching workbook, token log, and per-round output log are retained in ignored
local output directories.

This full result establishes that the selected SFT checkpoint can trigger and
complete the natural two-round reground path. The malformed mixed closing tags
in the training JSONL remain a data-quality issue and should still be corrected
for a future training run.
