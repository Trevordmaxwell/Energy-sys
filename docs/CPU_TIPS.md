# CPU Survival Tips

- Prefer batch sizes of 1–4 and accumulate gradients for larger effective batches.
- Keep sequence lengths at or below 512 tokens for demonstrations.
- Limit memory patterns to `K <= 4096` when using exact retrieval.
- Cap solver iterations to 10–15 with damping and energy checks enabled.
- Monitor iteration counts; reduce β or causal band size if they spike.
