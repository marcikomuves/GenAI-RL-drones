### According to the SynthER paper (https://arxiv.org/pdf/2303.06614):
- SynthER can augment static datasets and is highly effective for small datasets in ***Offline*** settings.
- SynthER can increase the UTD (*update-to-data*) ratio (sample efficiency) in ***Online*** settings by augmenting datasets collected through interaction with the environment:
	- SAC+SynthER can achieve UTD = 20
		- After every simulation step, i.e. every collected real transition, 20 gradient steps can be performed on the Critic networks.
			- For the original SAC, UTD = 1.
			- Synthetic data stabilizes gradient updates, eliminating the need for a large Q ensemble.
	- SAC+SynthER outperforms REDQ SAC.
	- Implicitly stated: SynthER results in a substantial performance increase over the *unmodified* version of SAC.
### The SynthER paper does not mention:
- SAC+LayerNorm can also achieve UTD = 20.
- SAC+LayerNorm outperforms REDQ SAC.
### Question:
How does **SAC+LayerNorm** perform compared with **SAC+SynthER**?
### Goal:
- Compare whether, in ***Online*** settings, augmenting with a diffusion model is worthwhile instead of using LayerNorm:
	- What is LayerNorm useful for? Demonstrate this by comparing SAC and SAC+LayerNorm at UTD = 1.
	- Compare SAC+LayerNorm, SAC+SynthER, and REDQ SAC at UTD = 20.
### Experimental training runs:

|                     | SAC (UTD=1) | SAC (UTD=20) | SAC+SynthER (UTD=20) | REDQ SAC (UTD=20) |
| ------------------- | ----------- | ------------ | -------------------- | ----------------- |
| **LayerNorm**       | ✅           | ✅            | -                    | -                 |
| **No LayerNorm**    | ✅           | -            | ✅                    | ✅                 |
