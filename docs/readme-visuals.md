# README artwork sources

The README artwork uses results and an example from [SIM-CoT, arXiv v2](https://arxiv.org/abs/2509.20317). The original paper figures remain in `assets/`.

| Artwork | Source and meaning |
| --- | --- |
| `assets/readme/hero.png` | Schematic step supervision; accuracy gains from Tables 1 and 3. `pp` means percentage points. The decoder is removed for ordinary inference. |
| `assets/readme/latent-walkthrough.gif` | Explanatory animation of the watermelon example from Figure 4 and Appendix I.1. It shows ordinary inference followed by optional interpretation. It is not a recorded model run or a latency measurement. |
| `assets/readme/latent-walkthrough.png` | Static final frame of the same animation. The equations are the published decoded steps. |
| `assets/readme/results-overview.png` | Paired GSM8k-Aug accuracy comparisons from Tables 1–3. Each pair uses the named baseline and backbone. The axis starts at zero. |
| `assets/readme/results-overview.svg` | Vector export of the results chart. |

Appendix I.1 describes four distinct latent steps for the watermelon example. Figure 4 merges the two middle steps into a single displayed box to save space. The animation shows all four steps separately, using the equations stated in Appendix I.1. Its `z1`–`z4` labels are schematic latent labels, not a specification for all training configurations. The hero's three nodes are a generic method schematic. The GIF summarizes the question without changing its quantities or answer.

The 2.3× token efficiency comparison is 27.6 / 12.2, rounded, on GPT-2 in Table 1. It compares average token counts against SFT-CoT; it is not a measured runtime speedup. Larger LLaMA models still have lower in-domain accuracy than SFT-CoT, so the README does not claim universal superiority over explicit CoT.

Values and provenance are recorded in [`source.json`](../assets/readme/source.json). To regenerate the artwork, install Pillow, NumPy and Matplotlib and run:

```bash
python assets/readme/make_visuals.py
```

The script uses Lato and DejaVu Sans Mono from standard Linux font paths; adjust its font paths for another system. No training, evaluation code, model checkpoint or dataset was changed for this presentation update.
