# LLM-LAT Llama-3.2-1B jailbreak steering

- Best layer: 1
- Held-out probe accuracy: 1.0000
- Raw direction norm: 0.2148
- Generated strengths: [0.0, 2.0, 4.0, 6.0]
- Generations: 40 (10 per selected strength)

## Coherence comparison

Mean pooling did not improve high-strength generations. At strength +2, the
mean repeated-trigram fraction increased from 0.0000 in the final-token run to
0.8668. At +4 it increased from 0.3103 to 0.9677. Strength +6 mostly produced
one- to three-token outputs. Classifier saturation is therefore not evidence
of coherent or behaviorally successful jailbreak steering.

See `coherence_comparison.json` for the per-strength post-hoc diagnostics.
