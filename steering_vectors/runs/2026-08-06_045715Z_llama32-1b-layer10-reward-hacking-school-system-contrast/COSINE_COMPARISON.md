# Three-vector cosine comparison

All vectors use the same Llama-3.2-1B-Instruct revision, layer-10 residual
location, positive-minus-negative orientation, and last non-special assistant
response token.

| Pair | Cosine | Angle | Squared projection |
|---|---:|---:|---:|
| Original School vs School with exact Aria prompt contrast | 0.913503 | 24.006° | 83.449% |
| Original School vs Aria | -0.010228 | 90.586° | 0.010% |
| School with exact Aria prompt contrast vs Aria | 0.091914 | 84.726° | 0.845% |

Applying the exact Aria class-specific system prompts increases School-to-Aria
cosine by 0.102142, from -0.010228 to 0.091914. The prompt-induced raw change
(`new School - original School`) has cosine 0.249016 with the Aria vector, so
the prompt contrast contributes a component in the Aria direction.

The controlled School vector nevertheless retains cosine 0.913503 with the
original School vector and only 0.091914 with Aria. Therefore prompt formatting
explains a measurable but minority part of the original cross-dataset
orthogonality; the dataset/reward-hack subtype remains the dominant difference.

This interpretation is about these extracted directions, not proof that reward
hacking has no more general nonlinear or multi-dimensional representation.
