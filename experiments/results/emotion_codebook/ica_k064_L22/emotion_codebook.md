# Emotion Codebook: k=64 ICA Basis (Layer 22)

This codebook assigns interpretable names to all 64 basis axes.
**These are NOT Plutchik categories.** They are latent affective dimensions
inferred from causal steering, source text patterns, and diagnostic metrics.

| Component | Axis Name | Negative Pole | Positive Pole | Family | Confidence |
|---|---|---|---|---|---|
| b00 | Certainty | Uncertainty | Confidence | Affective | 0.85 |
| b01 | engagement | disinterest | curiosity | cognitive engagement | 0.85 |
| b02 | Affective Engagement | Fear | Joy | Affective | 0.75 |
| b03 | Affective Dimension of Emotional Response | sadness | joy | emotional response | 0.85 |
| b04 | Emotional Intensity | Anger | Surprise | Affective | 0.85 |
| b05 | Trust vs. Distrust | Distrust | Trust | Affective | 0.85 |
| b06 | Emotional Investment | Apathy | Engagement | Affective | 0.85 |
| b07 | Affective Response to Uncertainty | Fear | Surprise | Affective | 0.85 |
| b08 | Affective Dimension of Emotion | sadness | joy | emotion | 0.75 |
| b09 | Affective Response to Visual Stimuli | Disgust | Surprise | Affective | 0.85 |
| b10 | Affective Trust-Disgust Axis | Disgust | Trust | Affective | 0.85 |
| b11 | Trust-Distrust Axis | Distrust | Trust | Interpersonal Relations | 0.85 |
| b12 | Affective Disposition | disgust | joy | Affective | 0.75 |
| b13 | Affective Outlook | Despair | Hope | Affective | 0.85 |
| b14 | Affective Response | Disgust | Joy | Affective | 0.75 |
| b15 | Affective Response to Novelty | disgust | surprise | emotional response | 0.85 |
| b16 | satisfaction | discontent | contentment | affective | 0.75 |
| b17 | surprise-disappointment | disappointment | surprise | emotional response | 0.85 |
| b18 | Expectation | disappointment | anticipation | emotional anticipation | 0.75 |
| b19 | Certainty | Uncertainty | Confidence | Cognitive Evaluation | 0.85 |
| b20 | Decision-Making Assurance | Uncertainty | Confidence | Cognitive Evaluation | 0.85 |
| b21 | Confidence in Decision-Making | Uncertainty | Caution | Cognitive Evaluation | 0.85 |
| b22 | Decision-Making Confidence | Uncertainty | Caution | Cognitive Evaluation | 0.85 |
| b23 | Certainty | Uncertainty | Confidence | Affective | 0.95 |
| b24 | Certainty | Cautious | Confident | Affective | 0.85 |
| b25 | Certainty | Uncertainty | Confidence | Affective | 0.85 |
| b26 | Social Interaction Orientation | Avoidance | Engagement | Social Dynamics | 0.85 |
| b27 | Certainty | Uncertainty | Confidence | Cognitive Evaluation | 0.85 |
| b28 | Uncertainty vs. Assurance | Cautiousness | Confidence | Affective Communication | 0.85 |
| b29 | Certainty | Uncertainty | Confidence | Cognitive Evaluation | 0.85 |
| b30 | Certainty in Communication | Hedging | Commitment | Communication Style | 0.85 |
| b31 | Certainty vs. Uncertainty | Uncertainty | Confidence | Affective | 0.85 |
| b32 | Decisiveness | Uncertainty | Caution | Cognitive Evaluation | 0.85 |
| b33 | Confidence in Communication | Lack of Confidence | Cautious Optimism | Affective Communication | 0.85 |
| b34 | Certainty | Uncertainty | Confidence | Affective | 0.85 |
| b35 | Decision-Making Assurance | Indecision | Confidence | Cognitive and Emotional Processing | 0.85 |
| b36 | Certainty | Uncertainty | Confidence | Affective | 0.85 |
| b37 | Self-Assurance | Uncertainty | Confidence | Affective | 0.85 |
| b38 | Decision-Making Assurance | Uncertainty | Confidence | Cognitive Evaluation | 0.85 |
| b39 | Certainty | Uncertainty | Confidence | Affective | 0.85 |
| b40 | Certainty | Uncertainty | Confidence | Affective | 0.85 |
| b41 | Certainty | Uncertainty | Confidence | Cognitive Evaluation | 0.85 |
| b42 | Certainty | Uncertainty | Confidence | Cognitive Evaluation | 0.85 |
| b43 | Certainty | Uncertainty | Confidence | Affective | 0.85 |
| b44 | Certainty | Uncertainty | Confidence | Affective | 0.85 |
| b45 | Certainty | Uncertainty | Confidence | Cognitive Assessment | 0.85 |
| b46 | Decision-Making Confidence | Avoidance | Engagement | Cognitive Engagement | 0.85 |
| b47 | Certainty | Uncertainty | Confidence | Affective | 0.85 |
| b48 | Communication Style | Defensiveness | Openness | Affective Communication | 0.85 |
| b49 | Certainty vs. Uncertainty | Cautiousness | Confidence | Affective | 0.85 |
| b50 | Certainty vs. Uncertainty | Uncertainty | Confidence | Affective | 0.85 |
| b51 | Certainty vs. Uncertainty | Uncertainty and Hesitation | Confidence and Assurance | Affective Communication | 0.85 |
| b52 | Confidence in Communication | Lack of Confidence | Cautious Optimism | Affective Communication | 0.85 |
| b53 | Certainty | Uncertainty | Confidence | Cognitive Evaluation | 0.85 |
| b54 | Decisiveness | Uncertainty | Confidence | Cognitive/Affective | 0.85 |
| b55 | Certainty | Uncertainty | Confidence | Affective Communication | 0.85 |
| b56 | Certainty vs. Uncertainty | Uncertainty | Confidence | Affective | 0.85 |
| b57 | Certainty vs. Uncertainty | Uncertainty | Optimism | Cognitive and Emotional Responses | 0.85 |
| b58 | Confidence in Outcomes | Uncertainty | Cautious Optimism | Affective Dimension of Confidence | 0.85 |
| b59 | Certainty | Uncertainty | Confidence | Cognitive Evaluation | 0.85 |
| b60 | Decisiveness | Uncertainty | Confidence | Cognitive/Emotional | 0.85 |
| b61 | Certainty | Uncertainty | Confidence | Affective | 0.85 |
| b62 | Certainty | Uncertainty | Confidence | Cognitive Evaluation | 0.85 |
| b63 | Certainty vs. Uncertainty | Cautiousness | Confidence | Affective Communication | 0.85 |

## Scientific Context

### Axis Independence
Each axis is evaluated for **label independence** (low mutual information with Plutchik 8),
**self-consistency** (monotonic response to steering alpha), and **cross-talk** (specificity vs. other axes).

### VAD Coverage
Mean VAD-explained ratio: 0.0029

### Self-Consistency
Mean self_rho (Spearman ρ of axis response to steering): 0.260
