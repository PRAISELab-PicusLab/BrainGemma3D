# BrainMedGemma3D
### From 3D Brain MRI to Clinically Grounded Report Draft
Submission to the Med-Gemma Impact Challenge https://www.kaggle.com/competitions/med-gemma-impact-challenge/overview

---
## What This Is
BrainMedGemma3D is an on-premise neuroradiology assistant that transforms a **3D brain MRI** into a structured radiology report draft.
The radiologist reviews, edits if needed, and signs.

_Human-in-the-loop.  
Privacy-preserving.  
Volumetrically grounded._

---

## Foundation Models (Challenge-Compliant)

Built directly on Google’s Health AI Developer Foundations:

- **MedSigLIP-448** (vision encoder)  
- **MedGemma 4B 1.5** (language model)  

Source:  
https://huggingface.co/collections/google/health-ai-developer-foundations-hai-def

We strictly use official HAI-DEF models as required by the competition.

---

## The Clinical Problem

Brain MRI is volumetric.

Slice-based inference:
- Breaks spatial continuity  
- Causes laterality errors  
- Encourages hallucinated findings  

Radiologists spend significant time drafting reports and ensuring spatial correctness.

---

## Our Contribution: 3D Grounding

We extend MedSigLIP and MedGemma to native 3D:

- 2D → 3D weight inflation  
- Volumetric token compression  
- Grounding projector into MedGemma  
- LoRA fine-tuning for domain specialization  

The MedGemma backbone remains frozen.

---

## Workflow

3D MRI  
↓  
Volumetric grounding  
↓  
Structured report generation  
↓  
PDF draft  
↓  
Radiologist validation & signature  

---

## Quantitative Comparison

Using the same HAI-DEF foundation models:

| Model | Pathology F1 | Laterality Accuracy | Hallucination Rate |
|-------|--------------|--------------------|--------------------|
| Slice-based MedGemma | 0.41 | 0.72 | 18% |
| **BrainMedGemma3D** | **0.95** | **0.85** | **4%** |

+130% improvement in pathology detection  
4× reduction in hallucinations  

The improvement comes from volumetric grounding.

---

## Reproducibility

- Fully open-weight
- On-premise compatible
- Kaggle notebook included
- Deterministic configs provided

---

Built for the Med-Gemma Impact Challenge.
