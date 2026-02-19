# 🧠 3DBrainAdapter 
### Native 3D Grounding for MedGemma: From Brain MRI to Clinical Report Draft

**Official Submission to the [Med-Gemma Impact Challenge**](https://www.kaggle.com/competitions/med-gemma-impact-challenge/overview)

**3DBrainAdapter** is an on-premise, privacy-preserving neuroradiology assistant. It transforms native **3D Brain MRIs** into structured, clinically grounded radiology report drafts using Google's HAI-DEF foundation models.

---

## 1. The Clinical Problem (Problem Domain)

Brain MRI interpretation is intrinsically volumetric. Neuro-oncologists assess tumor infiltration and edema across three spatial dimensions.
However, current Vision-Language Models (VLMs) process **2D slices**.

This 2D approximation leads to:

* ❌ Broken spatial continuity
* ❌ **Laterality inversion** (confusing Left/Right hemispheres)
* ❌ High rates of hallucinated lesions
* ❌ Radiologist burnout (up to 60% of time spent dictating reports)

---

## 2. Unlocking HAI-DEF to its Fullest Potential (Effective Use)

We built our solution upon Google’s **Health AI Developer Foundations**, bridging the gap between 2D foundations and 3D clinical needs.

* **Vision Encoder (MedSigLIP-448):** We apply a mathematical **weight inflation** strategy to upgrade MedSigLIP into a native 3D encoder without training from scratch.
* **Language Model (MedGemma-1.5-4B-IT):** Kept frozen to retain its exceptional medical reasoning.

*Fig 1. **3DBrainAdapter Architecture.** MedSigLIP is inflated to 3D. Volumetric tokens are compressed and projected into the frozen MedGemma LLM via soft-prompting.*

<img width="6576" height="2112" alt="MedGemma3D-Arch" src="https://github.com/user-attachments/assets/37217da3-91a0-4678-83bf-7e80a263f8f3" />

---

## 3. Staged Training & Feasibility (Product Feasibility)

To prevent OOM (Out-of-Memory) errors on standard hospital IT infrastructure, we use **Volumetric Token Compression** (reducing 3D patches to just 32 tokens) and **LoRA**.
To prevent the LLM from generating "caption-like" text, we align the modalities in 3 distinct stages:

*Fig 2. **Staged Learning.** (1) Contrastive latent alignment, (2) Projector Warmup, and (3) LoRA linguistic adaptation.*

<img width="6220" height="1444" alt="MedGemma3D-Arch2" src="https://github.com/user-attachments/assets/0fd86e6b-2ec5-443e-8a88-c1f3460a8db8" />

---

## 💡 4. Anticipated Clinical Impact (Impact Potential)

By moving from a slice-based approach to native 3D grounding, our framework drastically improves diagnostic factualness.

### Quantitative Performance (BraTS Dataset)

We benchmarked our framework against state-of-the-art 3D generalists and the standard 2D slice-based MedGemma.

| Model | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr | Lat F1 | Anat F1 | **Path F1** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Med3DVLM** *(3D Generalist)* | 0.051 | 0.005 | 0.083 | 0.007 | 0.300 | 0.225 | 0.119 |
| **MedGemma 1.5** *(2D Slice-based)* | 0.245 | 0.024 | 0.189 | 0.029 | 0.526 | 0.461 | 0.413 |
| **3DBrainAdapter** *(Ours)* | **0.302** | 0.098 | **0.289** | 0.293 | **0.689** | **0.691** | **0.951** |

*Note: The **+130% gain in Pathology F1** over the strong 2D MedGemma baseline proves that volumetric modeling is non-negotiable for diagnostic factualness.*

### Qualitative Results

Our model achieves **zero hallucinations on healthy controls** and correctly resolves complex spatial relationships that confuse 2D baselines.

*Fig 3. **Qualitative Comparison.** 3DBrainAdapter correctly identifies the lesion location and pathologies, whereas baselines hallucinate or fail.*

<img width="1598" height="530" alt="Screenshot 2026-02-19 121304" src="https://github.com/user-attachments/assets/4b9301f8-6053-4e65-b898-613204b497bc" />

---

## 5. Execution & Reproducibility (Execution)
We provide a complete, transparent, and reproducible package.
[DA INSERIRE HUGGING FACE, COME USARE IL CODICE, COME USARE LA DEMO, IMMMAGINI DELLA DEMO

* 🎥 **[Watch the Video Demo Here](https://www.google.com/search?q=%23)** *(UI & Human-in-the-loop workflow)*
* 📓 **[Kaggle Notebook](https://www.google.com/search?q=%23)** *(End-to-end inference code)*
* 📄 **[Technical Write-up / Research Paper](https://www.google.com/search?q=%23)** *(Detailed methodology)*

### 🌐 **Notes**
This project was developed by Mariano Barone, Francesco Di Serio, Giuseppe Riccio, Antonio Romano, Marco Postiglione, and Vincenzo Moscato  
*University of Naples, Federico II*

*Built with ❤️ for the Med-Gemma Impact Challenge.*
