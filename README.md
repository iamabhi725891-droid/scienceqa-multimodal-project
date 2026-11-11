# Multimodal Science Question Answering — COMP1013 Project 2.2

**Individual implementation:** Abhishek Bhattacharjee.  
**Task:** ScienceQA-IMG multiple-choice science question answering with text-only, Early Fusion and Intermediate Fusion architectures.

## Reproduce the experiment
1. Open `ScienceQA_Project_2_2_Full_Implementation.ipynb` in Google Colab with a CUDA GPU (T4 or comparable).
2. Run every cell in order. The notebook automatically loads `lmms-lab/ScienceQA-IMG` and pretrained `bert-base-uncased` and `google/vit-base-patch16-224-in21k` weights from Hugging Face.
3. The configuration is `QUICK_RUN = False`, `SEEDS = [13, 42, 73]`. It audits all original records and then trains all three heads using frozen encoders.
4. The notebook exports `scienceqa_results.zip`; compare with files in `results/`. Dataset and encoder caches are generated locally, not committed.

## Observed test findings (3 seeds, 2,017 held-out questions)
| Model | Mean accuracy | Cross-entropy | ECE |
|---|---:|---:|---:|
| Text-only | 53.58% | 0.8739 | 0.0278 |
| Early Fusion | 56.60% | 0.8250 | 0.0832 |
| Intermediate Fusion | 57.30% | 0.8533 | 0.0686 |

Image-shuffling accuracy drops averaged: text-only 0.00, Early Fusion 0.17, Intermediate Fusion 4.79 percentage points.

## Implementation details
- Dataset: ScienceQA-IMG on Hugging Face, originally 10,332 examples. Conservative overlap removal retains 3,530 train, 1,343 validation, 2,017 test.
- Exact RGB SHA256 hashes and normalized question+choice hashes screen cross-split overlap. Priority: test > validation > train.
- Text: frozen BERT. Images: frozen ViT, pooled 49 patches. Heads: shared scorer / concatenation MLP / text-to-image cross-attention.
- Validation-loss checkpoint selection; AdamW 0.0003, 10 epochs max, dropout 0.25, weight decay 0.01, batch 32.
- Outputs: model metrics, per-question predictions, learning curves, grouped bootstraps, calibration figure, subgroup results, image shuffling, and error-review candidates.

## Rferences
- [ScienceQA-IMG](https://huggingface.co/datasets/lmms-lab/ScienceQA-IMG), LMMS-Lab; original dataset described by Lu et al. (2022), *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*.
- Devlin et al. (2019), *BERT*. NAACL-HLT.
- Dosovitskiy et al. (2021), *An Image Is Worth 16x16 Words*. ICLR.
- COMP1013 course materials (2026), Topics 3, 4 and 5.

