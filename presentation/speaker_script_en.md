# English Speaker Script — Defense (13-slide deck)

**Author:** Nguyen Trong Bach — 23BI14057
**Target:** ~10 minutes of talk + 15 minutes of Q&A
**Aligned to:** `slides_en.pdf` after the 2026-07-13 restructure (added a
Preprocessing & Augmentation slide, moved the Training Strategy slide to
appendix A28)

Read this once, then **speak the ideas, not the sentences**. Every slide has a
target duration; if you fall behind, drop the italic sentences first.

---

## Pacing at a glance

| Slide | Content                                         | Cumulative |
|:-----:|-------------------------------------------------|:----------:|
| 1     | Title                                            | 0:20       |
| 2     | Outline                                          | 0:35       |
| 3     | Problem & Objectives                             | 1:35       |
| 4     | Data & Challenges                                | 2:35       |
| 5     | Pipeline overview                                | 3:10       |
| 6     | Preprocessing & Augmentation                     | 4:00       |
| 7     | Four architectures                               | 4:40       |
| 8     | Single models + Ensemble                         | 5:50       |
| 9     | Confusion matrix                                 | 6:30       |
| 10    | Ablation                                         | 7:25       |
| 11    | Grad-CAM + Streamlit                             | 8:20       |
| 12    | Conclusion + Future Work                         | 9:00       |
| 13    | Questions and Discussion                         | 9:20       |

Aim to finish by 9:30, leaving ~30 seconds of buffer. If time is tight, cut
the italic sentences first.

---

## Slide 1 — Title (20 s)

> "Good morning, distinguished members of the committee.
> My name is Nguyen Trong Bach, from cohort K23 of the University of Science
> and Technology of Hanoi, ICT department. Today I would like to present my
> Bachelor thesis titled *A Comparative Evaluation of Deep Learning Models for
> Skin Lesion Classification on Public Dermoscopic Datasets*, supervised by
> Dr. Vu Trong Sinh externally and Dr. Nghiem Thi Phuong internally."

**Delivery:** stand straight, look at the committee, do not read the title.

---

## Slide 2 — Outline (15 s)

> "The talk has six parts: introduction, data and challenges, methodology,
> experimental results, ablation, and interpretability with the application.
> I will keep the main talk under ten minutes to leave time for questions."

**Delivery:** point briefly to the six sections, do not read them one by one.

---

## Slide 3 — Problem & Objectives (60 s)

> "Melanoma is the deadliest form of skin cancer — yet when caught early, more
> than ninety-nine percent of patients survive past five years. Dermoscopy lets
> doctors examine a lesion up close, but the final decision still depends
> heavily on the dermatologist's experience.
>
> Any AI system trying to assist that decision faces three barriers: severe
> data imbalance, high morphological similarity between melanoma and common
> nevi, and the black-box nature of deep networks — a doctor cannot trust a
> decision they cannot inspect.
>
> This thesis makes four contributions: an imbalance-aware training pipeline,
> a fair comparison of three CNNs and one Transformer, a soft-voting ensemble
> that beats every single model, and Grad-CAM together with a Streamlit
> prototype so clinicians can see where the model is looking."

**Delivery:** the 99 % figure is memorable — pause for half a second after it.
The four contributions are the roadmap for the rest of the talk.

---

## Slide 4 — Data & Challenges (60 s)

> "The dataset is ISIC 2018 Task 3, also known as HAM10000: ten thousand
> fifteen dermoscopy images across seven lesion classes, split 70/15/15 in a
> stratified manner — meaning the class ratios in train, val and test all
> match the original distribution.
>
> The critical property of this dataset is its extreme imbalance: melanocytic
> nevi make up sixty-seven percent of all images, while dermatofibroma sits
> at only one point one percent. The ratio between the largest and smallest
> class is fifty-eight to one.
>
> This matters, because a lazy model can predict 'nevus for everything' and
> still reach sixty-seven percent accuracy — clinically useless. That is why
> we optimise Macro F1 and Macro AUC instead, so every class counts equally.
>
> Four mechanisms address the imbalance directly: a Weighted Sampler to
> balance each mini-batch, Mixup and CutMix to smooth decision boundaries,
> Label Smoothing to reduce over-confidence, and Macro F1 as the criterion
> for selecting the best checkpoint. *The formulas behind these four
> techniques are in appendix A28.*"

**Delivery:** the 58:1 number is the punchline — slow down when you say it.

---

## Slide 5 — Pipeline Overview (35 s)

> "This diagram summarises the end-to-end pipeline in one picture: a raw
> dermoscopy image goes through preprocessing and augmentation, then one of
> the four ImageNet-pretrained backbones, then a classifier head that outputs
> seven class probabilities.
>
> The next slide zooms into the first two steps — preprocessing and
> augmentation — before I move on to the four architectures."

**Delivery:** point at each block as you name it. Do not go deeper — details
follow on the next slide.

---

## Slide 6 — Preprocessing & Augmentation (50 s)

> "Before any image reaches the model, it passes through two stages.
>
> Preprocessing applies to every split — simply resizing to two-twenty-four
> by two-twenty-four and normalising with ImageNet statistics, since all four
> backbones are ImageNet-pretrained.
>
> Augmentation is different — it only applies to the training set, and it
> runs fresh every epoch, nothing is precomputed. The reason: some classes
> have only a few dozen images — Dermatofibroma has just eighty-one training
> images — so training on the exact same pixels over and over would make the
> model memorise those specific images instead of learning the underlying
> pathology.
>
> So every epoch, each image is randomly transformed: rotated, flipped
> horizontally and vertically, cropped and resized, jittered slightly in
> colour and brightness, softly blurred, and partially erased. The figure
> below is generated from the project's actual code — each panel is one
> isolated technique.
>
> As a result, the model never sees the exact same image twice during the
> entire training run."

**Delivery:** this slide has a figure — give the committee 2-3 seconds to
look at it before your closing sentence. If asked why no strong geometric
warping (shear, perspective) is used, answer: the lesion's shape and border
irregularity is itself a diagnostic signal, so it should not be distorted.

---

## Slide 7 — Four Architectures (40 s)

> "Four backbones are compared. EfficientNet-B0 with only five point three
> million parameters — a compact CNN that captures fine local features.
> ResNet50 with twenty-five million — a deeper classical CNN. DenseNet121
> with eight million — a CNN that exploits feature reuse. And Swin-Tiny with
> twenty-eight million — a Vision Transformer providing global context.
>
> All four are fine-tuned from ImageNet weights through the timm library.
> The mix of CNN and Transformer architectures is deliberate — it gives the
> ensemble uncorrelated errors to exploit."

**Delivery:** emphasise the word *uncorrelated* — it is the reason the
ensemble works. Keep this slide brief, do not read the whole table.

---

## Slide 8 — Single Models and the Ensemble (70 s)

> "Here are the headline numbers on the test set.
>
> The four single models cluster between eighty-five and eighty-eight percent
> accuracy. EfficientNet-B0 leads the CNNs on Macro F1 despite being by far
> the smallest — evidence that architecture matters more than raw parameter
> count on limited medical data.
>
> The last row is the punchline: the soft-voting ensemble reaches
> **eighty-nine point four nine percent accuracy, zero point eight four five
> Macro F1, and zero point nine eight zero AUC** — clearly better than any
> single model.
>
> To make sure this was not a lucky single seed, I repeated the entire
> pipeline across five random seeds. The ensemble stays at
> **eighty-nine point nine three percent, plus or minus zero point six six**.
> McNemar's paired test confirms the ensemble beats every single backbone
> with p less than ten to the minus seven against the strongest model.
>
> Soft voting works here because the CNN family captures local textures
> while Swin captures global structure, so their errors are largely
> uncorrelated. Averaging four uncorrelated probability vectors reduces
> variance — no extra training, four inference passes."

**Delivery:** this is the **first "wow moment"** — pause for one full second
after the McNemar p-value. Look at the committee, not the slide.

---

## Slide 9 — Confusion Matrix (40 s)

> "The confusion matrix confirms three important properties.
>
> First, the diagonal is strong across every class — recall is not sacrificed
> on the rare classes. Dermatofibroma and Vascular, which together have fewer
> than two hundred training images, are still recognised accurately — a
> direct result of Mixup plus the Weighted Sampler.
>
> Second, the residual confusion sits in the Melanoma, Nevus and
> Benign-Keratosis cluster — three classes that genuinely share morphology.
>
> Third, and most important: no class collapses. The classic failure mode of
> imbalanced classification does not happen here."

**Delivery:** use the pointer, walk the committee across the diagonal
briefly.

---

## Slide 10 — Ablation (55 s)

> "To measure the contribution of each component, I removed one at a time
> from the EfficientNet-B0 baseline and re-trained.
>
> The full pipeline reaches Macro F1 zero point eight three one. Removing
> Mixup and CutMix drops F1 by four point five percent — this is by far the
> largest single lever, because with only eighty-one DF images the model
> immediately memorises the rare samples without them.
>
> Removing Label Smoothing raises AUC but drops F1 by three point one percent
> — a textbook trade-off between calibration and hard decisions.
>
> Removing strong augmentation actually raises Accuracy slightly, but Macro
> F1 drops — the minority classes are being abandoned. This is exactly why
> Accuracy is not the primary criterion.
>
> Every component contributes positively to Macro F1 — no module is
> redundant."

**Delivery:** the phrase *"Accuracy is not the primary criterion"* is a key
methodological statement — slow down when saying it.

---

## Slide 11 — Grad-CAM + Streamlit (55 s)

> "Interpretability closes the loop from model back to clinician. Grad-CAM
> produces a heatmap of the last convolutional layer, weighted by the
> gradient with respect to the predicted class — effectively showing what
> pixels made the model decide.
>
> On this melanoma sample, all four models attend to the central pigmented
> region rather than to the background, the hair, or the gel bubbles. And
> critically, I do not stop at qualitative inspection: Grad-CAM focus-ratio
> is above zero point six for every backbone — a quantitative validation.
>
> The Streamlit prototype puts this into a live tool: the clinician uploads
> an image and receives seven-class probabilities plus a Grad-CAM heatmap in
> under two seconds. The user can switch which model classifies the image —
> the ensemble or any single backbone — and which model is shown in the
> heatmap."

**Delivery:** if asked whether the tool could replace a clinician, answer
clearly: it is a reference/support tool, not a replacement for medical
diagnosis. Do not let it be understood as a standalone diagnostic system.

---

## Slide 12 — Conclusion + Future Work (40 s)

> "To summarise the five key contributions: the ensemble reaches
> eighty-nine point four nine percent Accuracy and zero point nine eight
> AUC — SOTA level for image-only ISIC 2018 methods. It is robust across
> seeds, with p less than ten to the minus seven. Every ablated component
> contributes positively. Grad-CAM focus-ratio above zero point six confirms
> the model attends to the lesion, not to artefacts. And there is a working
> Streamlit prototype that clinicians can actually use.
>
> Four directions of future work: domain adaptation for smartphone photos
> — a topic I explore in appendix A23; open-set detection so the model can
> flag inputs that are not skin lesions; an active-learning loop where
> clinicians correct mispredictions; and edge deployment through ONNX or
> TensorRT."

**Delivery:** this is the closing summary — deliver it with confidence, look
at every committee member in turn.

---

## Slide 13 — Questions and Discussion (20 s)

> "Thank you for your attention. I welcome your questions and discussion."

**Delivery:** smile, stand straight, hands relaxed. Wait for the first
question — do not fill the silence.

---

## Contingency notes

- **Behind schedule (over 6:30 at slide 8):** drop the italic sentences,
  compress slides 6 and 7 into 25 s each, still finish slide 12 by 9:30.
- **Committee interrupts with questions during the talk:** answer briefly
  and mark where you were. Do not restart the section.
- **Streamlit demo requested during the talk:** politely say *"I have the
  demo running and I would be happy to show it after we finish the main
  slides"* — this protects your pacing.

## Q&A preparation

The ten most likely questions with model answers are in
[defense_qa_en.pdf](defense_qa_en.pdf). Focus on:

- Q1: Why soft voting, not stacking?
- Q2: How do you know Grad-CAM is correct?
- Q5: What about images outside the seven classes?
- Q6: How many melanoma false negatives?
- Q9: Bias against darker skin tones — Vietnam context.

If asked for the exact Mixup/CutMix, Label Smoothing, Weighted Sampler
formulas, or the optimisation setup (AdamW, mixed-precision, gradient
clipping, early stopping) — all of it now lives in **appendix A28**, not on
a main slide.

## Two hard rules

1. **Do not read the slides.** The committee can read faster than you speak.
   Use the slide as a prop, not a script.
2. **Do not over-claim.** If asked "is this ready for deployment?" — the
   answer is *no*, and here is what would be needed. Honesty scores points.
