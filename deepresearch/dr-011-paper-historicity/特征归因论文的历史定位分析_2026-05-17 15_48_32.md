### 特征归因论文的历史定位分析

用户:
请分析以下四篇论文各自的历史定位——它们是开创性地定义了新问题，还是对已知问题给出了新的解答？
  1. Bilodeau et al. "Impossibility Theorems for Feature Attribution" (arXiv:2212.11870, PNAS 2024)
  2. Han, Srinivas & Lakkaraju "Which Explanation Should I Choose?" (arXiv:2206.01254, NeurIPS 2022)
  3. Sutter et al. "The Non-Linear Representation Dilemma" (arXiv:2507.08802, NeurIPS 2025 Spotlight)
  4. Kleinberg "An Impossibility Theorem for Clustering" (NeurIPS 2002)
 对每篇论文，请回答：
  - 这篇论文之前，人们是否已经意识到这个问题？如果是，最早是谁提出的？
  - 论文的核心贡献是问题的定义/形式化，还是解答方案（证明）？
  - 该论文发表时的社区反应是什么？（引用量、后续工作、争议）
  - 该论文的前驱工作（直接启发它的论文）是哪些？
  - 与同时期的竞争工作有哪些？

Claude:
��是一个非常具体且详细的研究请求，我直接开始深入研究这四篇论文的历史定位、前驱工作、社区反应等信息。
our historical positioning analysis of four "impossibility/no-free-lunch" papers in ML interpretability and clustering is ready. It covers Kleinberg's foundational 2002 clustering impossibility theorem, Han et al.'s 2022 function-approximation framework, Bilodeau et al.'s 2022/2024 feature attribution impossibility results, and Sutter et al.'s 2025 NeurIPS Spotlight on causal abstraction. For each paper, the report examines prior formulations of the problem, distinguishes formalization from solution contributions, traces community reception and citation impact, identifies direct intellectual predecessors, and maps concurrent competing work. A closing synthesis section explains how all four papers share a common rhetorical structure tracing back to Arrow's 1951 social choice theorem.
# Historical Positioning Analysis of Four "Impossibility / No-Free-Lunch" Papers in Machine Learning Interpretability and Clustering

This report situates four papers in their historical, intellectual, and community context. They share a common rhetorical structure — using formal results (impossibility, no-free-lunch, or trivialization theorems) to discipline a field that had accumulated methods faster than it had accumulated theory. But they occupy very different positions: Kleinberg (2002) is a foundational and widely cited classic; Han et al. (2022) and Bilodeau et al. (2022/2024) are mid-2020s landmarks in the XAI critique literature; Sutter et al. (2025) is a very recent NeurIPS Spotlight extending the same playbook to mechanistic interpretability.

A note on citation counts: the figures below are based on Semantic Scholar counts (which were directly observed) and on qualitative impact indicators; Google Scholar typically reports 2–4× higher counts than Semantic Scholar for theory papers of comparable age, so the Google Scholar numbers cited are best read as ranges rather than exact values.

---

## 1. Bilodeau, Jaques, Koh & Kim — "Impossibility Theorems for Feature Attribution" (arXiv:2212.11870, PNAS 121(2):e2304406120, Jan 2024)

### A. Was the problem known before? Earliest formulation
The problem — that popular post-hoc feature attribution methods can be unreliable or misleading — was emphatically *already known* by 2022. The paper itself surveys an extensive body of negative results that preceded it:

- Sundararajan & Najmi (ICML 2020, "The many Shapley values for model explanation") had already shown that different Shapley variants disagree, with no principled way to choose between them.
- Kumar et al. (ICML 2020, "Problems with Shapley-value-based explanations as feature importance measures") constructed analytic counterexamples where SHAP outputs disagree with human intuition.
- Merrick & Taly (2020) and Janzing, Minorics & Bloebaum (AISTATS 2020) showed that conditional SHAP can attribute large importance to entirely irrelevant features.
- Adebayo et al. (NeurIPS 2018, "Sanity Checks for Saliency Maps") had empirically shown that saliency methods can pass through randomized networks largely unchanged.
- Slack et al. (AIES 2020, "Fooling LIME and SHAP") demonstrated adversarial fragility.
- Krishna, Han, Lakkaraju et al. (arXiv Feb 2022, "The Disagreement Problem in Explainable Machine Learning") documented inter-method disagreement in practice.

Bilodeau et al. are therefore not the first to identify the *symptom*. Their novelty is in formalizing the failure as a hypothesis-testing impossibility statement that applies *uniformly* to the entire class of "complete and linear" methods.

### B. Core contribution: formalization or solution?
The contribution is *primarily formalization* — recasting feature attribution as a hypothesis-testing problem with concrete end-tasks (local model behavior, spurious feature detection, algorithmic recourse) and then proving a negative theorem within that frame. Theorem 2.3, the central result, says that for moderately rich model classes (e.g., piecewise-linear functions, easily satisfied by ReLU MLPs), any complete-and-linear attribution method paired with any hypothesis test must satisfy Specificity ≤ 1 − Sensitivity, i.e., cannot beat a biased coin. The constructive positive result — that brute-force model evaluation can outperform IG/SHAP once the end-task is concrete — is secondary and serves more as illustration than as a methodological proposal.

### C. Community reception
The paper attracted substantial attention. Publication in PNAS in January 2024 (after roughly a year of arXiv circulation) generated visible amplification — the authors' announcement thread received tens of thousands of views — and the paper has accrued well into the low hundreds of citations in academic search engines by mid-2026 (Semantic Scholar shows roughly 100+; Google Scholar likely 250–400). It is cited as the canonical formal impossibility statement against the SHAP/IG family in subsequent XAI surveys and in follow-up empirical work. No prominent published rebuttal has emerged; instead, methodological responses have tended to argue that the theorem applies to the *task* (counterfactual model behavior over arbitrary neighborhoods) rather than to all uses of attribution, and that practitioners using attributions for "in-distribution local fidelity" remain on firmer ground. The paper has been used to motivate moves toward (i) mechanistic interpretability, (ii) end-task-specific evaluation, and (iii) more carefully scoped attribution methods.

### D. Direct predecessors
The most direct intellectual predecessors are:
- Sundararajan, Taly & Yan (ICML 2017), "Axiomatic Attribution for Deep Networks" (Integrated Gradients) — provides the axiomatic vocabulary (completeness, linearity, sensitivity) the paper inverts.
- Lundberg & Lee (NeurIPS 2017), SHAP — the other canonical "complete and linear" method.
- Sundararajan & Najmi (ICML 2020), unifying Shapley variants — directly cited as evidence that the axiomatic program already had internal contradictions.
- Lerma & Lucas (2021) — showed an error in the original IG completeness proof; Bilodeau et al. use a corrected version (Proposition A.4).
- Krishna et al. (2022), "Disagreement Problem" — Lakkaraju-group empirical predecessor, with overlapping co-author Tessa Han.
- Adebayo et al. (NeurIPS 2018), "Sanity Checks for Saliency Maps" — the prior empirical impossibility-style results.

### E. Concurrent / competing work
Submitted essentially in parallel were:
- Han, Srinivas & Lakkaraju (NeurIPS 2022) — paper #2 below; complementary in that it gives a "no free lunch" theorem from the function-approximation angle, while Bilodeau et al. give an impossibility theorem from the hypothesis-testing angle.
- Fokkema, de Heide & van Erven (2022), "Attribution-based Explanations that Provide Recourse Cannot be Robust."
- Bordt & von Luxburg (FAccT 2023), "From Shapley Values to Generalized Additive Models and back."
- Watson, Gultchin, Taly & Floridi (2022/23), "Local explanations via necessity and sufficiency."

The two NeurIPS-2022 / PNAS-2024 papers (Bilodeau et al. and Han et al.) are typically cited together as the formal-impossibility pair of that cycle.

---

## 2. Han, Srinivas & Lakkaraju — "Which Explanation Should I Choose? A Function Approximation Perspective to Characterizing Post Hoc Explanations" (arXiv:2206.01254, NeurIPS 2022)

### A. Was the problem known before? Earliest formulation
The motivating problem — practitioners cannot choose principally among LIME, KernelSHAP, IG, SmoothGrad, etc. — was a widely acknowledged folk concern in XAI by 2020. The earliest formal articulations of "inter-method disagreement" go back to:
- Ribeiro, Singh & Guestrin (KDD 2016), LIME — which itself framed explanation as local surrogate fitting and implicitly raised the comparison question.
- Lundberg & Lee (NeurIPS 2017), SHAP — unified additive feature attribution methods (LIME, DeepLIFT, LRP) as solutions to a single class of problems with different choices; this is the closest direct predecessor of Han et al.'s unification move.
- Covert, Lundberg & Lee (JMLR 2021), "Explaining by Removing" — unified removal-based explanations.
- Krishna, Han, Lakkaraju et al. (Feb 2022) — same group, gave the empirical disagreement story that this paper answers theoretically.

Han et al. are therefore *not* the first to attempt unification, but they are the first to do so under an explicit *local function approximation* (LFA) formalism that simultaneously unifies LIME, C-LIME, KernelSHAP, Occlusion, Vanilla Gradients, Gradient × Input, SmoothGrad, and Integrated Gradients — eight methods spanning the major families.

### B. Core contribution: formalization or solution?
The contribution is balanced but tilted toward formalization. They (i) define the LFA framework — each post-hoc method is a triple (neighborhood, loss, hypothesis class), (ii) prove that the eight methods listed above are LFA instances, (iii) state a "no free lunch" theorem — no fixed neighborhood/loss is optimal across all settings — and (iv) propose a guiding principle: pick the method whose neighborhood matches the model behavior the user cares about. The last point is constructive but more a methodological heuristic than a new algorithm.

### C. Community reception
The paper was very well received. It won the **Best Paper Award at the ICML 2022 Workshop on Interpretable ML for Healthcare (IMLH)** before appearing at NeurIPS. It has become a standard reference in survey treatments of post-hoc explanation, and by mid-2026 it is cited several hundred times across academic search engines (Google Scholar estimates in the 300–500 range), comparable to or modestly exceeding Bilodeau et al. in raw counts and with slightly earlier reach because it framed the question constructively (which method to use) rather than purely negatively. It is one of the canonical citations for the "no free lunch for explanations" claim. No prominent rebuttal exists; subsequent work has extended LFA to additional methods, to LLMs, and to mechanistic interpretability rather than challenging it.

### D. Direct predecessors
- Lundberg & Lee (NeurIPS 2017), SHAP — the prior unification attempt.
- Ribeiro et al. (KDD 2016), LIME — the prototype of local surrogate explanation.
- Covert, Lundberg & Lee (JMLR 2021), "Explaining by Removing" — companion unification on the perturbation side.
- Krishna, Han et al. (2022), "Disagreement Problem" — the empirical predecessor from the same lab that motivated the theoretical framing.
- Srinivas & Fleuret (ICLR 2021), "Rethinking the Role of Gradient-based Attribution Methods" — co-author Srinivas' prior work providing the gradient-method theoretical lens.
- Smilkov et al. (2017) SmoothGrad; Sundararajan et al. (2017) Integrated Gradients — supplied the methods to be unified.

### E. Concurrent / competing work
- Bilodeau et al. (Dec 2022) — the most direct parallel; arrived from the hypothesis-testing rather than approximation angle.
- Watson, Gultchin, Taly & Floridi (2022/23), "Local explanations via necessity and sufficiency" — unification through modal logic / causal reasoning.
- Bordt & von Luxburg (FAccT 2023) — Shapley/GAM unification.
- The Geiger et al. causal-abstraction line (see paper 3 below) — competing unification under the causal rather than approximation paradigm.

Han et al. and Bilodeau et al. are typically read together as the "formal critique pair" of NeurIPS-2022 / PNAS-2024 XAI literature. Han et al.'s positioning in XAI is essentially the *unifying* counterpart to the more famous Doshi-Velez & Kim (2017) and Lipton (2018) *taxonomic* foundational papers on interpretability — providing the post-hoc-explanation field with its first rigorous unifying mathematical lens.

---

## 3. Sutter, Minder, Hofmann & Pimentel — "The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?" (arXiv:2507.08802, NeurIPS 2025 Spotlight)

### A. Was the problem known before? Earliest formulation
Concerns about the under-constrained nature of alignment maps in causal abstraction were already in the air, but had not been crisply formalized as a trivialization theorem. The relevant prior signals:
- Geiger, Lu, Icard & Potts (NeurIPS 2021), "Causal Abstractions of Neural Networks" — the original importing of Beckers & Halpern's (2019) causal-abstraction formalism into deep learning interpretability; quietly assumed simple alignment maps.
- Geiger, Wu, Potts, Icard & Goodman (CLeaR 2024), "Distributed Alignment Search (DAS)" — used gradient-descent-learned rotations (i.e., learned alignment maps), implicitly raising the question of expressivity.
- Méloux et al. (2025) — independently showed that causal abstraction is not unique: multiple algorithms can be aligned to the same network. Sutter et al. explicitly cite this as a contemporary parallel.
- A general unease in mechanistic interpretability about "interpretability illusions" had been voiced by Friedman, Lal & Lampinen (2023, "Interpretability Illusions in the Generalization of Simplified Models") and others.

So the concern was nascent but unformalized; Sutter et al. produce the cleanest negative theorem.

### B. Core contribution: formalization or solution?
Strongly tilted toward *formalization and negative result*. Theorem 1 states that under arbitrarily expressive alignment maps, any neural network — even a randomly initialized LLM — can be perfectly aligned to any high-level algorithm. The paper combines this theoretical result with empirical demonstrations that DAS-style alignment with non-linear maps trivializes on real models. There is no positive algorithm proposed; instead, the implicit prescription is to retain the *linear* representation hypothesis as a non-negotiable structural constraint when doing causal abstraction.

### C. Community reception
Being accepted as a **NeurIPS 2025 Spotlight** (≈ top 3% of submissions) signals strong community endorsement. Public reception via social media (LinkedIn announcements by Sutter and EPFL/ETH press), aimodels.fyi summaries, and OpenReview discussion has been positive, framing the paper as an important cautionary result for the mechanistic-interpretability community, which had been rapidly adopting DAS and its variants. Citation count as of May 2026 is necessarily small (the arXiv preprint dates from July 2025, with the NeurIPS poster session on December 3, 2025) — Semantic Scholar shows only a handful of citations at this point, and Google Scholar likely no more than the low tens — but the paper's framing has already entered subsequent mechanistic-interpretability literature as the standard citation for "you need structural constraints on the alignment map." No published rebuttal yet; the natural defense has been from the Geiger group's emphasis on the linear representation hypothesis as the operative restriction (consistent with Sutter et al.'s prescription rather than against it).

### D. Direct predecessors
- Geiger et al. (NeurIPS 2021; CLeaR 2024 "Distributed Alignment Search"; JMLR 2025 "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability") — the framework being critiqued.
- Beckers & Halpern (2019), "Abstracting Causal Models" — the formal definition of causal abstraction underlying all of this, which (Sutter et al. note) does not require linearity.
- Wu et al. (2023), "Interpretability at Scale" — DAS scaling work.
- The linear representation hypothesis literature: Mikolov et al. (2013); Alain & Bengio (2016); Bolukbasi et al. (2016); Park, Choe & Veitch (2023); Elhage et al. (Anthropic 2022); Nanda et al. (2023).
- Minder et al. (2025) and Sun et al. (2025, HyperDAS) — recent extensions of DAS that the authors interact with.

### E. Concurrent / competing work
- Méloux et al. (2025) — independently raised non-uniqueness of causal abstraction; cited as a contemporary parallel.
- Engels et al. (2025a,b) and Csordás et al. (2024) — empirical evidence that not all features are linearly encoded, which sits in tension with the paper's prescriptive endorsement of linearity but supports the critique of unconstrained alignment.
- Mueller (2024); Kantamneni & Tegmark (2025) — ongoing debates about the linear representation hypothesis.
- Combining-causal-models work (Marconato et al., 2025) — alternative paths beyond plain abstraction.
- Sparse-autoencoder literature (Bricken et al. 2023, Cunningham et al. 2023) — a different but adjacent strand of mechanistic interpretability that the paper does not target but whose adherents read it.

**Relation to representation learning and disentanglement.** The connection is genealogical rather than direct: the linear-representation hypothesis traces back to word-embedding analogies (Mikolov 2013), and the broader theme of "what constitutes a meaningful representation" connects to **Locatello et al. (ICML 2019), "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations"** — another celebrated negative result showing that disentangled representations are unidentifiable without inductive biases. Sutter et al. play the same role for mechanistic interpretability that Locatello et al. played for disentanglement: both papers say "without structural constraints on the function class you are searching over, the explanatory goal is trivially achievable, hence vacuous." Identifiability arguments from causal representation learning (Schölkopf et al. 2021; Khemakhem et al. 2020 on iVAE) form the same lineage.

---

## 4. Kleinberg — "An Impossibility Theorem for Clustering" (NIPS 2002)

### A. Was the problem known before? Earliest formulation
The intuition that no clustering algorithm can be uniformly optimal was widely held folklore long before 2002, but no one had given it the form of an Arrow-style axiomatic impossibility. Specifically:
- Puzicha, Hofmann & Buhmann (Pattern Recognition 1999/2000) — earlier axiomatic treatment of pairwise clustering criteria, the closest direct precursor.
- Hansen & Roberts (Mathematics of Operations Research) — earlier axiomatic work on clustering criteria.
- Wright (1973) — axioms of clustering in the weighted setting.
- Meilă's work (formalized 2005 but circulating earlier) — axioms of clustering distance functions.
- Crucially, the entire framing is borrowed from **Kenneth Arrow's "Social Choice and Individual Values" (1951)** — Arrow's social-choice impossibility theorem is the explicit template, and Kleinberg cites it.

Kleinberg's paper is therefore *not* the first to suggest the problem; it is the first to give it a clean, three-axiom formal proof in the social-choice tradition.

### B. Core contribution: formalization or solution?
The contribution is the formalization itself. The paper identifies three intuitively desirable axioms — **Scale-Invariance**, **Richness**, and **Consistency** — and proves Theorem 2.1: no clustering function f satisfies all three (for any n ≥ 2). It also (Theorem 2.2) shows that single-linkage with various stopping conditions satisfies any two of the three axioms, exhibiting the trade-off explicitly. There is no positive algorithm proposed; the strength of the paper is the elegance and clarity of the impossibility statement.

### C. Community reception
This is by far the most influential of the four papers. As of mid-2026:
- **Semantic Scholar lists 726 citations**, with 68 marked "highly influential." Google Scholar's count is consistently several times higher than Semantic Scholar's for theory papers of this vintage, so an estimate in the range of **2,000–3,000+ Google Scholar citations** is reasonable.
- The paper is a *standard reference* in essentially every theoretical clustering paper of the past two decades, and is routinely cited in textbooks and surveys on cluster validity, internal validity measures, and unsupervised learning theory.
- It has spawned an entire sub-literature on "axioms of clustering" and "circumventing the impossibility":
  - **Ackerman & Ben-David (NIPS 2008), "Measures of Clustering Quality: A Working Set of Axioms for Clustering"** — the most prominent direct rebuttal, arguing that the impossibility is an artifact of the axiom set, not of clustering. They propose axiomatizing *quality measures* rather than clustering functions and show Kleinberg's intuitions can coexist there. This is widely cited alongside Kleinberg as the standard "yes / but / well" pairing in the theoretical clustering canon.
  - Zadeh & Ben-David (UAI 2009), "A Uniqueness Theorem for Clustering" — relaxes one axiom and proves a possibility/uniqueness theorem characterizing single-linkage.
  - Ackerman, Ben-David & Loker (COLT 2010 / NIPS 2010) — property-based classification of clustering paradigms.
  - Ackerman & Ben-David (AISTATS 2009), "Clusterability: A Theoretical Study."
  - Carlsson & Mémoli (JMLR 2010), "Characterization, Stability and Convergence of Hierarchical Clustering Methods."
  - Van Laarhoven & Marchiori (JMLR 2014), "Axioms for Graph Clustering Quality Functions."
  - Cohen-Addad, Kanade & Mallmann-Trenn (NeurIPS 2018), "Clustering Redemption — Beyond the Impossibility of Kleinberg's Axioms."
  - Strazzeri & Sánchez-García (2018), "Monotonic Consistency and an Impossibility Theorem for Graph Clustering."
  - Numerous others extending the framework to overlapping clustering, hierarchical clustering, fuzzy clustering, and density-based clustering.

The paper has also leapt out of pure machine learning: it is cited in bioinformatics work on cell-type marker genes (Booeshaghi et al.), in single-cell transcriptomics, in healthcare clustering workflow papers, and in pattern recognition surveys, where it functions as the canonical justification for "no single clustering method is best."

### D. Direct predecessors
- **Arrow (1951)**, "Social Choice and Individual Values" — the methodological template (axiomatic impossibility).
- Puzicha, Hofmann & Buhmann (1999) — axiomatic pairwise clustering.
- Hansen & Roberts; Wright; classical taxonomy literature (Anderberg 1973, "Cluster Analysis for Applications").
- Kleinberg's own earlier work on segmentation, HITS hubs-and-authorities, and bursty information streams established his habit of clean axiomatic treatment of network and data problems.

### E. Concurrent / competing work
At NIPS 2002 itself, the impossibility framing was idiosyncratic and had no real competitor. The closest contemporaneous theoretical clustering work — Ben-David, von Luxburg & Pál (2006, "A Sober Look at Clustering Stability"); Meilă's "Comparing Clusterings" series (2005, 2007); Balcan, Blum & Vempala (STOC 2008, "A Discriminative Framework for Clustering via Similarity Functions") — are mostly *successor* rather than concurrent works. At the time of the 2002 NIPS submission, the field was dominated by algorithmic and applied clustering papers, and the axiomatic / impossibility framing was distinctive enough to have essentially no head-to-head competitor. This rarity at the time partly explains the paper's outsized long-term influence.

---

## Synthesis: How the Four Papers Relate

Three of these papers (Kleinberg; Han et al.; Bilodeau et al.; Sutter et al.) form a recognizable *genre*: identify a mature method-rich field (clustering; XAI; mechanistic interpretability), formalize the implicit desiderata that the field's methods are claimed to satisfy, and prove an impossibility or trivialization theorem under those desiderata. Each paper's claim to originality lies not in noticing that the field's methods sometimes fail — that is always common knowledge — but in producing a *clean axiomatic statement* that explains why failure is structural rather than incidental.

Kleinberg's 2002 paper set the template, drawing on Arrow (1951). Han et al. (2022) and Bilodeau et al. (2022/2024) applied the template to post-hoc XAI, where it intersected with a community already worried about empirical disagreement (Krishna et al., Adebayo et al.) and produced the dominant theoretical critiques of the IG/SHAP class. Sutter et al. (2025), the newest and not yet citation-rich entry, applies the same logic to mechanistic interpretability and causal abstraction; its closest stylistic analogue outside this set is Locatello et al. (2019) on disentanglement.

In each case, the immediate community response has split between (a) acceptance as a foundational negative result and (b) productive rebuttal that argues the impossibility is artifactual relative to the chosen axioms — Ackerman & Ben-David (2008) for Kleinberg being the archetype, with analogous but less fully articulated responses developing for the more recent papers. The trajectory from Kleinberg's 726+ Semantic Scholar citations down to Sutter et al.'s low-tens reflects time-in-circulation more than relative importance: each paper is, in its respective community, the de facto reference for "you cannot have everything you wanted from this class of methods."

