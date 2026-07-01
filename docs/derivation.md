# Weighting and Pruning for a Fuzzy Lexical-Overlap Retrieval Engine
### A probabilistic-IR derivation under a thermodynamic lens

This document derives the scoring layer of a fast, typo-tolerant lexical-overlap retrieval engine — a component that streams a bounded set of candidate segments to a downstream reranker or rank-fusion ensemble. Its job is recall under a near-constant per-query work budget: given a short, possibly-corrupted query, surface every segment that plausibly matches, cheaply, and let a more expensive stage sort them. Matching is on character $n$-grams, the standard basis for approximate string search [Ukkonen1992].

The derivation rests on a standard probabilistic-IR model (binary independence / likelihood ratio). A thermodynamic and information-theoretic reading runs alongside it, as intuition and as a source of cross-checks. Its organizing identification, used throughout, is that a gram's inverse document frequency is the energy it carries (§1); a second, the count credit as a chemical potential, follows once the credit is derived (§3). The reading also marks where it breaks down — the collective, phase-like analogies a few-gram query cannot support (§10). The probabilistic-IR model does the derivations.

Throughout, the unit of indexing is a segment: a short document or document fragment. The engine is not aware of any application-level grouping of segments into larger documents.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $N$ | number of indexed segments (the unit of retrieval; fixed for a given index snapshot) |
| $df_g$ | segment-frequency of gram $g$ — the number of segments containing it |
| $p_g = df_g/N$ | marginal probability that a random segment contains $g$ |
| $\kappa = 0.5$ | Jeffreys smoothing pseudocount (estimation correction) |
| $\nu$ | corroboration depth; sets the contamination floor (§4) $df_{\min}=N^{(\nu-1)/\nu}$ (default 2) |
| $df^{\mathrm{eff}}_g = \max(df_g, df_{\min})$ | floored segment-frequency used in the weight |
| $E_g = \ln\dfrac{N - df^{\mathrm{eff}}_g - \kappa}{df^{\mathrm{eff}}_g + \kappa}$ | per-gram energy: the RSJ log-odds (logit-idf) of the smoothed, floored estimate; used clamped as $\max(0,E_g)$. Its rare-gram limit is the surprisal $\mathrm{idf}_g = \ln\frac{N}{df^{\mathrm{eff}}_g+\kappa}$ |
| $E_{\max} = \tfrac{1}{\nu}\ln N$ | single-gram energy ceiling implied by the floor |
| $\beta,\ Z$ | inverse temperature; partition function (Boltzmann form, §1) |
| $Q$ | the (deduplicated) set of query grams |
| $P \subseteq Q$ | the pruned set actually scored |
| $r$ | reliability: probability a relevant document matches a real query gram — a corpus constant per channel ($\sigma$ query-side; $\rho$ doc-side, itself per gram order $n$) |
| $\sigma,\ \rho,\ \varepsilon$ | query-side / doc-side reliabilities and the doc-side per-character error rate, with $\rho = \sigma\,(1-\varepsilon)^n$ (topicality $\sigma$ × corruption-survival); $\rho < \sigma$ always — the doc-side channel carries the same topicality $\sigma$ as the query side, then multiplies in corruption |
| $\mu = \max(0,\operatorname{logit} r)$ | count credit — nats per matched, non-floored gram (§4) |
| $L_d,\ \bar L$ | distinct gram count of segment $d$; its corpus mean |
| $K_P$ | length-null slope (§6), summed over $P$, mirroring the accumulator (§6/§7) weights |
| $c$ | stopping margin (a Cantelli parameter, not a z-score) |
| $\varphi_d$ | gram co-failure correlation at start-distance $d$; the per-character-independent $(r^{d/n}-r)/(1-r)$ is the no-clustering reference (clustering raises co-failure toward $1$, anti-clustering lowers it) — the recall-safe operating value is the comonotone $\varphi=1$ block (§5) |
| $k$ | target candidate-pool size; the stop aims for $\ln(N/k)$ nats |
| $C$ | work budget — the cap on $\sum_{g\in P} df_g$, bounding posting-list cost |
| $\Delta$ | quantization step for the bit-sliced energy weights |
| $\Delta H = H_3 - H_2$ | vocabulary-complexity gap between trigram/bigram df-distributions, setting the fusion weight (§8) |
| $k_{\mathrm{RRF}},\ w_{\mathrm{tri}},\ w_{\mathrm{bi}}$ | reciprocal-rank-fusion rank constant; per-view weights |

Logs are natural (nats); scoring is "more is better," so energy is added. Gram order $n$ is a property of the script (CJK bigrams, else trigrams; §8), so $n$ varies per gram in a mixed-script query; the selection class is the `(script, order)` pair; a stopping block is one query word (§5).

---

## 1. Surprisal is energy

Consider a distribution $\theta$ over the gram vocabulary. The maximum-entropy distribution consistent with a fixed mean energy $\sum_g \theta(g) E_g$ is the Boltzmann form [Jaynes1957; Berger1996]

$$\theta(g) = \frac{1}{Z}\,e^{-\beta E_g}, \qquad Z = \sum_g e^{-\beta E_g},$$

with $\beta$ the inverse temperature. Inverting this — asking which energies reproduce the empirical corpus statistics $p_g = df_g/N$ (the fraction of the $N$ segments containing gram $g$) — identifies the energies, up to an additive constant, with the surprisal:

$$-\ln p_g = \ln\frac{N}{df_g} = \mathrm{idf}_g \qquad(\text{up to an additive constant}).$$

Inverse document frequency is the energy a gram carries: rare grams are high-energy, common grams low-energy. The additive constant is immaterial, since ranking is invariant to it. Surprisal is the energy the reading assigns; the energy the scoring uses is the likelihood-ratio refinement of §2 — the log-odds $\ln\frac{1-p_g}{p_g}$ (logit-idf), of which surprisal is the rare-gram limit. The two coincide for rare grams, the regime the lens is valid in anyway (§10), so "surprisal is energy" holds wherever the lens does any work. ($Z$ never enters quantitatively: the corpus marginals sum to the mean distinct-gram count $\bar L$ rather than to one, so $\theta = p$ is an identification, not a literal probability assignment.)

The maximum-entropy step is used here as inference and is scale-free, valid for a handful of grams; it is not a physical thermodynamic claim, and the physical reading (energy, collective entropy, phase behavior) is meaningful only in aggregate and is not leaned on where the systems are small (§10). A fuller thermodynamics-of-IR treatment — partition function and specific heat over a corpus — is given by [Koroutchev2009]; this derivation keeps the local, per-gram identifications and parts from it at the collective quantities, including the specific-heat reading of fragility (§4).

---

## 2. Scoring as a likelihood ratio

Score the relevance of a segment $d$ to a query $Q$ by a per-gram log-likelihood ratio, treating grams as independent:

$$S_d = \sum_{g \in Q \cap d} \log\frac{P(\text{match}\mid R)}{P(\text{match}\mid \bar R)} \;+\; \sum_{g \in Q \setminus d} \log\frac{P(\text{miss}\mid R)}{P(\text{miss}\mid \bar R)},$$

where $R$ denotes relevance and a random (non-relevant) segment matches gram $g$ with its corpus marginal, $P(\text{match}\mid \bar R) = p_g$. This is the binary-independence (Robertson–Spärck-Jones) model [RSJ1976; Robertson2004], whose exact matched-gram weight is $\mu + \log\frac{1-p_g}{p_g}$ — the count credit $\mu$ of §3 plus a logit-idf. We adopt this logit-idf as the operating energy $E_g = \log\frac{1-p_g}{p_g}$, so the matched weight $E_g + \mu$ is the exact RSJ log-likelihood ratio for the idealized model energy. In the clean, rare-gram limit a reported score is then a calibrated log-odds in nats — the property a downstream consumer that reads magnitudes (weighted fusion, a learned reranker, a score threshold) depends on. The operating score departs from this exact log-odds by the recall-safe approximations of §4–§7 — the $\max(0,\cdot)$ clamp, the contamination floor, $\Delta$-quantization, and the subtracted length null — so the calibration holds in the clean limit and decays under corruption (§10).

The familiar summed-IDF overlap $\sum_{g\in Q\cap d}\mathrm{idf}_g$ is its rare-gram limit. Dropping $\log(1-p_g)$ leaves the surprisal $-\ln p_g = \mathrm{idf}_g$ [Shannon1948; SparckJones1972], equal to $E_g$ in the rare tail and diverging only for common grams. The gap is $-\log(1-p_g)$ — about $0.36$ nats at $p_g = 0.3$ and a full $0.69$ at $p_g = 0.5$, where a gram in half the corpus carries zero discriminative evidence that surprisal would still score at $0.69$. The matched sum is the matched part of the query's cross-entropy against the corpus marginal — the background-model reading of the language-model / risk-minimization retrieval framework [LaffertyZhai2001], scored against the corpus rather than a per-document model — and logit-idf is the consistent likelihood-ratio choice with surprisal as its approximation.

The full cross-entropy — the query's gram distribution scored against the corpus background — also includes the second sum above: a penalty for query grams absent from the document. It is tempting to read the matched-only score as simply discarding that penalty for robustness. That is not what happens, and that difference is the subject of §3. A typo leaves no gram for the penalty to act on, since the mangled gram drops out of the query entirely, but it injects an artifact in its place. It is the artifacts, not the lost grams, that a naive penalty mishandles.

Under the noise model the absent-gram penalty splits in two. Artifact grams — rare or absent in the corpus, and no more likely in a relevant document than in a random one — are inert: their match and miss log-ratios are both zero, so a document missing one is neither rewarded nor penalized. This is where raw KL on the unfiltered query distribution goes wrong: it treats artifacts as real words and penalizes relevant documents for lacking them. The contamination floor of §4 and this inertness are the mechanisms that prevent it. Real query grams should appear in a relevant document, so their absence is evidence against relevance, and its reliability-bearing part is retained — reorganizing into a per-query constant (which drops from ranking) plus the per-match count credit $\mu$ of §3.

The score thus takes an overlap-only form: the engine walks only the matched posting lists, never the absent grams, while still accounting for the absent-gram information. A real gram's absence reorganizes into the count credit $\mu$ plus a per-query constant that drops from ranking, and the matched energy $E_g$ carries the $\log(1-p_g)$ factor; the artifact part is exactly zero. With logit-idf as the energy, nothing is approximated away relative to surprisal — the common-gram $\log(1-p_g)$ factor that surprisal would drop is retained. So the overlap-only model score is the exact binary-independence likelihood ratio up to the rank-invariant constant, before the operational clamp, floor, quantization, and length null of §4–§7.

---

## 3. The noise model and the count credit

The absent-gram term, handled rather than dropped, is what produces the count credit $\mu$ and fixes its meaning. Two channels of noise are worth distinguishing, because they license different scoring.

### 3.1 Query-side noise (the default)

The corpus is clean; the query is corrupted by the user's typing. The received query grams split into real grams — correct fragments of the intended query that survived — and junk grams — substitution artifacts, which are rare or absent in the corpus.

For a real gram, a relevant (clean, on-topic) document contains it with reliability $\sigma$, close to one: $P(\text{match}\mid R) = \sigma$. For a junk gram, a relevant document contains the artifact only at its corpus rate, $P(\text{match}\mid R) \approx p_g = P(\text{match}\mid \bar R)$. The junk gram's matched and missed log-ratios are therefore both $\log 1 = 0$: junk grams are inert on the miss side, and a document missing one is not penalized. On the match side, the contamination floor of §4 hands a junk gram the ceiling energy $E_{\max}$; what keeps that from out-ranking a genuine match is the count-credit policy of §9, not the energy (developed in §4 and §9).

For the real grams, with $\#m$ matched out of $\#\text{real}$ present in the query, the absent-gram sum's reliability part regroups (its frequency part is the energy $E_g = \log\frac{1-p_g}{p_g}$ of §2):

$$\#m\,\log\sigma + (\#\text{real} - \#m)\log(1-\sigma) = \underbrace{\#\text{real}\,\log(1-\sigma)}_{\text{constant in } d} + \#m\,\operatorname{logit}\sigma.$$

Collecting per matched real gram, so that each contributes the full RSJ weight $E_g + \mu$:

$$\boxed{\;S_d = \sum_{g \in Q \cap d,\,\text{real}} E_g \;+\; \#m \cdot \mu, \qquad \mu = \operatorname{logit}\sigma = \log\frac{\sigma}{1-\sigma}.\;}$$

The flat per-match bonus $\mu$ is the count credit. In the thermodynamic reading it is a chemical potential: the conjugate to particle number, the reward for admitting one more matched "particle." Its sign and magnitude follow from the reliability — $\sigma > \tfrac12$ gives $\mu > 0$, with $\mu \to \infty$ as $\sigma \to 1$ (perfect reliability makes a missing query gram strongly disqualifying).

### 3.2 Doc-side noise (opt-in)

Here the query is clean but the corpus is noisy (OCR, user-generated text, spelling variants), so a relevant document contains a clean query gram only with reliability $\rho$. There are no artifacts. The same algebra gives $\mu = \operatorname{logit}\rho$, now applied to every query gram. Two things must both hold for a relevant document to match a clean query gram: its intended on-topic text must contain it — topicality, the same sub-one factor the query-side $\sigma$ carries, since a relevant document need not contain every query gram even uncorrupted — and the gram must survive corruption. So $\rho = \sigma\cdot(1-\varepsilon)^n$: the topicality $\sigma$ times the corruption-survival derived next. Dropping the $\sigma$ would silently assert that every relevant document's clean text contains every query gram, sending $\rho\to 1$ as $\varepsilon\to 0$; an over-stated $\rho$ feeds the stop (§5) and fires it early — a recall loss, not just a magnitude error.

This channel has a single underlying quantity. Model a relevant document as the intended on-topic text with each character independently corrupted at rate $\varepsilon$ (write $q = 1-\varepsilon$). A length-$n$ gram matches iff all $n$ of its characters survive, so

$$\rho = \sigma\cdot q^{\,n} = \sigma\,(1-\varepsilon)^{n}.$$

So one declarable number $\varepsilon$ — a property of the ingestion source, not its content (clean structured text $\varepsilon \approx 0$, OCR $\varepsilon \approx 0.01$–$0.03$, heavy user text higher) — together with the same topicality $\sigma$ the query side already needs, fixes $\rho$, the count credit $\mu = \operatorname{logit}\rho$, and (via §5) the stopping variance's diagonal and its no-clustering reference; the operating off-diagonal is the parameter-free comonotone block (§5). Because $\varepsilon$ is a per-**field** ingestion property and the contamination floor is baked into each posting's bit-sliced energy planes — and a posting spans segments from many fields — the doc-side channel is a property of the *index*, not a per-search knob; it lands with the field-aware index milestone (per-field posting partitions), not as a search-time parameter. Because $\rho = \sigma(1-\varepsilon)^n$, doc-side reliability is always below the query-side $\sigma$, and corruption only deepens the gap, so "$\rho<\sigma$" is structural rather than a separately tuned "noisier corpus" comparison. The per-character idealization is just that: real errors cluster (keyboard adjacency, systematic OCR confusions), so $\varepsilon$ is a principled default. An application can also estimate it at index time from the fat tail of singleton grams that are one substitution from a common gram — a label-free, corpus-internal signal that needs no relevance judgments.

### 3.3 Reliability is a corpus constant

A single reliability $r$ — $\sigma$ for query-side, $\rho$ for doc-side — drives both the count credit $\mu = \operatorname{logit} r$ and the stopping rule (§5). The decisive property is that $r$ is a corpus and relevance property, not a query property. It is the probability that a relevant document contains a correct query gram, and that probability does not depend on how the user mistyped elsewhere in the query. "elephant" and "eelphant" share their surviving real grams, and a relevant document contains those reals at the same rate in both cases. Query corruption changes the query's composition — more junk, fewer surviving reals — but it cannot move $r$. The consequence for the count credit is developed in §9.

This is invariance per gram. It need not hold for the survivor-averaged reliability — the quantity the realized credit actually rides on — when true reliability varies by gram and corruption is gram-selective. If harder-to-spell words are both mistyped more often and carry different reliability, conditioning on "this gram survived" selects a biased subset, so the average reliability over survivors can drift with corruption even though each gram's value is fixed. This drift's sign is not pinned; it is set by the unknown correlation between a gram's mistype-rate and its reliability. If harder-to-spell grams are more reliable, survivors skew toward easy, less-reliable grams and the averaged reliability drifts down, over-stating the stop's $r$ and firing it early (recall-unsafe); the opposite correlation drifts it up. Unlike the floor leakage of §9, whose sign is pinned upward, this one is indeterminate, and it is set aside as second-order for the same reason (no per-gram reliability data day-one), not because it is known to be recall-safe. It is flagged here so the per-gram invariance is not mistaken for the stronger averaged claim.

---

## 4. Per-gram weighting

Two finite-sample corrections refine the bare energy.

**Estimation.** The maximum-likelihood estimate $\hat p_g = df_g/N$ is overconfident in the rare tail: a gram seen once has a wide posterior on its true rarity. Jeffreys smoothing [Jeffreys1946] corrects it to $\hat p_g = (df_g + \kappa)/N$, and the energy is the log-odds of the corrected estimate:

$$E_g = \ln\frac{1-\hat p_g}{\hat p_g} = \ln\frac{N - df_g - \kappa}{df_g + \kappa}, \qquad \kappa = 0.5,$$

with rare-gram limit the smoothed surprisal $\mathrm{idf}_g = \ln\frac{N}{df_g+\kappa}$. This $E_g$ is the standard Robertson–Spärck-Jones relevance-free term weight [RSJ1976], not a fresh construction, with the smoothing in the well-studied family of language-model smoothing for IR [ZhaiLafferty2004]. The smoothing is always applied. It noticeably affects the continuous quantities (the pruning budget of §5) and is nearly invisible to coarsely-quantized weights. (The strict Jeffreys posterior mean is $(df_g+\tfrac12)/(N+1)$; the $N$-versus-$N{+}1$ normalization is dropped as immaterial at scale.)

**Contamination.** Under query-side noise, some grams are rare because they are not language — the substitution artifacts — and the energy mistakes them for highly informative. A floor caps their energy from below:

$$df^{\mathrm{eff}}_g = \max(df_g, df_{\min}), \qquad df_{\min} = N^{(\nu-1)/\nu}.$$

This is a reparametrization: choosing the floor as a power of $N$ caps the single-gram energy at a clean fraction of the identification budget,

$$E_{\max} = \ln\frac{N}{df_{\min}} = \tfrac{1}{\nu}\ln N.$$

Identifying one of $N$ segments costs $\ln N$ nats, so a ceiling of $\tfrac1\nu\ln N$ means no single gram can identify a segment alone: at least $\nu$ matched grams must agree. (The expression for $E_{\max}$ drops the $\kappa$ and the $\log(1-\hat p_{\min})$ correction — together $\lesssim\!10^{-2}$ in relative terms for $N\gtrsim 10^4$, growing as $N$ shrinks since $\hat p_{\min}=N^{-1/\nu}\to 0$ only asymptotically; logit-idf and surprisal coincide at the floor.) The parameter $\nu$ (default 2) is the corroboration depth. The floor is applied only on the query-side channel; under doc-side noise every gram is a genuine word and a low df is real information, so only the estimation smoothing applies.

One caution, carried to §9: the floor does not push a junk gram below real grams. It caps the junk gram's energy at the ceiling $E_{\max}$, which still exceeds every non-floored real gram's energy (all of which have $df > df_{\min}$). What actually orders junk below real is the count-credit policy of §9, not the floor.

Numbers, symbols, and punctuation tokenize as a `Common`-class run (§8) and are floored on the same footing. A real but rare number or ID is capped at $E_{\max}$ and denied the credit: recall-safe (the document is still retrieved, and excluding the gram from the stop's running mean only defers the stop, §5/§9), precision-suboptimal, and the correct default, because IDs are mistyped too — a fat-fingered ID produces the rare artifacts the floor exists to cap. An application that knows a field is a structured identifier can opt that field into the doc-side channel (no floor, full idf, credit), an explicit opt-in with floor-on by default.

A caution on reading $\nu$ as a typo margin: one substitution destroys not one gram but the $\sim n$ contiguous $n$-grams that span the changed character [Navarro2001] (in "hello", changing the second character kills two of the three trigrams), and surviving grams that are positionally adjacent share characters, so a single substitution can take both. Robustness therefore depends on the positional spread of the surviving grams, not merely their count, and the per-typo tolerance is closer to $(m-\nu)/n$ than $m-\nu$.

Weights are linear in the energy. Because the matched sum is a log-odds, weights must be linear in $E_g$ for the sum to remain interpretable and to compose additively with $\mu$. Weighting by a power of $df$ (exponential in information) would break this.

**A note on fragility.** It is tempting to measure a query's fragility — its vulnerability to a single lost gram — by the variance of its grams' energies, the heat-capacity analog [Koroutchev2009]. This is wrong, and it is where the present argument parts from a specific-heat reading. Variance is spread, but fragility is single-point-of-failure, and the two diverge: $\{12, 12, 3, 3\}$ has high variance yet is robust (two rare grams; lose one, the other carries), while $\{12, 3, 3\}$ has lower variance yet is fragile (one rare gram doing the work). The right kind of statistic is an order statistic — the top energy against the sum of the rest, $\max(0,\,E_{\text{top}} - \sum_{\text{rest}} E)$ — which gives $0$ for the robust case and a positive value for the fragile one. (This is one reasonable order statistic among several; the essential point is that spread is the wrong family and an order statistic the right one.)

---

## 5. Pruning

To bound tail latency, the query is pruned to a subset $P \subseteq Q$ before scoring. This is a knapsack: maximize collected energy $\sum_{g\in P} E_g$ subject to a work budget $\sum_{g\in P} df_g \le C$, the posting-list cost. Value per unit cost is $E_g/df_g$, which decreases in $df_g$, so rarest-first is near-optimal. "Rarest" is measured by class-normalized rarity — a z-score of the gram's df within its own `(script, order)` class, not raw global df — so a multi-script query interleaves each script's rarest grams, turning a systematic burial of a low-global-energy script into a benign tie-break.

Interleaving alone is not a guarantee: one script can still monopolize the few grams collected before the stop fires. So the pruner additionally seats the rarest in-corpus gram (floored or not) of every present `(script, order)` class unconditionally — a per-class floor that generalizes the typo floor below, where a floored seat is cheap and stays out of the stop and credit while guaranteeing the posting is walked. This makes per-script representation a true invariant for every present class with an in-budget ($df \le C$) gram (§8). A class whose only grams are common is abundantly represented already, so its seat is skipped when it would breach the budget. A `df > C` gram is excluded by the work budget like any other, since one cannot walk a posting larger than the candidate budget; dropping a class whose only gram is over-budget is a small, bounded recall cost — the same $C$-bounded latency/recall tradeoff every query already pays, not a multi-script-specific drop. The floor's added work is thus one walk per script: cheap whenever the class has a rare gram, bounded by $C$ otherwise.

The energy that is scored stays the global log-odds of §2; only the selection key is class-normalized. Within a single class this reduces to plain rarest-by-df.

Two subtleties. First, the class-normalized key is not globally monotone in $df$ (a rare-within-class CJK bigram can carry a larger global $df$ than a common-within-class Latin trigram), so the budget cutoff cannot be a clean prefix break: a break at the first over-budget gram could exclude later, cheaper minority-script grams and undo the interleaving. The cutoff is therefore skip-and-continue — a gram that would breach $C$ is skipped and scanning continues, filling the budget with the cheapest grams the class-normalized order surfaces. (This also keeps the $E_{\max}$-priced floored grams of §9 from monopolizing $C$: an over-budget floored gram is skipped, not a barrier.) Second, the true-$df$-ascending tiebreak resolves z-score ties, and in the floored tail, where energies coincide at $E_{\max}$, it keeps the cheapest first. Grams with $df = 0$ (artifacts with no postings) are dropped before pruning: they match nothing and only consume weight range.

### How far to prune: a confidence-bounded stop

Identifying one of $N$ segments costs $\ln N$ nats; narrowing to a candidate pool of size $k$ costs $\ln(N/k)$. But the evidence a truly relevant document accumulates is random — it matches each kept gram only with reliability $r$ — so the stop should clear the target with margin, not on average. The variance below is organized into blocks, each block being one query word, with matches modeled as Bernoulli($r$):

$$\text{stop when } \sum_{g\in P} r\,\max(0,E_g) \;-\; c\,\sigma_{\text{match}} \;\ge\; \ln(N/k), \qquad \sigma_{\text{match}}^2 = r(1-r)\sum_{\text{blocks } b}\Big(\sum_{g\in b}\max(0,E_g)\Big)^2.$$

This confidence-bounded stop over a work budget differs from score-upper-bound dynamic pruning — WAND [Broder2003], MaxScore [Turtle1995], Block-Max WAND [DingSuel2011] — which safely skips documents that cannot enter the top-$k$ by score. Here the criterion is a probabilistic Cantelli bound on collected identification evidence against a posting-cost budget, not a score upper bound.

Two points. First, over the handful of grams in a pruned query there is no central-limit regime, so $c$ is not a Gaussian z-score; the distribution-free guarantee is Cantelli's [Cantelli1928], $P(\text{evidence} < \text{mean} - c\sigma) \le 1/(1+c^2)$ — at $c=2$ that is at most 20%, not the 2.3% a normal tail would suggest. Cantelli holds at any sample size, which is why it applies here. Second, the covariance is positive, query-dependent, and must be carried, but it is not a free parameter: the noise model organizes it. Contiguous grams share a failure cause, so the true variance exceeds the independent estimate, and a constant inflation cannot bound an excess that grows with the query's contiguity. The clean way to carry it is to group co-failing grams into comonotone blocks — a block is a maximal contiguous run of kept grams, the grams of one query word — and sum independent block variances, an $O(|P|)$ pass over the query-word boundaries the tokenizer marks (the realization marks these on **whitespace**, and because the identical tokenizer runs on both index and query the two agree by construction — the single-tokenizer invariant — so whitespace-level word-breaking is recall-safe; finer delimiter-level breaking is a deferred precision refinement):

$$\text{block } b:\quad \operatorname{Var}\Big(\textstyle\sum_{g\in b}\max(0,E_g)\,B_g\Big) = r(1-r)\Big(\textstyle\sum_{g\in b}\max(0,E_g)\Big)^2,\qquad \sigma_{\text{match}}^2 = \sum_{\text{blocks } b}\operatorname{Var}(b).$$

**Why a block is one comonotone unit.** Query-side, a relevant document contains a whole query word or none of it — a word-level event — so the grams of one word are comonotone by construction and $\varphi = 1$ is exact. Doc-side, a single corruption burst over a word's characters removes every gram spanning it, and the worst case, a burst as long as the run, couples them all: the comonotone Fréchet upper bound $\varphi = 1$ [Dhaene2002]. Either way a block contributes the variance boxed above, and distinct blocks (distinct words) are independent. Because one block is a single script run, it carries one gram order and so one reliability $r$; a mixed-script doc-side query therefore uses a per-block $r_b$, which the §12 algorithm applies per gram. This form is positive-semidefinite by construction — it is the genuine variance of a comonotone vector, unlike a pairwise $\varphi=1$ summed over an overlap window, which can specify a non-PSD quadratic form (e.g. three trigrams at start-positions $0,2,4$ give a correlation matrix of determinant $-1$). It is per distinct gram-string: a string appears once in its block, so a repeated gram cannot double-count its own covariance. And it closes the long-range residual a pairwise overlap window ($d<n$) leaves open — a burst longer than one gram couples non-overlapping grams that the window scores as independent, under-counting $\sigma_{\text{match}}$ and stopping early (recall-unsafe); the whole-run block captures that coupling.

**The no-clustering reference.** The block uses $\varphi = 1$; the interior value an overlapping pair takes under purely independent corruption is worth recording, both as the calibration reference and as a less-conservative operating option. Under the per-character model of §3.2 — a relevant document is the intended text with each character independently intact with probability $q$, so the per-character survival reliability is $r = q^{n}$ (this is the topicality-free $\sigma=1$ case; doc-side it is the survival factor $\rho/\sigma$, not the full $\rho = \sigma q^n$) — a length-$n$ gram matches iff its $n$ characters survive, and two grams at start-distance $d$ (with $0 < d < n$) span a union of $n+d$ characters and both match iff all of them survive. Hence, using $q = r^{1/n}$,

$$\operatorname{Cov}(B_g, B_h) = q^{\,n+d} - q^{2n} = r\big(r^{d/n} - r\big), \qquad \boxed{\;\varphi_d = \frac{r^{d/n} - r}{1 - r}.\;}$$

The ends check: $d \to 0$ gives $\varphi = 1$, $d = n$ gives $\varphi = 0$, monotone between; for trigrams only $d = 1$ and $d = 2$ are correlated. This $\varphi_d$ is exact under i.i.d. (independent and identically distributed) per-character corruption, but it is the no-clustering reference, not a bound on the operating value. Real errors cluster, raising co-failure above $\varphi_d$ toward the comonotone $\varphi = 1$ the block uses, while anti-clustered errors would push it below $\varphi_d$. So $\varphi_d$ is an interior reference point, and $\varphi = 1$ is the true Fréchet upper end: a within-block burst can couple the whole run, not merely an adjacent pair, which is why the block, not the pair, is the right unit. An application can interpolate between the block bound and this i.i.d. reference using the realized adjacent co-failure rate (the singleton-tail signal of §3.2), trading a little recall safety for an earlier stop; the block bound is the recall-safe default. One caution on that knob: because $\varphi_d$ here uses the topicality-free survival $r=q^n$, doc-side it sits below the true within-word co-failure, which inherits the shared word-level topicality $\sigma$. Interpolating toward $\varphi_d$ by plugging the full $\rho$ into the formula therefore understates the variance by more than the no-clustering picture suggests, so the topicality-inclusive co-failure should be used instead.

The stop is recall-safe only when its mean is not over-stated and its variance not under-stated. Both errors push the stop the same way, toward early firing, and they compound, so they are worth stating together.

The variance is now under-stated only across block boundaries: co-occurring query words are treated as independent (the cross-word assumption), which slightly under-counts when two query words tend to appear together in relevant documents. This is the one residual the block form does not close — the same family as the within-block burst, but between words rather than within one. A single gram string occurring in two query words is the same residual seen sharply: it is assigned to one block and omits its coupling with the other. A strict implementation merges query words that share a gram into one block, which only raises the variance and so stays recall-safe; but that remedy closes only this shared-string case, and co-occurring distinct query words with no shared gram remain an accepted residual.

The mean $\sum_g r\max(0,E_g)$ is over-stated when $r$ is set too high. Doc-side this is the topicality factor in $r = \sigma(1-\varepsilon)^n$ (§3.2): dropping it inflates $r$, which raises the mean and (for $r > \tfrac12$) shrinks $r(1-r)$, so the stop clears the target with fewer grams and the Cantelli guarantee is evaluated at the wrong, too-high $r$. The survivor-averaged drift of §3.3 is a second, smaller such effect, of indeterminate sign. The covariance bites whenever $r$ is meaningfully below one, most acutely doc-side, where relevant documents are themselves noisy. So the recall-safe posture is to keep the mean from being over-stated (carry topicality) and the variance conservative (the block bound), and over-collection in the corrupted regime is the cheap, intended price (§9: over-retrieval is cheap; the work budget $C$ still bounds it). Concretely the block stop can fire at all only when $r \ge c^2/(B+c^2)$, with $B$ the number of independent query words collected (at $c = 2$: $0.80$ for one word, $0.67$ for two, $0.50$ for four); a low-$r$, few-word doc-side query falls below that threshold and simply collects to the budget $C$ — the declared, bounded price of an all-or-nothing comonotone word.

(Floored grams are excluded from the stop's running mean and variance on the query-side channel, for the same reason they receive no count credit in §9: a junk gram contributes zero identification power, and crediting it $r\,E_{\max}$ would stop collection early. A query that is all junk then never reaches sufficiency; its grams are rare, with short posting lists, so few segments match and it correctly emits the resulting sparse candidate set rather than filtering.)

**Setting the budget by the stop.** The work budget $C$ need not be a free dial. The stop fixes how much identification evidence to collect — $\ln(N/k)$ nats; the budget fixes how much work to spend collecting it — postings scanned. These are one requirement in two spaces, and the exchange rate between them is the idf itself: scanning a gram's $df$ postings buys $\approx\ln(N/df)$ nats. So the work to collect the target from grams of representative frequency $\bar d$ is the reliability-inflated target divided by the nats-per-posting of the available grams,

$$C = \frac{1}{r}\,\ln\!\frac{N}{k}\,\cdot\,\frac{\bar d}{\ln(N/\bar d)},$$

the Lagrangian dual of the knapsack — the budget that just funds the stop. Deriving $C$ this way closes the loop: setting it independently of $k$ would double-specify the same identification requirement. The representative frequency $\bar d$ is read from the corpus as a high percentile of the `(script, order)` class's $\ln df$ distribution, $\bar d = \exp(\text{mean}_{\ln df} + Z\,\text{sd}_{\ln df})$, with $Z$ a shape constant erring toward a commoner, larger $\bar d$ (hence a larger, recall-safer $C$); the pruned selection then keeps every gram rarer than $\bar d$ and lets the §7/§12 rescue floor recall on the rest. So $C$ self-calibrates from index statistics and needs no tuning, though a caller may still override it. Recall-safe degeneracies — $N\le k$, or a $\ln df$ distribution too sparse or too ubiquitous to place $\bar d$ strictly inside $(1,N)$ — fall back to an unbounded budget rather than a too-tight one. This also dissolves the last tuned count cutoff: with the budget load-bearing, selection is $F$ (the typo floor) plus the per-`(script, order)` floor plus rarest-first collection until the stop or $C$, with no separate count ceiling.

### Reading the query before and after pruning

Pruning is a rarity-biased subsample — it keeps the rare tail and drops the common head — so the full and pruned queries carry different diagnostics. Comparing the pruned collected energy against the full query's distinguishes two kinds of starvation. If the full query had far more energy than the pruned set, the budget cut away usable signal (loosen the budget, or restore the cheap mid-rarity grams it dropped). If the full query was itself weak, no budget helps and the right move is to fall to a lower gram order (§8). The grams pruning held back are a corroboration reserve: under fragility, restoring those real mid-rarity grams is a better first response than dropping to bigrams, because they are genuine language rather than redundant sub-grams.

---

## 6. Length correction

A flat count credit $\mu > 0$ rewards matching more grams, and longer segments have more opportunities to match — particularly to chance-match common grams. Left uncorrected, this is a length bias toward long documents, the effect that document-length normalization has long addressed [Singhal1996; RobertsonWalker1994]. Note that the energy term alone ($\mu = 0$) is already largely length-insensitive, because a long document's extra matches are the low-energy common grams worth almost nothing. Under logit-idf the residual mid-rarity inflation is muted further, since the energy-part length slope $p_g\ln\frac{1-p_g}{p_g}$ is smaller than the surprisal slope $p_g\ln\frac1{p_g}$ and vanishes by $p_g = 0.5$; the bias is mainly an artifact of the count credit flattening the energy spread.

The principled correction subtracts a segment's expected score under the null that it is a random draw from the corpus. Under a bag model, a gram is present in a length-$L_d$ segment with probability $\pi_g(L_d) = 1 - (1-p_g)^{L_d/\bar L}$ — the presence rate consistent with the marginal $p_g$ at mean length $\bar L$, with length entering as the relative length $L_d/\bar L$. For a rare gram this is $\approx (L_d/\bar L)\,p_g$ (linear), but for a common gram in a long segment the linear form exceeds one — an impossible probability that over-debits the null — whereas $\pi_g$ saturates below one. The correction is then

$$\boxed{\;S^{\text{adj}}_d = \underbrace{\sum_{g\in P\cap d}\big(\max(0,E_g) + \mu\cdot\mathbb{1}[g\text{ non-floored}]\big)}_{\text{accumulator}} \;-\; \sum_{g\in P} \pi_g(L_d)\big(\max(0,E_g) + \mu\cdot\mathbb{1}[g\text{ non-floored}]\big), \qquad \pi_g(L_d) = 1-(1-p_g)^{L_d/\bar L}.\;}$$

For the rare grams that dominate a typical pruned set, $\pi_g \approx (L_d/\bar L)\,p_g$ is linear, so their contribution collapses to the separable $(L_d/\bar L)\,K_P$ with $K_P = \sum_{g\in P} p_g(\max(0,E_g) + \mu\,\mathbb{1}[\text{non-floored}])$ precomputed once; only the few common grams need the per-candidate saturating term. The saturation matters most in the count-and-length regime (common-heavy queries, energy $\approx 0$), where the null is the entire ranking signal: the un-saturated form would debit a long relevant segment that matched everything below shorter, weaker ones, and — if the emitted set is truncated to $k$ — push it out of the pool, a recall loss.

The governing principle is that the null is the expectation of the exact statistic the accumulator computes, so it must mirror the accumulator's weights term for term — summed over the pruned set $P$ (only those grams enter any score), with the energy clamped to non-negative (matching the accumulator's clamp), and with the count credit applied only to non-floored grams (matching where the accumulator actually adds it). A null that sums over the full query, or credits floored grams, or uses raw negative energies, reintroduces a length-correlated bias. The mirror of the accumulator weight is exact — the null tracks the quantized weight $\Delta\cdot\mathrm{round}(\max(0,E_g)/\Delta)$, not its continuous value — while the presence model is the bag-model $\pi_g$ above, exact in the rare tail and saturating (no longer first-order) for commons.

The $\mu$ part of the slope is the substantive term; the energy part self-damps as $p_g\ln(1/p_g)\to 0$ for rare grams but does not vanish for a mid-rarity set, so the correction is applied unconditionally rather than gated on $\mu > 0$. The energy-part length slope is the logit-idf slope $p_g\log\frac{1-p_g}{p_g}$ — strictly smaller than surprisal's and zero by $p_g = 0.5$ — so the mid-rarity residual is already muted by the §2 energy choice; the null cancels whatever remains. It is a per-candidate floating-point adjustment in the ranking step, outside the inner accumulation loop, so applying it always costs nothing in the hot path. Like the concentration cap (§9), this correction is only worth its cost if the emitted set is truncated to roughly $k$ below the budget-bounded union: a consumer that re-scores the whole union downstream gets these refinements for free, but a recall stage that hands a reranker a bounded top-$k$ depends on them — which is why the saturation above is not optional.

---

## 7. Representation and the work budget

The accumulator weights are quantized to small non-negative integers, $\tilde w_g = \max(0, \mathrm{round}(E_g/\Delta))$, and accumulated by bit-slicing the matched posting lists — Roaring bitmaps [Lemire2018] under a bit-sliced index arithmetic [O'NeilQuass1997; Rinfret2001]. Because the count credit is a flat shift,

$$\sum_{g\in P\cap d}(E_g + \mu) = \underbrace{\sum_{g\in P\cap d} E_g}_{\text{bit-sliced}} + \mu\cdot\underbrace{|P\cap d|}_{\text{popcount}},$$

only the energy part is bit-sliced. The plane count is then at most $\lceil\log_2(E_{\max}/\Delta + 1)\rceil$ — independent of $\mu$, and in practice the exact bit-width of the realized maximum weight — and the count credit is added in the post-pass as $\mu$ times a popcount of matched (non-floored) grams, the same quantity the engine's overlap mechanism already produces. This keeps the hot loop narrow and makes the count credit free to be zero or signed.

Two corners need pinning. The plane count is floored at 1 (an all-common query whose grams all quantize to zero would otherwise request zero planes), and the post-pass iterates the union of the bit-sliced candidates and the popcount candidates — so a segment that matched only zero-weight (clamped) grams still receives its count-and-length score rather than vanishing. The result is that a query with no rarity discrimination degrades to a count-and-length ranking instead of returning empty. (One precondition: a segment matching only floored grams is recovered through the bit-sliced side, since floored grams are excluded from the popcount; this holds only if the quantization is fine enough to keep their weight nonzero, $\Delta < 2E_{\text{floored}}$, the realized floored-gram energy, marginally below $E_{\max}$. A coarser step would zero them out and drop such segments.)

Finally, the count-credit and length terms are post-accumulation floats, so any top-$k$ truncation must be applied after them, on the corrected scores — never on the bare bit-sliced sum, which would truncate on the wrong ordering. The post-pass is cheap enough to run over the full candidate set first: the work budget $\sum_{g\in P} df_g \le C$ bounds the budget greedy, and the unconditional per-class floor seats (§5) add at most one in-budget posting per present class, so the candidate union obeys $\#\text{candidates} \le \sum_{g\in P} df_g \le (1+\#\text{classes})\,C$ — still $O(C)$ for a fixed, small number of script classes. The post-pass is then $O(C)$ and the inner loop $O(C\cdot\text{planes})$. "Near-constant work per query" means "work bounded by the df budget." The budget $C$ itself defaults to the corpus-derived Lagrangian dual of the stop (§5), $C = \tfrac1r\ln(N/k)\,\bar d/\ln(N/\bar d)$, so this bound needs no tuned constant; selection carries no separate count ceiling, since the query's finite gram set bounds count and $C$ bounds work.

---

## 8. Multiple gram orders and scripts

Two different situations put grams of different orders in one query, and they combine differently. Same-run multi-order — one script run windowed at two orders, a Latin word as both trigrams and bigrams — is redundant: a bigram is a near-deterministic function of its containing trigram (`abc` implies `ab` and `bc`), so pooling a single contiguous trigram match would fire the count credit three times and add the two sub-bigram energies on top, counting one piece of evidence as three, with the overcount correlated with contiguity and length. Cross-script multi-order — a mixed-script query whose Latin run yields trigrams and whose CJK run yields bigrams — is complementary: those grams span disjoint characters with no containment, so they are simply more evidence about the same segment and pool into one accumulator (the ordinary heterogeneous overlap sum, already what a single-order multi-script index does). The rule is therefore pool the disjoint, fuse the contained: fusion is reciprocal-rank fusion (RRF) [Cormack2009], which reads only ranks and is robust to the additive double-counting; pooling is the plain weighted sum.

Gram order is a property of the script, not a global choice. A dense script — CJK (Han, Hiragana, Katakana, Hangul) — is windowed at bigrams, where one bigram is already about as discriminating as a Latin trigram; everything else at trigrams. Each script run is scored at its primary order plus, when the query is starved or corrupted, a richness-gated secondary one order shorter — Latin trigram + bigram, CJK bigram + unigram — RRF-fused (the secondary is sub-gram-contained in the primary). The secondary is the more robust, less selective layer: a substitution corrupts fewer of the shorter grams (each spans fewer characters) and there are more of them. The shortest order doubles as the structural fallback for a query too short to produce the primary (a two-character Latin query, a one-character CJK query). A CJK unigram secondary is not noise even though a lone common character is unselective: a Han character is a morpheme (nearer a Latin word than a Latin letter), and the rarity weighting already drives a common character's energy to zero and prunes its long posting list on the budget — so only rare, discriminating morphemes survive selection, exactly the corroboration a starved bigram view lacks.

For a mixed-script query the two layers generalize to ranked views, not absolute orders. The primary view pools every script's primary-order grams (Latin trigrams ∪ CJK bigrams — disjoint, so pooled without double-counting); the secondary view pools every script's richness-gated secondary (Latin bigrams ∪ CJK unigrams). One RRF over `[primary, secondary]` fuses them, with `missing="omit"` so a per-script gate that runs no secondary for a script simply omits it. The sub-gram containment (trigram⊃bigram within Latin, bigram⊃unigram within CJK) lives across the two views, where RRF reads ranks not summed energy — so a contiguous match ranks well in both views (the intended robustness) without being additively tripled. Pooling and fusion thus compose: pool disjoint scripts within a view, fuse the contained orders across views.

Pooling cross-script grams into one view does not by itself make scripts compare fairly — a globally-rarer script would out-score and, worse, out-collect: the rarest-first pruning (§5) would fill the budget with one script's grams and the confidence stop (§5) could fire before a minority script's grams are ever selected, dropping documents relevant only to the minority script. The fix lives in selection, not in the energy: the energy that is scored stays the global log-odds (§2), while the rarest-first selection key is the class-normalized rarity — a z-score of the gram's df within its own `(script, order)` class — so each script's rarest grams interleave into the pool, turning a systematic burial into a benign tie-break; a per-class floor that seats the rarest in-corpus gram (floored or not) of every present `(script, order)` class unconditionally (§5) then turns representation from a tendency into a guarantee for every present class with an in-budget ($df \le C$) gram. This is the multi-script half of §5's pruning, and it is why the energy can remain global without starving a script (class-normalizing the energy itself would be worse, not better — it only lowers a minority gram's weight further, since within-class df-fraction ≥ global).

The fusion weight between two views is informed by how much richer each script's primary inventory is than its secondary — a per-`(script, order)` vocabulary-complexity gap $\Delta H = H_{\text{primary}} - H_{\text{secondary}}$ between the two segment-frequency distributions, computed once per index. This is a directional heuristic, not a conditional entropy even approximately: the two distributions are tabulated independently over different supports and are not a joint/marginal pair ($\sum_c df(xyc) \ne df(xy)$ in general), so $\Delta H$ supplies the sign and scale directly from the index, while the monotone map from $\Delta H$ to the fusion weight is a fixed shape choice (a linear map suffices); a mixed-script query combines the per-script gaps (or, simplest, equal-weights them). The fusion's rank constant controls how sharp the fusion is — a flatter fusion trusts more candidates — and a segment retrieved by one view but not the other is simply omitted from the absent view's contribution rather than treated as worst-ranked.

---

## 9. The count credit as a policy

The count credit $\mu = \max(0, \operatorname{logit} r)$ is a per-channel constant (floored at zero for a recall stage, where penalizing matches would be counterproductive and over-retrieval is cheap). Its value differs between channels — $\operatorname{logit}\sigma$ query-side, $\operatorname{logit}\rho$ doc-side — and on the doc-side channel it legitimately tracks corpus noise through $\rho$. Within the query-side channel, the credit parameter does not vary with the degree of query corruption, because reliability $\sigma$ is a corpus constant (§3.3): corruption changes query composition — junk grams receive no credit, and there are fewer real matches to reward — not $\sigma$. As the user types and the query fills in, what changes smoothly is this junk exclusion and the count of real matches, not the parameter $\mu$.

The realized credit drifts slightly, and in one direction only. The credit is applied to the grams the floor judges real (non-floored), and the floor has a false-negative rate: a junk artifact that collides with a real corpus word ($df \ge df_{\min}$) clears the floor and collects the full $\mu$, despite its true reliability being near the corpus rate rather than $\sigma$. Such collisions grow with corruption, so the realized per-match credit is over-stated by an amount that rises with corruption. The drift is upward — over-crediting chance matches — hence in the recall-safe direction, and the length null absorbs part of it (it debits a high-$p_g$ leaked gram by $(L_d/\bar L)\,p_g\,\mu$). So the parameter is corruption-invariant; the realized credit is invariant only to the extent the floor separates real from junk. Correcting the residual would mean discounting $\mu$ by an estimated corruption level — worth doing only if a magnitude-consuming downstream consumer needs the calibration (§10), not for recall, where the drift is already safe.

**What the credit is.** The count credit is reliability-weighted count evidence, not a corroboration reward. Corroboration — multiple independent rare matches reinforcing each other — is already rewarded by the summed energy, which is large for rare grams. The credit captures something separate: under high reliability, a relevant document is expected to contain query grams, so each match earns a flat bonus and each absence is mild evidence against. This is why the credit is withheld from floored grams on the query-side channel: a floored gram is junk-suspicious, of unknown reliability, and should not receive the reliability-based bonus. Doing so costs nothing for rare real grams (which a low df may have floored) — they already dominate on summed energy — and correctly denies it to artifacts.

This withholding is essential, not incidental. The contamination floor parks a junk gram at the ceiling energy $E_{\max}$, above every non-floored real gram (§4), so on energy alone a junk match would out-rank a real one; the count-credit policy — $\mu$ on non-floored grams only — is what restores the junk-below-real ordering, and only while $\mu \gtrsim E_{\max} - E_g$. A real gram common enough that its energy falls more than $\mu$ below $E_{\max}$ can still be out-scored by a single floored junk match (at the defaults $N=10^6$, $\sigma=0.9$, $\nu=2$ the crossover is near $df \approx 9000$); the §6 length null and multi-gram corroboration absorb this residual rather than eliminating it — a precision distortion the reranker undoes, not a rare-signal recall loss. The same fact has a pruning-side face: because floored grams all carry $E_{\max}$ they sort to the front of the rarest-first queue, so a budget-tight query can spend $C$ on floored grams before reaching non-floored ones — but the df-ascending tiebreak (§5) admits the rare reals (small df) first, so only low-energy common reals, which the budget drops anyway, are displaced; the rare signal is preserved.

**Concentration.** One structural case warrants reducing the credit below its baseline. When the pruned set holds a single dominant rare gram amid several common, easily-chance-matched grams, a large flat $\mu$ lets an off-topic segment that happens to hit the commons outweigh an on-topic segment that hit the one rare gram. The credit should then be capped so the count cannot swamp the discriminating gram. "Common" is defined relative to the query itself, not by a corpus cutoff: a gram counts as common here if its energy is well below the top gram's, say $E_g < \tfrac12 E_{\text{top}}$. The cap applies only when the set is concentrated — a dominant gram at the top energy and at least two such commons below it — so an all-common query, whose grams are comparable and have no dominant member, is left uncapped and degrades to count-and-length ranking (§7) rather than having its credit zeroed. For a concentrated set,

$$\mu \le \max\!\left(0,\ \frac{\max(0,E_{\text{top}}) - \sum_{g\in\text{common}} \max(0,E_g)}{\#\text{common} - 1}\right),$$

restricted to the query-relative commons, so that a query spread across several comparable rare grams — which has no dominant member, hence no commons under the relative definition — remains uncapped. The dominant gram $E_{\text{top}}$ and its commons range over **all** of the pruned grams, floored ones included (the literal reading of $P$). This is deliberate: because $df_{\min}=\sqrt N$ is a low bar, a genuinely rare, real discriminating gram is itself floored and sits at $E_{\max}$, so admitting floored grams as $E_{\text{top}}$ lets that rare gram anchor the cap and shield the on-topic document from commons-count — the common, valuable "find the document carrying this rare term" case. Excluding floored grams (considered and rejected) would instead leave no dominant gram for exactly those queries and *disable* the cap, letting a commons-only document out-credit the on-topic one. The price of including them — a query that pairs a junk floored gram (whose ceiling energy loosens the cap) with a real non-floored discriminator — is rare, since a rare gram the user actually typed is almost always the intended discriminator, and is in any case a precision distortion the reranker undoes rather than a recall loss. The hard floor at zero, reached when several commons collectively outweigh the dominant gram, discards all count evidence in a single step; a smoother form that shrinks $\mu$ toward the cap is preferable in tuning. Because the threshold is relative to the query's own top energy, it self-calibrates per query with no corpus-specific cutoff; only the fraction ($\tfrac12$) and the hard-versus-smooth choice are universal shape constants. (Concentration is a property of query structure, not corruption level: a short clean query can be concentrated, and a heavily corrupted query can still be spread.)

---

## 10. The unifying picture and its limits

The whole system moves along a single axis with a chemical-potential and temperature reading. At one pole — clean, complete queries against a reliable corpus — the count credit is high, rarity and corroboration are trustworthy, the engine runs cold and selective on trigrams with tight pruning. At the other — starved or corrupted queries — composition shifts (junk excluded, fewer matches), the bigram layer comes in, the budget keeps its corroboration reserve, and the stop runs conservative. The engine emits a score together with its components (summed energy, match count, length). A downstream stage that re-ranks loses nothing in any regime — the recall-safe approximations move ranks only in recall-safe directions. A stage that consumes magnitudes is served a calibrated, log-odds-shaped score in the clean limit; the recall-safe approximations — the $\max(0,\cdot)$ clamp, the contamination floor and its leakage, the concentration cap, the subtracted null — preserve recall while distorting ranks (recall-safely) and eroding magnitude calibration as corruption rises, so a consumer that trusts the bare order or the raw magnitude degrades rather than either holding exactly under corruption.

| | clean / cold | starved / warm | doc-side (noisy corpus) |
|---|---|---|---|
| contamination floor | off / light | $N^{(\nu-1)/\nu}$, $\nu=2$ | off |
| count credit $\mu$ | $\operatorname{logit}\sigma$ (constant) | same, with composition shifting and the concentration cap if structurally concentrated | $\operatorname{logit}\rho,\ \rho=\sigma(1-\varepsilon)^n$ |
| credit on floored grams | — | no | yes |
| gram views | primary order only | primary + richness-gated secondary (one shorter), RRF-fused | per query cleanliness |
| pruning budget | tight | keep reserve | tight |
| length correction | always on | always on | always on |
| stopping variance | small ($\sigma\to1$) | per-word comonotone block, $\varphi=1$ (exact) | per-word comonotone block, $\varphi=1$ Fréchet bound (i.i.d. $\varphi_d$ the no-clustering reference) |

The lens is reliable for local, per-gram quantities — surprisal and its logit-idf refinement, the per-gram weight, the count credit as a form — which are well-defined for a handful of grams. It breaks for collective phenomena that require a thermodynamic limit. A "phase transition" in a four-gram query is noise, not criticality; there is nothing to anneal in one-shot retrieval; the entropy of mixing two gram orders collapses to the vocabulary-complexity gap of §8. Collective and phase analogies, the Clausius–Clapeyron relation among them, need a thermodynamic limit a few-gram query lacks, so the design does not use them — past the local identifications the analogy becomes decoration.

---

## 11. Assumptions and open questions

- **Gram independence.** The likelihood ratio and the matched sum treat grams as independent. Within an order, adjacent grams overlap and are correlated. This correlation is accounted for in the stopping variance (§5) but not in the score mean, so a segment whose matches are contiguous is somewhat over-credited relative to one with the same match count spread out. For a recall stage this is acceptable — it is a precision distortion the reranker can undo — but correcting it would require a positional-overlap redundancy penalty, trading away the additive simplicity the design keeps. This is a chosen tradeoff, open to revisit.
- **The matched-weight model is the exact logit-idf** $\mu + \log\frac{1-p_g}{p_g}$ (§2); surprisal $-\ln p_g$ is its rare-gram limit, used nowhere operationally. The reported score is this log-odds only in the clean, rare-gram limit — the operational $\max(0,\cdot)$ clamp, contamination floor, $\Delta$-quantization, and subtracted null (§4–§7) move it off the exact RSJ value in recall-safe directions, so it is log-odds-shaped and decalibrates gracefully under corruption (§10), not an unconditional "calibrated log-odds." The intended approximations in the weight are the smoothing and contamination floor (§4) and gram independence (below), not the common-gram inflation surprisal would have carried. Relative to surprisal, common grams now weigh less (zero at $p_g \ge \tfrac12$), so a relevant document whose only matches are common grams scores marginally lower — a slight recall cost carried by the count credit $\mu$ and the concentration cap rather than by common-gram energy.
- **The bag-model length null** (§6) uses the saturating presence $\pi_g = 1-(1-p_g)^{L_d/\bar L}$; it ignores within-segment burstiness and structure but captures the dominant length effect and no longer over-debits long segments on common grams.
- **The noise model is idealized.** The real/junk split is treated as sharp, with the floor as its detector; in reality the floor has a false-negative rate, and a small fraction of junk leaks through, which is what makes the realized count credit drift upward with corruption (§9), in the recall-safe direction. The invariance is of the credit parameter, not the applied credit (nor the survivor-averaged reliability, whose drift sign is indeterminate, §3.3). A single reliability per channel, uniform across real grams, stands in for a per-gram quantity. For the stopping covariance, the per-character closed form $\varphi_d = (r^{d/n}-r)/(1-r)$ (§5) is exact under i.i.d. corruption and is the no-clustering reference, not a bound: clustering raises co-failure above it toward $\varphi = 1$, anti-clustering lowers it. The operating choice is the comonotone-block $\varphi = 1$ (the Fréchet upper bound, exact query-side), which closes the within-word burst residual the old pairwise overlap-window left open. The only residual it does not close is cross-word coupling: a shared gram string across two query words is closed by merging them into one block, but co-occurring distinct query words with no shared gram remain an accepted residual.
- **The stop's recall-safety is conditional, and the conditions compound.** The confidence-bounded stop (§5) is recall-safe only to the extent its mean $r$ is not over-estimated and its variance $\sigma_{\text{match}}$ not under-estimated; both errors fire it early and stack. Three mechanisms erode it: the doc-side topicality factor in $r=\sigma(1-\varepsilon)^n$ (§3.2, a first-order modeling choice — dropping the $\sigma$ over-states $r$), bursty co-failure across a block boundary (§5, cross-word), and the survivor-averaged reliability drift (§3.3, second-order, indeterminate sign). The design carries topicality and the block bound to control the first two; the third is set aside for lack of per-gram data.
- **The pruning budget assumes $C$ dwarfs the floored mass.** Floored grams all carry $E_{\max}$ and sort to the front of the budget queue (§9); the df-ascending tiebreak admits the rare reals first, so the rare signal is preserved, but if the leaked-junk mass near $df_{\min}$ is a large fraction of $C$ it can crowd out non-floored common reals (which the budget drops by design anyway). Bounded and recall-benign provided $C \gg \#\text{floored}\cdot df_{\min}$, with $df_{\min}=N^{(\nu-1)/\nu}$.
- **$\Delta H$** (§8) is a vocabulary-complexity gap used as a directional fusion heuristic, not a conditional entropy, for the structural reason noted there.
- **The concentration cap** (§9) has the right form and sign; its "common" threshold is query-relative ($E_g < \tfrac12 E_{\text{top}}$), so it needs no corpus cutoff, leaving only the fraction and the hard-versus-smooth clamp as universal shape constants to settle.
- **Multi-script handling is class-aware in selection, global in energy.** Gram order is per script (CJK bigrams, else trigrams), so a mixed-script query carries several orders, pooled into one accumulator with per-gram reliability $r = \sigma(1-\varepsilon)^n$ (§8). The energy stays the global log-odds (§2), while multi-script fairness is carried by the class-normalized selection key (§5) — rarest-within-`(script, order)`, so a minority script's burial becomes a benign tie-break rather than systematic — and a per-class floor (seat the rarest in-corpus gram, floored or not, of every present `(script, order)` class, §5) makes per-script representation a true invariant for every present class with an in-budget ($df \le C$) gram. Class-normalizing the energy itself would be strictly worse, since it only lowers a minority gram's weight (within-class df-fraction $\ge$ global). Two residual class-asymmetries are recall-safe and deferred. The global floor $df_{\min}$ over-floors a large-vocabulary script: the CJK bigram space dwarfs the Latin trigram space, so more rare CJK bigrams hit $E_{\max}$, a precision skew a per-class $df_{\min}$ would remove. The `Common` class (numbers, symbols, punctuation) is one bucket a finer general-category split would refine. The stop's word blocks come from whitespace and delimiter punctuation, which also break gram windows so no gram straddles two query words (§5/§8); intra-word punctuation (the apostrophe in don't) is word-internal and optionally stripped for recall.
- **Parameters, and how few are corpus-bound.** What looks like a parameter sweep is mostly not. The doc-side channel reduces to one declarable number — the ingestion error rate $\varepsilon$ — together with the same topicality $\sigma$ the query side already needs, from which $\rho = \sigma(1-\varepsilon)^n$, the count credit, and the stopping variance's diagonal and no-clustering reference follow in closed form (§3.2, §5). An application declares $\varepsilon$ from provenance or estimates it label-free at index time. (The recall-safe operating covariance uses the parameter-free comonotone-block $\varphi = 1$ bound, not $\varepsilon$.) The fusion ratio comes from $\Delta H$, computed from the df distributions the index already holds, so it self-calibrates the moment data lands; only the monotone $\Delta H \to$ weight map is a shape choice. The cap threshold is query-relative (above). The one corpus- and relevance-bound quantity is the query-side reliability $\sigma$ — it depends on what "relevant" means and cannot be read from index statistics — but its sensitivity is low (rare grams are dominated by summed energy, commons are bounded by the cap, and $\mu$ is inert on single-gram queries), so a high constant default ($\sigma \approx 0.9$) is safe day-one, with optional self-supervision from click/selection logs. The stopping margin $c$ and the work budget $C$ are latency/safety dials, not corpus quantities. So the irreducible corpus dependence is essentially $\varepsilon$ plus a low-sensitivity $\sigma$ default — both supplied without a sweep.

---

## 12. Algorithm

```text
# index-time, once per snapshot
constants(index):
    N       = segment_count
    Lbar    = mean(distinct_gram_count(d) for d in segments)
    Emax     = ln(N) / nu                                             # single-gram energy CEILING (Section 4)
    df_min   = N**((nu - 1) / nu)                                     # contamination floor (query-side, Section 4)
    E_floored= ln((N - df_min - KAPPA) / (df_min + KAPPA))            # realized floored-gram energy (<= Emax); DELTA < 2*E_floored (Section 7)
    order(script)            = CJK(script) ? 2 : 3                    # primary gram order per script class (Section 8); secondary = order - 1
    zstats[(script, order)]  = (mean, sd) of ln(df) over that class   # class-normalized SELECTION key (Section 5/8); welford, online, never persisted
    dH[(script, order)]      = H(df over (script,order)) - H(df over (script,order-1))   # per-(script,order) fusion gap (Section 8)

energy(df_g, channel, N):                                              # RSJ log-odds (logit-idf); surprisal is the rare-gram limit
    df_min = (channel == QUERY_SIDE) ? N**((nu - 1) / nu) : 0
    df_eff = max(df_g, df_min)
    return ln( (N - df_eff - KAPPA) / (df_eff + KAPPA) )               # KAPPA = 0.5; < 0 for p>0.5, zeroed by max(0,.) at use

score_query(query, channel):
    # Tokenize into grams tagged with script + order; a mixed-script query yields several orders. Word
    # boundaries come from whitespace + delimiter punctuation, which ALSO break gram windows so no gram
    # straddles two words (Section 8/4). RANK-VIEWS, not absolute orders: grams_at_rank(PRIMARY) pools every
    # script's primary-order grams (cross-script disjoint); grams_at_rank(SECONDARY) pools the one-shorter
    # order (Latin bigram, CJK unigram) of each STARVED script -- a PER-SCRIPT gate (Section 8). A script is
    # STARVED if it is richness-low OR produced no df>0 primary-order gram; the latter makes the shortest order
    # the structural fallback (a 2-char Latin query has no trigram, so Latin is starved and contributes bigrams).
    # A rich script with a primary gram omits its secondary.
    starved(s) = richness_low(s) or (no df>0 gram of order(s) in script s)
    rank_views = any(starved(s) for s in present_scripts(query)) ? [PRIMARY, SECONDARY] : [PRIMARY]
    while rank_views and not any(any(lookup_df(g) > 0 for g in grams_at_rank(rv)) for rv in rank_views):
        rank_views = (rank_views == [PRIMARY]) ? [SECONDARY] : []    # structural fallback to the shorter rank, only when EVERY
        #                                                              #   rank-view lacks an in-corpus (df>0) gram (test df>0, not raw presence)
    per_view = []

    for rv in rank_views:                                            # rv = a RANK (PRIMARY / SECONDARY), pooling all scripts at that rank
        Q = [g for g in grams_at_rank(rv) if lookup_df(g) > 0]       # pool each script's grams at this rank; drop df=0 artifacts
        if not Q: continue                                          # this rank-view is empty -> skip it, never abort the query (Section 8)
        for g in Q:
            g.df = lookup_df(g);  g.E = energy(g.df, channel, N)     # GLOBAL energy (n-independent), the log-odds (Section 2)
            g.script = script_of(g);  g.n = order_of(g)              # selection class = the (script, order) pair (CJK 2/1, else 3/2)
            g.r  = (channel == DOC_SIDE) ? SIGMA * (1 - eps)**g.n : SIGMA   # PER-GRAM reliability: topicality SIGMA x per-order survival
            g.floored = (channel == QUERY_SIDE) and (g.df <= df_min)   # query-side only; doc-side has no floor (Section 4) so every gram keeps its credit
            g.word = query_word_of(g)                                # comonotone-block id = g's query word. Usually one script run (one order/r); a no-space
            #                                                          #   mixed-script word merges differing-r runs, which the per-gram g.r update below over-states -> conservative (recall-safe)

        # pruning: CLASS-NORMALIZED rarest-first (z-score of df within (script,order)); per-class floor, then
        # skip-and-continue budget. The z-score key only de-biases collection (turns a systematic burial of a
        # low-global-energy script into a benign tie-break); the per-class FLOOR is what guarantees a seat.
        sort Q by (zscore(g, zstats) asc, true df asc)              # rarest-within-class first (Section 5/8); true-df-asc breaks z-score ties
        # DERIVED work budget C (Section 5/7): the Lagrangian dual of the stop. dbar = the Z-percentile of
        # ln df pooled over the query's present (script,order) classes; a caller df_budget overrides it.
        dbar = exp(pooled_mean_lndf(Q) + Z * pooled_sd_lndf(Q))     # Z = 2 (~P98): errs toward a commoner, recall-safer dbar
        C    = (N <= k or dbar not in (1, N-1)) ? +inf              # recall-safe guards -> unbounded, never too-tight
             : (1/SIGMA) * ln(N/k) * dbar / ln(N/dbar)
        P = [];  sum_df = 0;  sumE = 0;  sumVar = 0;  target = ln(N / k);  block_sum = {}
        def admit(g):                                              # add g to P and fold it into the stop's running mean/variance
            P.append(g);  sum_df += g.df
            if channel == DOC_SIDE or not g.floored:                  # floored grams carry no identification power
                e = max(0, g.E)                                       # mirror the accumulator's clamp; PER-GRAM reliability g.r
                s = block_sum.get(g.word, 0)                         # comonotone-block variance (phi=1): EXACT query-side, Frechet UB doc-side;
                sumE += g.r * e                                      #   a block is one script run -> one r; PSD; per distinct string
                sumVar += g.r * (1 - g.r) * (2*s*e + e*e);  block_sum[g.word] = s + e
        # (1) per-class floor: the rarest df>0 gram (FLOORED OR NOT) of EACH present (script,order) class whose
        #     seat fits the budget (df <= C). A floored seat is cheap and stays out of the stop/credit, but
        #     guarantees its posting is walked -> an all-floored minority class is no longer dropped. A class
        #     whose only grams have df > C is excluded by the work budget like any df>C gram (the C-bounded
        #     tradeoff -- a small bounded recall cost, not multi-script-specific) -- so floor work is ~ #classes*C.
        floor_seats = [ first g per distinct (g.script, g.n) with g.df <= C, in z-score order ]   # >= 1 present gram per representable class
        for g in floor_seats: admit(g)                             # UNCONDITIONAL (even past C and the stop) -- generalizes the typo floor; per-class representation invariant for in-budget classes
        # (2) budget/stop greedy over the rest, rarest-within-class first
        for g in (Q minus floor_seats):
            if sum_df + g.df > C: continue                          # SKIP over-budget (z-score order is NOT df-monotone), keep scanning (Section 5)
            admit(g)
            if sumE - c * sqrt(sumVar) >= target: break             # enough class-fair identification evidence; iid phi_d is the no-clustering reference
        if not P and Q: P = [Q[0]]                                   # honor Section 7: admit the rarest-within-class gram (Q[0]) even if over budget (one posting walk)
        # P is now non-empty: the empty-Q view was skipped above, and the line above rescues P from a non-empty Q
        if collected_energy_far_below(full_query): loosen_budget_or_restore_reserve()

        # per-gram count credit mu_g = max(0, logit(g.r)); one value per ORDER present (query-side: uniform SIGMA)
        for g in P: g.mu = max(0, logit(g.r))
        # concentrated(P): a dominant gram (top energy) AND >= 2 query-relative commons (E < E_top/2),
        # both ranging over ALL of P (floored included, literal Section 12) -- a genuinely rare gram is
        # itself floored (df <= sqrt N), so keeping it as E_top lets it anchor and TIGHTEN the cap that
        # shields it. An all-common query has no dominant gram -> not concentrated -> mu survives (Sec 7).
        if concentrated(P): for g in P: g.mu = min(g.mu, concentration_cap(P))

        # bit-sliced accumulation of the ENERGY part only (hot loop)
        assert DELTA < 2 * E_floored                                 # Section 7: keeps round(E_floored/DELTA) >= 1, so a floored gram's weight
        #                                                              #   never quantizes to 0 and vanishes from the union below
        for g in P: g.wq = max(0, round(g.E / DELTA))
        planes  = max(1, ceil(log2(max(g.wq for g in P) + 1)))       # floored at 1
        E_acc   = bitsliced_overlap(P, [g.wq], planes)               # {segment -> integer energy sum}
        # count credit is added per gram-ORDER: mu varies by order (doc-side), so the popcount is BUCKETED by n.
        # USE THE CAPPED per-order credit (the same g.mu above), NOT a recomputed logit, so the concentration cap
        # (mu = min(mu, cap) above) is not DEFEATED in the score and the null (which uses g.mu) still mirrors it.
        cred_acc = {seg -> sum over orders n of mu_capped[n] * popcount_n(seg)}   # mu_capped[n] = g.mu of any order-n non-floored gram in P

        # null = expected score of a random length-L segment; saturating presence pi_g = 1-(1-p_g)^(L/Lbar).
        # rare grams are ~linear -> separable; only commons need the per-candidate saturating term (Section 6).
        weight(g) = g.wq*DELTA + (g.mu if not g.floored else 0)      # per-gram credit
        K_rare  = sum( (g.df/N) * weight(g) for g in P if g.df/N <  P_LINEAR )    # precomputed once
        commons = [g for g in P if g.df/N >= P_LINEAR]
        scored = {}
        for seg in keys(E_acc) | keys(cred_acc):                     # union: count-only candidates survive
            null = (L[seg]/Lbar) * K_rare
            for g in commons: null += (1 - (1 - g.df/N)**(L[seg]/Lbar)) * weight(g)
            scored[seg] = E_acc.get(seg, 0)*DELTA + cred_acc.get(seg, 0) - null
        per_view.append(scored)

    if not per_view: return EMPTY                                    # every rank-view empty even after the fallback (e.g. a 1-char Latin query)
    if len(per_view) == 1:
        return emit(sort_desc(per_view[0]), with_components=True)
    w_primary, w_secondary = view_weights_from(dH)                   # per-(script,order) dH; combined / equal-weighted for mixed-script queries (Section 8)
    return emit(RRF(per_view, [w_primary, w_secondary], k_rrf, missing="omit"), with_components=True)
```

---

## 13. Summary

- Surprisal is energy — the reading's identification, exact in the rare tail where the reading is valid. The operating weight is its refinement: a matched gram is worth $E_g + \mu$, the binary-independence log-likelihood ratio, with energy $E_g = \log\frac{1-p_g}{p_g}$ (logit-idf, of which surprisal $-\ln p_g$ is the rare-gram limit). A reported score is a log-odds in nats in the clean limit, decalibrating under the recall-safe clamp, floor, quantization, and null of §4–§7.
- The absent-gram penalty is not discarded. Typo artifacts are inert, and a real gram's absence reorganizes into the count credit $\mu = \operatorname{logit} r$ plus a rank-invariant constant — a chemical potential governed by a single reliability $r$ that also drives the stopping rule.
- The credit is reliability-weighted count evidence, distinct from the corroboration the summed energy already supplies: a per-channel constant in value, modulated only by a structural concentration cap, and corruption-invariant per gram. (Reliability is a corpus property; the realized credit drifts upward only as junk leaks the imperfect floor, which is recall-safe.)
- Pruning is an information knapsack stopped by a distribution-free Cantelli bound. Its variance carries the co-failure of overlapping grams as per-word comonotone blocks — operationally the comonotone-block $\varphi = 1$ bound: recall-safe, PSD, and closing the within-word burst residual a pairwise overlap window would leave open, with the i.i.d. closed form kept as the no-clustering reference.
- Length bias is a credit artifact, removed by subtracting a saturating null that mirrors the accumulator weight term for term without over-penalizing long segments. Same-run gram orders are fused and cross-script orders pooled (§8), and the count credit is separated from the bit-sliced accumulator so the hot loop stays narrow and the work stays bounded by the segment-frequency budget.
- The thermodynamic lens supplies the organizing intuitions and marks its own boundary at the collective phenomena it cannot support; the probabilistic-IR model supplies the derivations.
- Almost nothing needs a corpus sweep. The doc-side channel reduces to one declarable ingestion error rate $\varepsilon$, together with the same topicality $\sigma$ the query side already needs, since $\rho=\sigma(1-\varepsilon)^n$ — fixing the reliability, the credit, and the variance's diagonal and i.i.d. reference, while the recall-safe operating covariance uses the parameter-free comonotone-block $\varphi = 1$ bound. The fusion weight and cap threshold self-calibrate from the index and the query, and the one relevance-bound quantity — the topicality $\sigma$ (the query-side reliability, and the doc-side topicality factor) — is low-sensitivity and safe at a constant default.

---

## References

[Shannon1948] C.E. Shannon. A Mathematical Theory of Communication. Bell System Technical Journal, 27:379–423, 623–656, 1948.

[Jaynes1957] E.T. Jaynes. Information Theory and Statistical Mechanics. Physical Review, 106(4):620–630, 1957.

[Jeffreys1946] H. Jeffreys. An Invariant Form for the Prior Probability in Estimation Problems. Proceedings of the Royal Society of London A, 186(1007):453–461, 1946.

[Cantelli1928] F.P. Cantelli. Sui confini della probabilità. Atti del Congresso Internazionale dei Matematici, Bologna, 1928. (The one-sided Chebyshev / Cantelli inequality.)

[SparckJones1972] K. Spärck Jones. A Statistical Interpretation of Term Specificity and its Application in Retrieval. Journal of Documentation, 28(1):11–21, 1972.

[RSJ1976] S.E. Robertson, K. Spärck Jones. Relevance Weighting of Search Terms. Journal of the American Society for Information Science, 27(3):129–146, 1976.

[Ukkonen1992] E. Ukkonen. Approximate String-Matching with q-grams and Maximal Matches. Theoretical Computer Science, 92(1):191–211, 1992.

[RobertsonWalker1994] S.E. Robertson, S. Walker. Some Simple Effective Approximations to the 2-Poisson Model for Probabilistic Weighted Retrieval. SIGIR '94, 232–241, 1994.

[Turtle1995] H. Turtle, J. Flood. Query Evaluation: Strategies and Optimizations. Information Processing & Management, 31(6):831–850, 1995.

[Berger1996] A.L. Berger, S.A. Della Pietra, V.J. Della Pietra. A Maximum Entropy Approach to Natural Language Processing. Computational Linguistics, 22(1):39–71, 1996.

[Singhal1996] A. Singhal, C. Buckley, M. Mitra. Pivoted Document Length Normalization. SIGIR '96, 21–29, 1996.

[O'NeilQuass1997] P. O'Neil, D. Quass. Improved Query Performance with Variant Indexes. SIGMOD '97, 38–49, 1997.

[Navarro2001] G. Navarro. A Guided Tour to Approximate String Matching. ACM Computing Surveys, 33(1):31–88, 2001.

[Rinfret2001] D. Rinfret, P. O'Neil, E. O'Neil. Bit-Sliced Index Arithmetic. SIGMOD '01, 47–57, 2001.

[LaffertyZhai2001] J. Lafferty, C. Zhai. Document Language Models, Query Models, and Risk Minimization for Information Retrieval. SIGIR '01, 111–119, 2001.

[Dhaene2002] J. Dhaene, M. Denuit, M.J. Goovaerts, R. Kaas, D. Vyncke. The Concept of Comonotonicity in Actuarial Science and Finance: Theory. Insurance: Mathematics and Economics, 31(1):3–33, 2002.

[Broder2003] A.Z. Broder, D. Carmel, M. Herscovici, A. Soffer, J. Zien. Efficient Query Evaluation Using a Two-Level Retrieval Process. CIKM '03, 426–434, 2003.

[Robertson2004] S. Robertson. Understanding Inverse Document Frequency: On Theoretical Arguments for IDF. Journal of Documentation, 60(5):503–520, 2004.

[ZhaiLafferty2004] C. Zhai, J. Lafferty. A Study of Smoothing Methods for Language Models Applied to Information Retrieval. ACM Transactions on Information Systems, 22(2):179–214, 2004.

[Cormack2009] G.V. Cormack, C.L.A. Clarke, S. Büttcher. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR '09, 758–759, 2009.

[Koroutchev2009] K. Koroutchev, J. Shen, E. Koroutcheva, M. Cebrian. Thermodynamics of Information Retrieval. arXiv:0903.2792 [cs.IT], 2009.

[DingSuel2011] S. Ding, T. Suel. Faster Top-k Document Retrieval Using Block-Max Indexes. SIGIR '11, 993–1002, 2011.

[Lemire2018] D. Lemire, O. Kaser, N. Kurz, L. Deri, C. O'Hara, F. Saint-Jacques, G. Ssi-Yan-Kai. Roaring Bitmaps: Implementation of an Optimized Software Library. Software: Practice and Experience, 48(4):867–895, 2018.
