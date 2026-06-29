# Weighting and Pruning for a Fuzzy Lexical-Overlap Retrieval Engine
### A probabilistic-IR derivation under a thermodynamic lens

This document derives the scoring layer of a fast, typo-tolerant lexical-overlap retrieval engine — a component that streams a bounded set of candidate segments to a downstream reranker or rank-fusion ensemble. Its job is recall under a near-constant per-query work budget: given a short, possibly-corrupted query, surface every segment that plausibly matches, cheaply, and let a more expensive stage sort them.

The derivation rests on a standard probabilistic-IR model (binary independence / likelihood ratio). Layered over it is an information-theoretic and thermodynamic reading that earns its place as intuition and as a source of cross-checks: **surprisal is energy**, the **count credit is a chemical potential**, ranking sharpness is a **temperature**, and the matched score is a **free energy**. The lens illuminates the structure and occasionally flags a tempting idea that does not survive (§10); the likelihood-ratio model does the rigorous work.

Throughout, logs are natural (nats), scoring is "more is better" so energy is *added*, and the unit of indexing is a **segment** (a short document or document fragment; the engine is not aware of any application-level grouping of segments into larger documents).

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $N$ | number of indexed segments (the unit of retrieval; fixed for a given index snapshot) |
| $df_g$ | segment-frequency of gram $g$ — the number of segments containing it |
| $p_g = df_g/N$ | marginal probability that a random segment contains $g$ |
| $\kappa = 0.5$ | Jeffreys smoothing pseudocount (estimation correction) |
| $\nu$ | corroboration depth; sets the contamination floor $df_{\min}=N^{(\nu-1)/\nu}$ (default 2) |
| $df^{\mathrm{eff}}_g = \max(df_g, df_{\min})$ | floored document-frequency used in the weight |
| $E_g = \ln\dfrac{N - df^{\mathrm{eff}}_g - \kappa}{df^{\mathrm{eff}}_g + \kappa}$ | per-gram energy: the RSJ log-odds (logit-idf) of the smoothed, floored estimate; used clamped as $\max(0,E_g)$. Its rare-gram limit is the surprisal $\mathrm{idf}_g = \ln\frac{N}{df^{\mathrm{eff}}_g+\kappa}$ |
| $E_{\max} = \tfrac{1}{\nu}\ln N$ | single-gram energy ceiling implied by the floor |
| $\beta,\ Z$ | inverse temperature; partition function (Boltzmann form, §1) |
| $Q$ | the (deduplicated) set of query grams |
| $P \subseteq Q$ | the pruned set actually scored |
| $r$ | reliability: probability a relevant document matches a real query gram — a per-channel corpus constant ($\sigma$ query-side, $\rho$ doc-side) |
| $\sigma,\ \rho,\ \varepsilon$ | query-side / doc-side reliabilities and the doc-side per-character error rate, with $\rho = (1-\varepsilon)^n$; $\rho < \sigma$ when the corpus is noisier than the query |
| $\mu = \max(0,\operatorname{logit} r)$ | count credit — nats per matched, non-floored gram |
| $L_d,\ \bar L$ | distinct gram count of segment $d$; its corpus mean |
| $K_P$ | length-null slope (§6), summed over $P$, mirroring the accumulator weights |
| $c$ | stopping margin (a Cantelli parameter, not a z-score) |
| $\varphi_d$ | gram co-failure correlation at start-distance $d$; iid reference $(r^{d/n}-r)/(1-r)$, a lower bound — the recall-safe operating value is $\varphi=1$ (§5) |
| $k$ | target candidate-pool size; the stop aims for $\ln(N/k)$ nats |
| $C$ | work budget — the cap on $\sum_{g\in P} df_g$, bounding posting-list cost |
| $\Delta$ | quantization step for the bit-sliced energy weights |
| $\Delta H = H_3 - H_2$ | vocabulary-complexity gap between trigram/bigram df-distributions, setting the fusion weight (§8) |
| $k_{\mathrm{RRF}},\ w_{\mathrm{tri}},\ w_{\mathrm{bi}}$ | reciprocal-rank-fusion rank constant; per-view weights |

Logs are natural (nats); scoring is "more is better," so energy is added; a *segment* is the unit of indexing.

---

## 1. Surprisal is energy

Consider a distribution $\theta$ over the gram vocabulary. The maximum-entropy distribution consistent with a fixed mean energy $\sum_g \theta(g) E_g$ is the Boltzmann form

$$\theta(g) = \frac{1}{Z}\,e^{-\beta E_g}, \qquad Z = \sum_g e^{-\beta E_g},$$

with $\beta$ the inverse temperature. Reading this backwards — asking which energies reproduce the empirical corpus statistics $p_g = df_g/N$ (the fraction of the $N$ segments containing gram $g$) — fixes the energies, up to an additive constant, as the **surprisal**:

$$-\ln p_g = \ln\frac{N}{df_g} = \mathrm{idf}_g \qquad(\text{up to an additive constant}).$$

This is the keystone of the lens. Inverse document frequency *is* the energy a gram carries: rare grams are high-energy, common grams low-energy. The additive constant is immaterial — ranking is invariant to it. The surprisal is the energy the *lens* assigns; the energy the *scoring* uses is the exact likelihood-ratio refinement of §2 — the log-odds $\ln\frac{1-p_g}{p_g}$ (logit-idf), of which surprisal is the rare-gram limit. The two coincide for rare grams, which is the regime the lens is valid in anyway (§10), so "surprisal is energy" reads exactly true where it does any work. (The corpus marginals sum to the mean segment length, not to one, so $\theta = p$ is a gauge-fixing identification rather than a literal probability assignment; $Z$ is never used quantitatively.)

The maxent step is used here as inference — the variational principle that selects a distribution under a constraint — and it is scale-free. It is distinct from the *physical* thermodynamic reading (energy, collective entropy, phase behavior), which is meaningful only in aggregate and which we are careful not to lean on where the systems are small (§10).

---

## 2. Scoring as a likelihood ratio

Score the relevance of a segment $d$ to a query $Q$ by a per-gram log-likelihood ratio, treating grams as independent:

$$S_d = \sum_{g \in Q \cap d} \log\frac{P(\text{match}\mid R)}{P(\text{match}\mid \bar R)} \;+\; \sum_{g \in Q \setminus d} \log\frac{P(\text{miss}\mid R)}{P(\text{miss}\mid \bar R)},$$

where $R$ denotes relevance and a random (non-relevant) segment matches gram $g$ with its corpus marginal, $P(\text{match}\mid \bar R) = p_g$. This is the binary-independence (Robertson–Spärck-Jones) model, whose exact matched-gram weight is $\mu + \log\frac{1-p_g}{p_g}$ — the count credit $\mu$ of §3 plus a *logit-idf*. We adopt this logit-idf as the operating energy $E_g = \log\frac{1-p_g}{p_g}$, so the matched weight $E_g + \mu$ is the exact RSJ log-likelihood ratio and a reported score is an honest log-odds in nats — the property a downstream that *reads* magnitudes (weighted fusion, a learned reranker, a score threshold) depends on. The familiar summed-IDF overlap $\sum_{g\in Q\cap d}\mathrm{idf}_g$ is its rare-gram limit: dropping $\log(1-p_g)$ leaves the surprisal $-\ln p_g = \mathrm{idf}_g$, equal to $E_g$ in the rare tail and diverging only for common grams — the gap is $-\log(1-p_g)$, about $0.36$ nats at $p_g = 0.3$ and a full $0.69$ at $p_g = 0.5$, where a gram in half the corpus carries *zero* discriminative evidence that surprisal would still score at $0.69$. The matched sum is the matched part of the query's cross-entropy against the corpus marginal, and logit-idf is the consistent likelihood-ratio choice with surprisal as its approximation.

The full cross-entropy — the query's gram distribution scored against the corpus background — also includes the second sum above: a penalty for query grams *absent* from the document. It is tempting to read the matched-only score as simply discarding that penalty for robustness, but that is not what happens, and the difference is the substance of §3. A typo leaves no gram for the penalty to act on — the mangled gram drops out of the query entirely — but it *injects* an artifact in its place, and it is the artifacts, not the lost grams, that a naive penalty mishandles. Under the noise model the absent-gram penalty splits in two. Artifact grams — rare or absent in the corpus, and no more likely in a relevant document than in a random one — are **inert**: their match and miss log-ratios are both zero, so a document missing one is neither rewarded nor penalized. (This is exactly where raw KL on the unfiltered query distribution goes wrong: it treats artifacts as real words and penalizes relevant documents for lacking them. The contamination floor of §4 and this inertness are the mechanisms that prevent it.) **Real** query grams should appear in a relevant document, so their absence *is* evidence against relevance — and its reliability-bearing part is retained, reorganizing into a per-query constant (which drops from ranking) plus the per-match count credit $\mu$ of §3. The score thus takes an overlap-only *form* — the engine walks only the matched posting lists, never the absent grams — while fully accounting for the absent-gram information: a real gram's absence reorganizes into the count credit $\mu$ plus a per-query constant that drops from ranking, and the matched energy $E_g$ carries the $\log(1-p_g)$ factor; the artifact part is exactly zero. With logit-idf as the energy, nothing is approximated away — the overlap-only score is the exact binary-independence likelihood ratio up to the rank-invariant constant.

---

## 3. The noise model and the count credit

The absent-gram term, handled rather than dropped, is what produces the count credit $\mu$ and fixes its meaning. Two channels of noise are worth distinguishing, because they license different scoring.

### 3.1 Query-side noise (the default)

The corpus is clean; the *query* is corrupted by the user's typing. The received query grams split into **real** grams — correct fragments of the intended query that survived — and **junk** grams — substitution artifacts, which are rare or absent in the corpus.

For a real gram, a relevant (clean, on-topic) document contains it with reliability $\sigma$, close to one: $P(\text{match}\mid R) = \sigma$. For a junk gram, a relevant document contains the artifact only at its corpus rate, $P(\text{match}\mid R) \approx p_g = P(\text{match}\mid \bar R)$. The junk gram's matched and missed log-ratios are therefore both $\log 1 = 0$: **junk grams are inert — they carry no evidence either way.**

For the real grams, with $\#m$ matched out of $\#\text{real}$ present in the query, the absent-gram sum's reliability part regroups cleanly (its frequency part is the energy $E_g = \log\frac{1-p_g}{p_g}$ of §2):

$$\#m\,\log\sigma + (\#\text{real} - \#m)\log(1-\sigma) = \underbrace{\#\text{real}\,\log(1-\sigma)}_{\text{constant in } d} + \#m\,\operatorname{logit}\sigma.$$

Collecting per matched real gram, so that each contributes the full RSJ weight $E_g + \mu$:

$$\boxed{\;S_d = \sum_{g \in Q \cap d,\,\text{real}} E_g \;+\; \#m \cdot \mu, \qquad \mu = \operatorname{logit}\sigma = \log\frac{\sigma}{1-\sigma}.\;}$$

The flat per-match bonus $\mu$ is the count credit. In the thermodynamic reading it is a **chemical potential**: the conjugate to particle number, the reward for admitting one more matched "particle." Its sign and magnitude follow from the reliability — $\sigma > \tfrac12$ gives $\mu > 0$, with $\mu \to \infty$ as $\sigma \to 1$ (perfect reliability makes a missing query gram strongly disqualifying).

### 3.2 Doc-side noise (opt-in)

Here the query is clean but the corpus is noisy (OCR, user-generated text, spelling variants), so a relevant document contains a clean query gram only with reliability $\rho < \sigma$. There are no artifacts. The same algebra gives $\mu = \operatorname{logit}\rho$, now applied to *every* query gram.

This channel has a single underlying quantity. Model a relevant document as the intended on-topic text with each character independently corrupted at rate $\varepsilon$ (write $q = 1-\varepsilon$). A length-$n$ gram matches iff all $n$ of its characters survive, so

$$\rho = q^{\,n} = (1-\varepsilon)^{n}.$$

So the one declarable number $\varepsilon$ — a property of the *ingestion source*, not its content (clean structured text $\varepsilon \approx 0$, OCR $\varepsilon \approx 0.01$–$0.03$, heavy user text higher) — fixes $\rho$, the count credit $\mu = \operatorname{logit}\rho$, and (via §5) the entire stopping variance. $\rho < \sigma$ then holds whenever the corpus is noisier than the query channel, rather than being a separately tuned comparison. The per-character idealization is exactly that — real errors cluster (keyboard adjacency, systematic OCR confusions) — so $\varepsilon$ is a principled default; an application can also estimate it at index time from the fat tail of singleton grams that are one substitution from a common gram (a label-free, corpus-internal signal that needs no relevance judgments).

### 3.3 Reliability is a corpus constant

A single reliability $r$ — $\sigma$ for query-side, $\rho$ for doc-side — drives both the count credit $\mu = \operatorname{logit} r$ and the stopping rule (§5). The decisive property is that **$r$ is a corpus and relevance property, not a query property.** It is the probability that a relevant document contains a *correct* query gram, and that probability does not depend on how the user mistyped elsewhere in the query. "elephant" and "eelphant" share their surviving real grams, and a relevant document contains those reals at the same rate in both cases. Query corruption changes the query's *composition* — more junk, fewer surviving reals — but it cannot move $r$. The consequence for the count credit is developed in §9.

This is invariance *per gram*. It need not hold for the *survivor-averaged* reliability — the quantity the realized credit actually rides on — if true reliability varies by gram and corruption is gram-selective: when harder-to-spell words are both mistyped more often and carry different reliability, conditioning on "this gram survived" selects a biased subset, and the average reliability over survivors can drift with corruption even though each gram's value is fixed. This is a second-order effect in the same recall-safe direction as the floor leakage of §9, and it is set aside for the same reason (no per-gram reliability data day-one); it is flagged here so the per-gram invariance is not mistaken for the stronger averaged claim.

---

## 4. Per-gram weighting

Two finite-sample corrections refine the bare energy.

**Estimation.** The maximum-likelihood estimate $\hat p_g = df_g/N$ is overconfident in the rare tail: a gram seen once has a wide posterior on its true rarity. Jeffreys smoothing corrects it to $\hat p_g = (df_g + \kappa)/N$, and the energy is the log-odds of the corrected estimate:

$$E_g = \ln\frac{1-\hat p_g}{\hat p_g} = \ln\frac{N - df_g - \kappa}{df_g + \kappa}, \qquad \kappa = 0.5,$$

with rare-gram limit the smoothed surprisal $\mathrm{idf}_g = \ln\frac{N}{df_g+\kappa}$. This is always applied. It noticeably affects the continuous quantities (the pruning budget of §5) and is nearly invisible to coarsely-quantized weights. (The strict Jeffreys posterior mean is $(df_g+\tfrac12)/(N+1)$; the $N$-versus-$N{+}1$ normalization is dropped as immaterial at scale.)

**Contamination.** Under query-side noise, some grams are rare *because they are not language* — the substitution artifacts — and the energy mistakes them for highly informative. A floor caps their energy from below:

$$df^{\mathrm{eff}}_g = \max(df_g, df_{\min}), \qquad df_{\min} = N^{(\nu-1)/\nu}.$$

This is a reparametrization: choosing the floor as a power of $N$ caps the single-gram energy at a clean fraction of the identification budget,

$$E_{\max} = \ln\frac{N}{df_{\min}} = \tfrac{1}{\nu}\ln N.$$

Identifying one of $N$ segments costs $\ln N$ nats, so a ceiling of $\tfrac1\nu\ln N$ means **no single gram can identify a segment alone; at least $\nu$ matched grams must agree.** (The expression for $E_{\max}$ drops the $\kappa$ and the $\log(1-\hat p_{\min})$ correction — together $\lesssim\!10^{-2}$ in relative terms, since $\hat p_{\min}=N^{-1/\nu}\to 0$; logit-idf and surprisal coincide at the floor.) The parameter $\nu$ (default 2) is the corroboration depth. The floor is applied only on the query-side channel; under doc-side noise every gram is a genuine word and a low df is real information, so only the estimation smoothing applies.

A caution on reading $\nu$ as a typo margin: one substitution destroys not one gram but the $\sim n$ contiguous $n$-grams that span the changed character (in "hello", changing the second character kills two of the three trigrams), and surviving grams that are positionally adjacent share characters, so a single substitution can take both. Robustness therefore depends on the *positional spread* of the surviving grams, not merely their count, and the per-typo tolerance is closer to $(m-\nu)/n$ than $m-\nu$.

**Weights are linear in the energy.** Because the matched sum is a log-odds, weights must be linear in $E_g$ for the sum to remain interpretable and to compose additively with $\mu$. Weighting by a power of $df$ (exponential in information) would break this.

**A note on fragility.** It is tempting to measure a query's fragility — its vulnerability to a single lost gram — by the variance of its grams' energies (the heat-capacity analog). This is wrong. Variance is spread, but fragility is single-point-of-failure, and the two diverge: $\{12, 12, 3, 3\}$ has high variance yet is robust (two rare grams; lose one, the other carries), while $\{12, 3, 3\}$ has lower variance yet is fragile (one rare gram doing the work). The correct statistic is an order statistic — the top energy against the sum of the rest, $\max(0,\,E_{\text{top}} - \sum_{\text{rest}} E)$ — which gives $0$ for the robust case and a positive value for the fragile one.

---

## 5. Pruning

To bound tail latency, the query is pruned to a subset $P \subseteq Q$ before scoring. This is a knapsack: maximize collected energy $\sum_{g\in P} E_g$ subject to a work budget $\sum_{g\in P} df_g \le C$ (the posting-list cost). Value per unit cost is $E_g/df_g$, which decreases in $df_g$, so **rarest-first** is near-optimal.

One subtlety: every gram below the contamination floor has the same energy $E_{\max}$, so sorting by energy alone leaves a block of ties whose order is undefined, while the budget uses true $df$. Breaking ties by **true $df$ ascending** restores the value/cost greedy within that block (constant energy, so the ratio is maximized by the cheapest gram) and makes the kept sequence monotone in $df$, which lets the budget cutoff exit safely. Grams with $df = 0$ (artifacts with no postings) are dropped before pruning: they match nothing and only consume budget and weight range.

### How far to prune: a confidence-bounded stop

Identifying one of $N$ segments costs $\ln N$ nats; narrowing to a candidate pool of size $k$ costs $\ln(N/k)$. But the evidence a *truly relevant* document accumulates is random — it matches each kept gram only with reliability $r$ — so the stop should clear the target with margin, not on average. Modeling matches as Bernoulli($r$):

$$\text{stop when } \sum_{g\in P} r\,E_g \;-\; c\,\sigma_{\text{match}} \;\ge\; \ln(N/k), \qquad \sigma_{\text{match}}^2 = \sum_g r(1-r)E_g^2 + \sum_{g\ne h} E_g E_h \operatorname{Cov}(B_g, B_h).$$

Two points. First, over the handful of grams in a pruned query there is no central-limit regime, so $c$ is not a Gaussian z-score; the honest, distribution-free guarantee is Cantelli's, $P(\text{evidence} < \text{mean} - c\sigma) \le 1/(1+c^2)$ — at $c=2$ that is at most 20%, not the 2.3% a normal tail would suggest. Cantelli holds at any sample size, which is exactly why it is the right tool here. Second, the covariance term is positive, query-dependent, and must be carried — but it is *not* a free parameter; the noise model fixes it in closed form. Contiguous grams share a failure cause, so the true variance exceeds the independent estimate, and a constant inflation cannot bound an excess that grows with the query's contiguity. Co-failure runs only between grams whose character spans overlap, so the off-diagonal is an $O(|P|)$ sum over positionally adjacent kept pairs — the same order as the accumulation loop, computed from the gram positions the tokenizer already carries:

$$\sigma_{\text{match}}^2 = r(1-r)\Big(\sum_{g\in P} E_g^2 \;+\; 2\!\!\sum_{\substack{(g,h)\\ d=|\Delta\mathrm{pos}|<n}}\!\! \varphi_d\,E_g E_h\Big), \qquad \operatorname{Cov}(B_g, B_h) = \varphi_d\,r(1-r).$$

**The iid co-failure $\varphi_d$ is fixed in closed form (doc-side).** Under the per-character model of §3.2 — a relevant document is the intended text with each character independently intact with probability $q$, so $r = q^{n}$ — a length-$n$ gram matches iff its $n$ characters survive, and two grams at start-distance $d$ (with $0 < d < n$) span a union of $n+d$ characters and both match iff all of them survive. Hence, using $q = r^{1/n}$,

$$\operatorname{Cov}(B_g, B_h) = q^{\,n+d} - q^{2n} = r\big(r^{d/n} - r\big), \qquad \boxed{\;\varphi_d = \frac{r^{d/n} - r}{1 - r}.\;}$$

The ends check: $d \to 0$ gives $\varphi = 1$ (a gram is perfectly correlated with itself), $d = n$ gives $\varphi = 0$ (disjoint grams are independent), and $\varphi_d$ falls monotonically between. For trigrams only $d = 1$ and $d = 2$ are correlated. This $\varphi_d$ is *exact under iid-per-character corruption* — but iid-exact is not the same as conservative, and for the stop the distinction is what matters. Real errors cluster: a burst corrupting a contiguous run makes overlapping grams co-fail *more* than independent characters would, and holding the marginal $r$ fixed, that raises their covariance. The most it can reach is the comonotone (Fréchet) bound $\operatorname{Cov} = r(1-r)$, i.e. $\varphi = 1$, when the two grams' survival is perfectly coupled. So the true co-failure of an overlapping pair lies in $[\varphi_d^{\text{iid}},\,1]$, with the iid value at its *lower* end — and using it as the operating value under-counts the covariance, under-counts $\sigma_{\text{match}}$, and stops early, which is recall-*unsafe*. The recall-safe choice is therefore the upper end, $\varphi = 1$, for every co-failing pair, with the iid $\varphi_d$ kept as the no-clustering reference (and the magnitude-calibration value); an application can interpolate toward it using the realized adjacent co-failure rate from the singleton-tail signal of §3.2.

**Query-side has the same $\varphi = 1$ posture, and there it is exact.** When the *query* rather than the corpus is corrupted, a surviving real gram is itself intact, so its match failure is a relevant document not containing it — a *word*-level event, not a character one: a relevant document contains a whole query word or none of it. Grams within one query word are therefore comonotone *by construction*, so $\varphi = 1$ is exact, not a bound. (The hedge belongs instead on the *cross-word* assumption: grams in different query words are treated as independent, which slightly under-counts when two query words tend to co-occur in relevant documents.) So both channels use $\varphi = 1$ for co-failing pairs — within one query word query-side, within the character-overlap window $d < n$ doc-side — exact in the first case, the conservative Fréchet bound in the second. One residual is not covered by the window: a corruption burst *longer than a gram* couples non-overlapping grams ($d \ge n$) that the window treats as independent; the fully conservative limit would treat an entire contiguous run as comonotone, at the cost of heavy over-collection. The covariance bites whenever $r$ is meaningfully below one, most acutely doc-side, where relevant documents are themselves noisy.

(Floored grams are excluded from the stop's running mean on the query-side channel, for the same reason they receive no count credit in §9: a junk gram contributes zero identification power, and crediting it $r\,E_{\max}$ would stop collection early. A query that is all junk then never reaches sufficiency; its grams are rare, with short posting lists, so few segments match and it correctly emits the resulting sparse candidate set rather than filtering.)

### Reading the query before and after pruning

Pruning is a rarity-biased subsample — it keeps the rare tail and drops the common head — so the full and pruned queries carry different diagnostics. Comparing the pruned collected energy against the *full* query's distinguishes two kinds of starvation: if the full query had far more energy than the pruned set, the budget cut away usable signal (loosen the budget, or restore the cheap mid-rarity grams it dropped); if the full query was itself weak, no budget helps and the right move is to fall to a lower gram order (§8). The grams pruning held back are a **corroboration reserve**: under fragility, restoring those real mid-rarity grams is a better first response than dropping to bigrams, because they are genuine language rather than redundant sub-grams.

---

## 6. Length correction

A flat count credit $\mu > 0$ rewards matching *more* grams, and longer segments have more opportunities to match — particularly to chance-match common grams. Left uncorrected, this is a length bias toward long documents. Note that the energy term alone ($\mu = 0$) is already largely length-*insensitive*, because a long document's extra matches are precisely the low-energy common grams worth almost nothing — and under logit-idf the residual mid-rarity inflation is muted further, since the energy-part length slope $p_g\ln\frac{1-p_g}{p_g}$ is smaller than the surprisal slope $p_g\ln\frac1{p_g}$ and vanishes by $p_g = 0.5$; the bias is mainly an artifact of the count credit flattening the energy spread.

The principled correction subtracts a segment's *expected* score under the null that it is a random draw from the corpus. Under a bag model, a gram is present in a length-$L_d$ segment with probability $\pi_g(L_d) = 1 - (1-p_g)^{L_d/\bar L}$ — the presence rate consistent with the marginal $p_g$ at mean length $\bar L$, with length entering as the *relative* length $L_d/\bar L$. For a rare gram this is $\approx (L_d/\bar L)\,p_g$ (linear), but for a common gram in a long segment the linear form *exceeds one* — an impossible probability that over-debits the null — whereas $\pi_g$ saturates below one. The correction is then

$$\boxed{\;S^{\text{adj}}_d = \underbrace{\sum_{g\in P\cap d}\big(\max(0,E_g) + \mu\cdot\mathbb{1}[g\text{ non-floored}]\big)}_{\text{accumulator}} \;-\; \sum_{g\in P} \pi_g(L_d)\big(\max(0,E_g) + \mu\cdot\mathbb{1}[g\text{ non-floored}]\big), \qquad \pi_g(L_d) = 1-(1-p_g)^{L_d/\bar L}.\;}$$

For the rare grams that dominate a typical pruned set, $\pi_g \approx (L_d/\bar L)\,p_g$ is linear, so their contribution collapses to the separable $(L_d/\bar L)\,K_P$ with $K_P = \sum_{g\in P} p_g(\max(0,E_g) + \mu\,\mathbb{1}[\text{non-floored}])$ precomputed once; only the few common grams need the per-candidate saturating term. The saturation matters most in the count-and-length regime (common-heavy queries, energy $\approx 0$), where the null is the *entire* ranking signal: the un-saturated form would debit a long relevant segment that matched everything below shorter, weaker ones, and — if the emitted set is truncated to $k$ — push it out of the pool, a recall loss.

The governing principle is that **the null is the expectation of the exact statistic the accumulator computes, so it must mirror the accumulator's weights term for term** — summed over the *pruned* set $P$ (only those grams enter any score), with the energy clamped to non-negative (matching the accumulator's clamp), and with the count credit applied only to non-floored grams (matching where the accumulator actually adds it). A null that sums over the full query, or credits floored grams, or uses raw negative energies, reintroduces a length-correlated bias. The mirror of the accumulator *weight* is exact — the null tracks the *quantized* weight $\Delta\cdot\mathrm{round}(\max(0,E_g)/\Delta)$, not its continuous value — while the *presence* model is the bag-model $\pi_g$ above, exact in the rare tail and saturating (no longer first-order) for commons.

The $\mu$ part of the slope is the substantive term; the energy part self-damps as $p_g\ln(1/p_g)\to 0$ for rare grams but does not vanish for a mid-rarity set, so the correction is applied **unconditionally** rather than gated on $\mu > 0$. The energy-part length slope is the logit-idf slope $p_g\log\frac{1-p_g}{p_g}$ — strictly smaller than surprisal's and zero by $p_g = 0.5$ — so the mid-rarity residual is already muted by the §2 energy choice; the null cancels whatever remains. It is a per-candidate floating-point adjustment in the ranking step, outside the inner accumulation loop, so applying it always costs nothing in the hot path. Like the concentration cap (§9), this correction only earns its cost if the emitted set is **truncated** to roughly $k$ below the budget-bounded union: a consumer that re-scores the whole union downstream gets these refinements for free, but a recall stage that hands a reranker a bounded top-$k$ depends on them — which is why the saturation above is not optional.

---

## 7. Representation and the work budget

The accumulator weights are quantized to small non-negative integers, $\tilde w_g = \max(0, \mathrm{round}(E_g/\Delta))$, and accumulated by bit-slicing the matched posting lists. Because the count credit is a flat shift,

$$\sum_{g\in P\cap d}(E_g + \mu) = \underbrace{\sum_{g\in P\cap d} E_g}_{\text{bit-sliced}} + \mu\cdot\underbrace{|P\cap d|}_{\text{popcount}},$$

only the energy part is bit-sliced. The plane count is then at most $\lceil\log_2(E_{\max}/\Delta)\rceil$ — independent of $\mu$, and in practice the exact bit-width of the realized maximum weight — and the count credit is added in the post-pass as $\mu$ times a popcount of matched (non-floored) grams, the same quantity the engine's overlap mechanism already produces. This keeps the hot loop narrow and makes the count credit free to be zero or signed.

Two corners need pinning. The plane count is floored at 1 (an all-common query whose grams all quantize to zero would otherwise request zero planes), and the post-pass iterates the **union** of the bit-sliced candidates and the popcount candidates — so a segment that matched only zero-weight (clamped) grams still receives its count-and-length score rather than vanishing. The result is that a query with no rarity discrimination degrades gracefully to a count-and-length ranking instead of returning empty. (One precondition: a segment matching *only floored* grams is recovered through the bit-sliced side, since floored grams are excluded from the popcount; this holds only if the quantization is fine enough to keep their weight nonzero, $\Delta < 2E_{\max}$. A coarser step would zero them out and drop such segments.)

Finally, the count-credit and length terms are post-accumulation floats, so any top-$k$ truncation must be applied **after** them, on the corrected scores — never on the bare bit-sliced sum, which would truncate on the wrong ordering. The post-pass is cheap enough to run over the full candidate set first: the work budget $\sum_{g\in P} df_g \le C$ upper-bounds the candidate union ($\#\text{candidates} \le \sum df_g \le C$), so the post-pass is $O(C)$ and the inner loop $O(C\cdot\text{planes})$. "Near-constant work per query" is precisely "work bounded by the df budget."

---

## 8. Multiple gram orders

Bigrams and trigrams should not be pooled into one weighted sum. A bigram is a near-deterministic function of its containing trigram (`abc` implies `ab` and `bc`), so a single contiguous trigram match fires the count credit three times and adds the two sub-bigram energies on top — counting one piece of evidence as three, with the overcount correlated with contiguity and with length. Instead, the gram orders are scored as **separate views** and combined by reciprocal-rank fusion, which reads only ranks and is therefore robust to this additive double-counting. Per-query statistics (richness, fragility) are computed within each view.

Which orders to run is gated by richness. When the query is rich in rare trigrams, trigrams alone suffice and the bigram pass — whose posting lists are longer and costlier — is skipped. When the query is starved or heavily corrupted, the bigram pass is added: a substitution corrupts fewer of the longer-spanning bigrams, and there are more of them, so bigrams are the more robust, less selective layer.

The fusion weight between the two views is informed by how much richer the trigram inventory is than the bigram inventory — a vocabulary-complexity gap $\Delta H = H_3 - H_2$ between the two document-frequency distributions, computed once per index. This is a directional heuristic, not a conditional entropy even approximately: the two distributions are tabulated independently over different supports and are not a joint/marginal pair ($\sum_c df(xyc) \ne df(xy)$ in general), so $\Delta H$ supplies the sign and scale directly from the index — more trigram types than bigram types argues for weighting trigrams more — while the monotone map from $\Delta H$ to the fusion weight is a fixed shape choice (a linear map suffices). The fusion's rank constant plays the role of a temperature — flatter fusion trusts more candidates — and a segment retrieved by one view but not the other is simply omitted from the absent view's contribution rather than treated as worst-ranked.

---

## 9. The count credit as a policy

The count credit $\mu = \max(0, \operatorname{logit} r)$ is a **per-channel constant** (floored at zero for a recall stage, where penalizing matches would be counterproductive and over-retrieval is cheap). Its value differs between channels — $\operatorname{logit}\sigma$ query-side, $\operatorname{logit}\rho$ doc-side — and on the doc-side channel it legitimately tracks corpus noise through $\rho$. Within the query-side channel, the credit *parameter* does not vary with the degree of query corruption, because reliability $\sigma$ is a corpus constant (§3.3): corruption changes query composition — junk grams receive no credit, and there are fewer real matches to reward — not $\sigma$. As the user types and the query fills in, what changes smoothly is this junk exclusion and the count of real matches, not the parameter $\mu$.

The *realized* credit drifts slightly, and in one direction only. The credit is applied to the grams the floor judges real (non-floored), and the floor has a false-negative rate: a junk artifact that collides with a real corpus word ($df \ge df_{\min}$) clears the floor and collects the full $\mu$, despite its true reliability being near the corpus rate rather than $\sigma$. Such collisions grow with corruption, so the realized per-match credit is over-stated by an amount that rises with corruption. The drift is *upward* — over-crediting chance matches — hence in the recall-safe direction, and the length null absorbs part of it (it debits a high-$p_g$ leaked gram by $(L_d/\bar L)\,p_g\,\mu$). So the parameter is corruption-invariant; the realized credit is invariant only to the extent the floor separates real from junk. Correcting the residual would mean discounting $\mu$ by an estimated corruption level — worth doing only if a magnitude-consuming downstream needs the calibration (§10), not for recall, where the drift is already safe.

**What the credit is.** The count credit is *reliability-weighted count evidence*, not a corroboration reward. Corroboration — multiple independent rare matches reinforcing each other — is already rewarded by the summed energy, which is large for rare grams. The credit captures something separate: under high reliability, a relevant document is *expected* to contain query grams, so each match earns a flat bonus and each absence is mild evidence against. This is why the credit is withheld from floored grams on the query-side channel: a floored gram is junk-suspicious, of unknown reliability, and should not receive the reliability-based bonus. Doing so costs nothing for genuinely rare real grams (which a low df may have floored) — they already dominate on summed energy — and correctly denies it to artifacts.

**Concentration.** One structural case warrants reducing the credit below its baseline. When the pruned set holds a single dominant rare gram amid several common, easily-chance-matched grams, a large flat $\mu$ lets an off-topic segment that happens to hit the commons outweigh an on-topic segment that hit the one rare gram. The credit should then be capped so the count cannot swamp the discriminating gram. "Common" is defined *relative to the query itself*, not by a corpus cutoff: a gram counts as common here if its energy is well below the top gram's, say $E_g < \tfrac12 E_{\text{top}}$. The cap applies only when the set is genuinely concentrated — **a dominant gram** at the top energy *and* at least two such commons below it — so an all-common query, whose grams are comparable and have no dominant member, is left uncapped and degrades to count-and-length ranking (§7) rather than having its credit zeroed. For a concentrated set,

$$\mu \le \max\!\left(0,\ \frac{E_{\text{top}} - \sum_{g\in\text{common}} E_g}{\#\text{common} - 1}\right),$$

restricted to the query-relative commons, so that a query spread across several comparable rare grams — which has no dominant member, hence no commons under the relative definition — remains uncapped. The hard floor at zero, reached when several commons collectively outweigh the dominant gram, discards all count evidence in a single step; a smoother form that shrinks $\mu$ toward the cap is preferable in tuning. Because the threshold is relative to the query's own top energy, it self-calibrates per query with no corpus-specific cutoff; only the fraction ($\tfrac12$) and the hard-versus-smooth choice are universal shape constants. (Concentration is a property of query *structure*, not corruption level: a short clean query can be concentrated, and a heavily corrupted query can still be spread.)

---

## 10. The unifying picture and its limits

The whole system moves along a single axis with a chemical-potential and temperature reading. At one pole — clean, complete queries against a reliable corpus — the count credit is high, rarity and corroboration are trustworthy, the engine runs cold and selective on trigrams with tight pruning. At the other — starved or corrupted queries — composition shifts (junk excluded, fewer matches), the bigram layer comes in, the budget keeps its corroboration reserve, and the stop runs conservative. The engine emits a score together with its components (summed energy, match count, length). A downstream stage that consumes **ranks** loses nothing in any regime. A stage that consumes **magnitudes** is served a calibrated, log-odds-shaped score in the clean limit; the recall-safe approximations — the variance treatment, the floor's leakage — preserve ranks while eroding magnitude calibration as corruption rises, so magnitudes degrade gracefully rather than holding exactly under corruption.

| | clean / cold | starved / warm | doc-side (noisy corpus) |
|---|---|---|---|
| contamination floor | off / light | $N^{(\nu-1)/\nu}$, $\nu=2$ | off |
| count credit $\mu$ | $\operatorname{logit}\sigma$ (constant) | same, with composition shifting and the concentration cap if structurally concentrated | $\operatorname{logit}\rho,\ \rho=(1-\varepsilon)^n$ |
| credit on floored grams | — | no | yes |
| gram views | trigram only | trigram + bigram, fused | per query cleanliness |
| pruning budget | tight | keep reserve | tight |
| length correction | always on | always on | always on |
| stopping variance | small ($\sigma\to1$) | within-word covariance, $\varphi=1$ (exact) | overlap-window covariance, $\varphi=1$ bound (iid $\varphi_d$ is the lower reference) |

The lens is reliable for **local, per-gram** quantities — surprisal and its logit-idf refinement, the per-gram weight, the count credit as a form — which are well-defined for a handful of grams. It breaks for **collective** phenomena that require a thermodynamic limit. A "phase transition" in a four-gram query is noise, not criticality; there is nothing to anneal in one-shot retrieval; the entropy of mixing two gram orders collapses to the vocabulary-complexity gap of §8. The Clausius–Clapeyron relation is the sharpest example of the boundary: it governs the *coexistence* of two phases of one system at equal chemical potential, and bigrams and trigrams are not two phases of one system but two always-present refinements of the same text — so the relation has no content here, and its integrated form merely restates a log-ratio of presence probabilities. These are the cases where the analogy stops being a tool and starts being decoration, and the design does not use them.

---

## 11. Assumptions and open questions

- **Gram independence.** The likelihood ratio and the matched sum treat grams as independent. Within an order, adjacent grams overlap and are correlated. This correlation is accounted for in the stopping *variance* (§5) but not in the score *mean*, so a segment whose matches are contiguous is somewhat over-credited relative to one with the same match count spread out. For a recall stage this is acceptable — it is a precision distortion the reranker can undo — but correcting it would require a positional-overlap redundancy penalty, trading away the additive simplicity the design keeps. This is a deliberate choice, open to revisit.
- **The matched weight is the exact logit-idf** $\mu + \log\frac{1-p_g}{p_g}$ (§2), so reported scores are honest log-odds in nats; surprisal $-\ln p_g$ is its rare-gram limit, used nowhere operationally. The deliberate approximations in the weight are the smoothing and contamination floor (§4) and gram independence (below), not the common-gram inflation surprisal would have carried. Relative to surprisal, common grams now weigh less (zero at $p_g \ge \tfrac12$), so a relevant document whose only matches are common grams scores marginally lower — a slight recall cost carried by the count credit $\mu$ and the concentration cap rather than by common-gram energy.
- **The bag-model length null** (§6) uses the saturating presence $\pi_g = 1-(1-p_g)^{L_d/\bar L}$; it ignores within-segment burstiness and structure but captures the dominant length effect and no longer over-debits long segments on common grams.
- **The noise model is idealized.** The real/junk split is treated as sharp, with the floor as its detector; in reality the floor has a false-negative rate, and a small fraction of junk leaks through — which is what makes the *realized* count credit drift upward with corruption (§9), in the recall-safe direction. The invariance is of the credit parameter, not the applied credit (nor the survivor-averaged reliability, §3.3). A single reliability per channel, uniform across real grams, stands in for a per-gram quantity. For the stopping covariance, the per-character closed form $\varphi_d = (r^{d/n}-r)/(1-r)$ (§5) is *exact under iid corruption* but is the **lower** end of the admissible range: real errors cluster, raising true co-failure toward the comonotone bound $\varphi = 1$. The operating choice is therefore $\varphi = 1$ on co-failing pairs (the recall-safe upper bound, and exact query-side); the iid form is the no-clustering reference. A burst longer than a gram couples non-overlapping grams the overlap window neglects — a residual the window does not bound.
- **$\Delta H$** (§8) is a vocabulary-complexity gap used as a directional fusion heuristic, not a conditional entropy, for the structural reason noted there.
- **The concentration cap** (§9) has the right form and sign; its "common" threshold is query-relative ($E_g < \tfrac12 E_{\text{top}}$), so it needs no corpus cutoff, leaving only the fraction and the hard-versus-smooth clamp as universal shape constants to settle.
- **Parameters, and how few are corpus-bound.** What looks like a parameter sweep is mostly not. The **doc-side** channel reduces to one *declarable* number — the ingestion error rate $\varepsilon$ — from which $\rho = (1-\varepsilon)^n$, the count credit, and the stopping variance's diagonal and iid reference follow in closed form (§3.2, §5); an application declares it from provenance or estimates it label-free at index time. (The recall-safe operating covariance uses the parameter-free $\varphi = 1$ bound, not $\varepsilon$.) The **fusion ratio** comes from $\Delta H$, computed from the df distributions the index already holds, so it self-calibrates the moment data lands; only the monotone $\Delta H \to$ weight map is a shape choice. The **cap threshold** is query-relative (above). The one genuinely corpus- and relevance-bound quantity is the **query-side reliability $\sigma$** — it depends on what "relevant" means and cannot be read from index statistics — but its sensitivity is low (rare grams are dominated by summed energy, commons are bounded by the cap, and $\mu$ is inert on single-gram queries), so a high constant default ($\sigma \approx 0.9$) is safe day-one, with optional self-supervision from click/selection logs. The stopping margin $c$ and the work budget $C$ are latency/safety dials, not corpus quantities. So the irreducible corpus dependence is essentially $\varepsilon$ plus a low-sensitivity $\sigma$ default — both supplied without a sweep.

---

## 12. Algorithm

```text
# index-time, once per snapshot
constants(index):
    N    = segment_count
    Lbar = mean(distinct_gram_count(d) for d in segments)
    dH   = H(trigram_df_distribution) - H(bigram_df_distribution)

energy(df_g, channel, N):                                              # RSJ log-odds (logit-idf); surprisal is the rare-gram limit
    df_min = (channel == QUERY_SIDE) ? N**((nu - 1) / nu) : 0
    df_eff = max(df_g, df_min)
    return ln( (N - df_eff - KAPPA) / (df_eff + KAPPA) )               # KAPPA = 0.5; < 0 for p>0.5, zeroed by max(0,.) at use

score_query(query, channel):
    Q_strings = set(tokenize(query))                                  # deduplicated by string (presence is binary)
    views = needs_bigrams(richness_estimate) ? [TRIGRAM, BIGRAM] : [TRIGRAM]
    while views and not any(grams_of_order(v) for v in views):        # query shorter than the chosen order:
        views = [next_lower_order(views[0])]                          #   structural fallback trigram -> bigram -> unigram
    per_view = []

    for view in views:
        n = gram_order(view)
        r = (channel == DOC_SIDE) ? (1 - eps)**n : SIGMA              # eps = ingestion error rate; SIGMA ~ 0.9
        Q = [g for g in grams_of_order(view) if lookup_df(g) > 0]     # deduplicated strings; drop df=0 artifacts
        for g in Q:
            g.df = lookup_df(g);  g.E = energy(g.df, channel, N)
            g.floored = (g.df <= df_min)
            g.occ = occurrences_of(g, query)                         # [(pos, word), ...]; positional multiplicity kept

        # pruning: rarest-first, df-ascending tiebreak (kept sequence monotone in df)
        sort Q by (E desc, true df asc)
        P = [];  sum_df = 0;  sumE = 0;  sumVar = 0;  target = ln(N / k)
        kept_pos = {};  kept_word = {}                               # occurrence pos -> e ;  query-word -> [e, ...]
        for g in Q:
            if sum_df + g.df > C: break                              # safe: sequence is df-monotone
            P.append(g);  sum_df += g.df
            if channel == DOC_SIDE or not g.floored:                  # floored grams carry no identification power
                e = max(0, g.E)                                       # mirror the accumulator's clamp
                sumE   += r * e
                sumVar += r * (1 - r) * e * e                         # diagonal: B_g is per-string, counted once
                for (pos, word) in g.occ:                             # covariance edges per OCCURRENCE; phi = 1 (recall-safe Frechet bound)
                    if channel == DOC_SIDE:                           # co-failure within the character-overlap window
                        for d in 1 .. n-1:
                            for p in (pos - d, pos + d):
                                if p in kept_pos: sumVar += 2 * r*(1-r) * e * kept_pos[p]
                        kept_pos[pos] = e
                    else:                                            # query-side: grams within one query word co-fail
                        for h_e in kept_word.get(word, []): sumVar += 2 * r*(1-r) * e * h_e
                        kept_word.setdefault(word, []).append(e)
            if sumE - c * sqrt(sumVar) >= target: break              # iid phi_d=(r^(d/n)-r)/(1-r) is the lower-bound reference
        if not P and Q: P = [Q[0]]                                   # honor Section 7: admit cheapest gram even if over budget (one posting walk)
        if not P: return EMPTY                                       # genuinely no usable gram (e.g. 1-char query)
        if collected_energy_far_below(full_query): loosen_budget_or_restore_reserve()

        mu = max(0, logit(r))                                        # per-channel constant
        # concentrated(P): a dominant gram (top energy) AND >= 2 query-relative commons (E < E_top/2).
        # An all-common query has no dominant gram -> not concentrated -> mu survives (Section 7).
        if concentrated(P): mu = min(mu, concentration_cap(P))

        # bit-sliced accumulation of the ENERGY part only (hot loop)
        for g in P: g.wq = max(0, round(g.E / DELTA))
        planes  = max(1, ceil(log2(max(g.wq for g in P) + 1)))       # floored at 1
        E_acc   = bitsliced_overlap(P, [g.wq], planes)               # {segment -> integer energy sum}
        cnt_acc = overlap_count([g for g in P if not g.floored])     # {segment -> # matched non-floored grams}

        # null = expected score of a random length-L segment; saturating presence pi_g = 1-(1-p_g)^(L/Lbar).
        # rare grams are ~linear -> separable; only commons need the per-candidate saturating term (Section 6).
        weight(g) = g.wq*DELTA + (mu if not g.floored else 0)
        K_rare  = sum( (g.df/N) * weight(g) for g in P if g.df/N <  P_LINEAR )    # precomputed once
        commons = [g for g in P if g.df/N >= P_LINEAR]
        scored = {}
        for seg in keys(E_acc) | keys(cnt_acc):                      # union: count-only candidates survive
            null = (L[seg]/Lbar) * K_rare
            for g in commons: null += (1 - (1 - g.df/N)**(L[seg]/Lbar)) * weight(g)
            scored[seg] = E_acc.get(seg, 0)*DELTA + mu*cnt_acc.get(seg, 0) - null
        per_view.append(scored)

    if len(per_view) == 1:
        return emit(sort_desc(per_view[0]), with_components=True)
    w_tri, w_bi = view_weights_from(dH)
    return emit(RRF(per_view, [w_tri, w_bi], k_rrf, missing="omit"), with_components=True)
```

---

## 13. Summary

Surprisal is energy — the lens reading, exact in the rare tail where the lens does its work. The operating weight is its refinement: a matched gram is worth $E_g + \mu$, the binary-independence log-likelihood ratio, with energy $E_g = \log\frac{1-p_g}{p_g}$ (logit-idf, of which surprisal $-\ln p_g$ is the rare-gram limit) so a reported score is an honest log-odds in nats. The absent-gram penalty is not discarded: typo artifacts are inert, and a real gram's absence reorganizes into the count credit $\mu = \operatorname{logit} r$ plus a rank-invariant constant — a chemical potential governed by a single reliability $r$ that also drives the stopping rule. The credit is reliability-weighted count evidence, distinct from the corroboration the summed energy already supplies: a per-channel constant in value, modulated only by a structural concentration cap, and corruption-invariant per gram (reliability is a corpus property; the realized credit drifts upward only as junk leaks the imperfect floor, recall-safe). Pruning is an information knapsack stopped by a distribution-free Cantelli bound whose variance carries the co-failure of overlapping grams — operationally the comonotone $\varphi = 1$ bound, recall-safe, with the iid closed form kept as the no-clustering lower reference. Length bias is a credit artifact removed by subtracting a saturating null that mirrors the accumulator weight for weight without over-penalizing long segments; gram orders are fused, not pooled; and the count credit is separated from the bit-sliced accumulator so the hot loop stays narrow and the work stays bounded by the document-frequency budget. The thermodynamic lens supplies the organizing intuitions and marks its own boundary at the collective phenomena it cannot support; the probabilistic-IR model supplies the derivations. Almost nothing needs a corpus sweep: the doc-side channel reduces to one declarable ingestion error rate $\varepsilon$ — fixing the reliability, the credit, and the variance's diagonal and iid reference — while the recall-safe operating covariance uses the parameter-free $\varphi = 1$ bound, the fusion weight and cap threshold self-calibrate from the index and the query, and the one genuinely relevance-bound quantity, the query-side reliability $\sigma$, is low-sensitivity and safe at a constant default.
```
