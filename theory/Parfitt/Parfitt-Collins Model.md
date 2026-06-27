# Parfitt–Collins Model (TRB Share Prediction)

``` toc
max_depth: 3
```


> [!abstract] In one line An **early-warning** system that predicts the _stabilised_ market share of a newly launched brand long before it is visible in sales data, by decomposing share into **Trial**, **Repeat (RBR)** and **Buying Index**.

## 1. Overview

The Parfitt–Collins model was first presented at the 1961 ESOMAR congress by Baum and Dennis, and then by Parfitt and Collins (1968). It uses real continuous purchasing data from **household panels** (e.g. an HUL panel, or a FairPrice Singapore loyalty panel) and predicts market share from three drivers:

- **Cumulative brand penetration** (cumulative brand trial)
- **Repeat-buying rate (RBR)**
- **Buying Index** (heaviness of category buying)

$$\text{Market Share} = \text{Trial} \times \text{Repeat} \times \text{Buying Index}$$

- **Trial** — projected proportion of homes that would _try_ the new product
- **Repeat** — projected proportion of those triers' category purchases that go to the brand once they have tried it
- **Buying Index** — how heavy the brand's buyers are _as category buyers_, indexed to the average

> [!info] Why predict early after a launch, much can be gained from knowing the ultimate outcome **before** it becomes apparent from historic sales or share trends. By the time the outcome is visible in the usual way, it is often too late for effective action. Market share fluctuates constantly — and most violently during launch, when consumers are trying the product and the manufacturer is promoting it. **It usually takes more than a year for the sales baseline to stabilise** into a narrower band. TRB targets that stabilised share.

### Trial and Repeat as growth drivers (FMOT / SMOT)

Sales are decomposed into the two fundamental drivers, **Trial** and **Repeat**, which reveal the consumer's desire to try the brand and their willingness to keep buying it after experiencing it:

- **FMOT** (First Moment of Truth) — modeled by **Trial**: the attractiveness of the product concept, how it is communicated, how well the brand is positioned. Other elements influencing trials are distribution, promotion and sampling.
- **SMOT** (Second Moment of Truth) — modeled by **Repeat**: the extent to which the product lives up to or exceeds expectations. A new product must build a base of regular consumers who keep buying after the trial.

> [!warning] Scope limitation The model struggles in categories with **low penetration** or **long inter-purchase intervals**, because too few repeat occasions are observed to stabilise the estimates in useful time.

---

## 2. Trial Index

Once the penetration curve's shape is known and a **declining rate of increase** is observed, the ultimate penetration can be estimated — there is no need to wait until the full curve has been observed.

$$\text{Trial} = \frac{\text{TrialRate（Product）}}{\text{TrialRate（Category）}}$$

- **Trial rate (product)** — projected cumulative penetration of the new product
- **Trial rate (category)** — projected cumulative penetration of the category over the same period

So Trial is the share of _reachable_ buyers obtained, where a reachable (potential) buyer is anyone who buys the **category**.

The trial index depends on:

- Product concept (price included) and its communication through advertising and packaging
- Product distribution
- Consumer promotions, including sampling

> [!note] Practical note (Ferrero context) Where the brand has strong recognition and rapid weighted distribution, the **trial rate is almost always high and often not very informative** — early trial is easy to achieve. In that setting the diagnostic focus shifts to **RBR**, which carries the real signal of product strength.

### 2.1 Penetration forecast

The increase in penetration per step is modelled as a **modified exponential** (the form Fourt–Woodlock proposed and Parfitt adopted after testing a family of growth models):

$$\Delta P(t) = a(K - P(t)) + \epsilon(t)$$

- $K$ — ultimate penetration (asymptote)
- $a$ — growth-rate parameter
- $\epsilon(t)$ — random error at time $t$

**Intuition.** The rate of increase of penetration is proportional to the _residual_ potential customers — those who could still try the SKU but have not. Early on there are many possible triers, so penetration grows fast; as the pool empties, growth slows. (Same shape as Newton's cooling / decay toward an equilibrium.)

**Estimation.** The discrete definition Parfitt prints is the **centred, symmetric** difference:

$$\Delta P(t) = \frac{P(t+1) - P(t-1)}{2}$$

Since $P(t)$ is known, the model equation is **linear in $P$**: regressing $\Delta P(t)$ **on** $P(t)$ gives a straight line with

$$\text{slope} = -a \qquad \text{intercept} = aK$$

so $a = -\text{slope}$ and $K = \text{intercept}/a$. Parfitt fits it with **discounted least squares** — an OLS variant that weights the most recent observations more heavily (the late, decelerating points most critically determine $K$); the paper uses $w = 0.6$ in a worked example.

> [!fidelity] What the paper does _not_ give Parfitt cites the discounted-least-squares method (Gilchrist) but **does not publish the exact estimation algorithm** — he calls Anscombe's MLE "arduous" and defers details to a thesis. The linear "$\Delta P$ on $P$" regression above is faithful to the _equation he prints_, not to the unpublished method. Note also a discrete→continuous gap: the centred-difference model is discrete, but the closed form below solves the _continuous_ ODE $dP/dt = a(K-P)$; the two coincide only approximately.

**Solution.** Discarding the noise and solving gives the penetration curve:

$$P(t) = K(1 - e^{-a t})$$

$K$ is the **asymptote**: $P(t) \to K$ only as $t \to \infty$ — it is _never reached exactly_. The fraction of $K$ reached at time $t$ is $1 - e^{-a t}$. Inverting, the periods needed to reach a fraction $f$ of $K$ are $t(f) = \ln\big(1/(1-f)\big)/a$:

| Periods after launch | % of ultimate penetration $K$ |
| -------------------- | ----------------------------- |
| $1/a$                | 63.2 %                        |
| $3/a$                | 95 %                          |
| $4.6/a$              | 99 %                          |

> [!tip] Why this matters for early warning The whole point is **not** to wait until $K$ materialises (which never happens in useful time) but to _estimate_ it early, once the curve decelerates, and project. The gap between _when you can estimate $K$_ and _when $K$ actually materialises_ is the margin for action. The "period" unit is whatever the calendar axis uses (week, month): read $1/a$, $3/a$, $4.6/a$ in that unit, or you will be off by a large factor across categories of different frequency.

---

## 3. Repeat-Buying Rate (RBR)

The ultimate success of a brand depends on the willingness of consumers, **once having tried it**, to keep buying it — usually to the partial or total exclusion of brands they bought before. Persuading a consumer to _try_ is a function of distribution, advertising and promotion; getting them to _keep buying_ is a function of their **acceptance of the product**.

RBR measures the propensity to keep buying after the first recorded purchase.

> [!important] Definition **RBR(t) is the brand's share of the trialists' category purchases, computed over _all_ eligible trialists (lapsed buyers included, who may have bought zero of the brand) and evaluated in the $t$-th interval after their first trial.** It is _not_ restricted to those who repeat-purchased — lapsed buyers stay in the denominator. This is exactly why the worked example below sums to 100 in the denominator.

**RBR(5) = 15 %** means that, for the average trialist, the new product makes up 15 % of their category purchases in the 5th interval after trial.

|Shopper|Vol. new SKU|Vol. category|First trial|Evaluated interval|
|---|---|---|---|---|
|Anita|10|20|January|5th (June)|
|Betty|0|60|March|5th (August)|
|Claire|5|20|April|5th (September)|

$$\text{RBR}(5) = \frac{10 + 0 + 5}{20 + 60 + 20} = \frac{15}{100} = 15\%$$

- The count starts **the period after** the first trial. Anita bought the SKU again (10 of her 20 category volume).
- **Betty bought 0 of the brand** in her 5th interval: she is a **lapsed** buyer, _not_ a repeater — but she remains in the calculation (numerator 0, denominator 60). RBR is based on **everyone who ever tried** the SKU, current and lapsed.

> [!example] Trial-anchored, not calendar time The intervals are anchored to **each shopper's own trial date**, not to the calendar. Anita (Jan), Betty (Mar) and Claire (Apr) are evaluated in _different calendar months_ (Jun / Aug / Sep) because each is in her own 5th interval after trial. This is the relative-time axis of RBR, distinct from the calendar-time axis of penetration and share.

![[Pasted image 20260620204357.png|300]]

> [!note] Reading RBR(t) (interval indexing, corrected) Each RBR(t) is **one interval wide** and is the $t$-th interval after trial; RBR(t) begins after $(t-1)$ full intervals have elapsed. What differs between examples is the **interval width**, which shifts where RBR(t) lands on the calendar:
> 
> - monthly intervals → RBR(5) ≈ the 5th month after trial;
> - 4-week intervals → RBR(5) covers ~weeks 16–20 after trial (the 5th four-week block);
> - 2-week intervals → RBR(5) covers ~weeks 8–10 after trial.
> 
> RBR(1), RBR(2), …, RBR(19) are all the same width — **they just start later**. _But_ the **cohort still changes** depending on whether you run the RBR after 1 year of data or after 2: the set of trialists eligible for interval $t$, and their composition, shifts with the analysis date.

### 3.1 The declining-then-stabilising pattern

As novelty wears off (less buzz, less promotion, less visibility), interest diminishes. So RBR tends to **decline before it stabilises**: typically RBR(1) > RBR(2) > RBR(3) … The early purchases are still exploratory and still benefit from launch promotion.

> [!key] Most important parameter **RBR stabilises after a few intervals for successful new products.** The stable RBR is the measure of consumers' willingness to keep buying the brand — the single most important parameter in TRB for assessing an innovation, especially where weighted distribution is near 100 % from the start (so trial cannot discriminate).

### 3.2 RBR by stage of entry (cohort refinement)

A prolonged study found that, on average, **the sooner a buyer enters the market for a brand, the higher their RBR**. Early entrants therefore have **disproportionate importance** in contributing to the ultimate share. Hence it is worth analysing RBR by **point of entry** (cohort = the period of first brand trial).

This yields greater accuracy in three ways:

1. The RBR of buyers **still to enter** is set at the **latest marginal rate**, not the average — the average would **overestimate**, because it is inflated by the high-RBR early cohorts while future entrants resemble the recent (lower-RBR) cohorts.
2. Knowing the RBR of the **most recent entrants** indicates whether it is worth pushing penetration further.
3. It clarifies the factors that determine the brand's share.

> [!note] Base (pooled) vs cohort RBR The **base RBR(t)** above is _pooled_ over all trialists. The **stage-of-entry analysis** is a separate, parallel calculation: group by cohort _from the start_ and recompute RBR within each cohort, keeping a common category base. It is the refinement, not the default.

---

## 4. Buying Index

$\text{Trial} \times \text{RBR}$ is a fair approximation of expected share, but it assumes all consumers are **equally important**. In reality some shoppers are **heavy category buyers**, and those are the ones a new SKU's cohort most needs.

$$\text{Buying Index} = \frac{\text{Category consumption by brand repeaters}}{\text{Category consumption by all category buyers}}$$

- _Category consumption_ = average category **volume** purchased per buyer.
- **Numerator**: average category volume of the brand's **repeaters** (2+ brand purchases) — a _subset_ of all category buyers.
- **Denominator**: average category volume of **all category buyers**.
- A buying index of **1.25** means the average brand repeater buys 25 % more of the category than the average category buyer.

> [!fidelity] Repeaters (Charan) vs triers (Parfitt) — why they coincide Parfitt's formal definition uses **all brand buyers (triers)**, but **evaluated on a window far from launch**. By that late window, anyone still buying the brand is effectively the loyal core — so a trier _in a late window_ is, over their lifetime, almost certainly a **repeater**. Charan formalises this by taking the repeaters directly. The two definitions therefore converge for the same reason: the late-window trier _is_ the repeater. The buying index is read on the **latest available window**, where the brand's buyers are overwhelmingly repeaters and very few are fresh triers.

---

## 5. The Model

$$\boxed{\text{Market Share} = \text{Trial Index} \times \text{RBR} \times \text{Buying Index}}$$

> [!summary] How the three factors fit together
> 
> - **Penetration / Trial** → the _ceiling_: the maximum share you could ever reach.
> - **RBR / Repeat** → you would _hold_ that ceiling only if your buyers (everyone who at least tried) kept buying only you.
> - **Buying Index** → how _heavy_ your repeaters are as category buyers.

Assuming an adequate sample and unchanging market dynamics, the model gives a reliable estimate of the brand's **baseline** share. Since dynamics do change, the estimate should be **revised periodically through the first two years** after launch.

### When TRB works best

- It works well at **high purchase frequency**: forecasts within a few weeks of launch are possible.
- The **ideal** case is a roughly **monthly** cycle (e.g. biscuits): stabilisation after ~5 iterations.
- It can also serve categories with **~4 purchase acts per year**, but stabilisation then needs 2–3 cycles — i.e. towards the **end of year 1**.

### Diagnosing the numbers

Sales patterns fluctuate with RBR and penetration growth:

- **Low RBR** → custom research is needed to find _why_ consumers do not repurchase (product issue).
- **RBR meets expectations but sales are low** → push promotion (sampling, price, display) and review proposition/communication to understand why trial is too low.
- Consumers who try **solely due to promotions** are less likely to repurchase: long-term share gains from extra penetration are eroded by the decline in RBR.

### KPIs to track

- Penetration
- Repeat rate
- Market share
- Buying behaviour via **overlap analysis** and **Gain & Loss**

> [!note] Gain & Loss G&L reveals the **sources of growth** of the new product — in particular whether **cannibalisation** is occurring.

### Assumptions behind the prediction

1. Retail distribution of the new brand is **uniformly high** — or at least not worse now than it will be in the foreseeable future.
2. The market will **remain the same** in future, apart from launch-related advertising/promotion (including competitors' retaliatory measures). A very successful competitor launch can invalidate the prediction.

### Validating predictions

Failure predictions must be **accepted as they are** (the brand is usually withdrawn, so accuracy can't be checked); successful ones can be studied for accuracy. The forecasting period is the **6-month window between months 12 and 18** after launch; the single predicted value is compared against **6 quotas**, each measured over a 4-week period.

> [!note] Why compare against an average Market shares are never perfectly stable — promotional pressure alone causes movement. A brand that is steady on an annual basis can swing significantly on a 4-week basis. The prediction is a single value, so it is compared to the **average** of the realised 4-week shares.

---

## 6. Refinements (from the appendix)

- **Early estimation of ultimate penetration.** A penetration model lets you estimate the ultimate level _before_ the curve shows a marked tendency to level off (valuable in itself). It also shortens the minimum study period and makes that period depend **only on purchase frequency** — so it can be set _before the study begins_. Even when the curve has already begun to flatten by the end of the minimum period, it is best to use the model estimate **so as not to underestimate**.
- **Quantifying "bought" penetration.** The biggest advantage: the model gives the _expected_ growth of new buyers if nothing changes. If you then run a price cut or aggressive sampling, **real** penetration rises above the **projected** curve; the gap is the penetration you _bought_, and applying the RBR of those specific shoppers tells you how much **share** you actually bought.
- **No clear category? Use absolutes.** If you cannot define a category for the product, compute with **absolute values**: quantity per buying household per 4-week period (how much triers repurchase), combined with absolute penetration. Always possible, but it ignores that total spend may rise/fall for everyone — so **use quotas (shares) where possible** to normalise.
- **Changing total-market level / seasonality.** A key assumption is that the **total market level is not affected** by the launch. Seasonal markets, and cases where the category itself grows, need correction: accumulate **both** total category buyers and brand buyers, and compare their **rates of increase**. The brand's marginal penetration contributes to its ultimate level _only when the brand's accumulation grows faster than the category's_. (In Parfitt's Figure 10, a brand's penetration appears to surge, but it is only a seasonal lift of the whole field — true penetration does not change.)

> [!note] No natural equilibrium between penetration and repeat There is **no fixed equilibrium** linking a given penetration level to a given repeat level. If penetration rises slowly but repeat is good, the levers are promotion, tasting, and improved distribution.

---

## 7. Marketing Factors (the strategic core)

> [!key] The non-obvious theorem Doubling penetration does **not** double share. A brand at 20 % penetration and 30 % repeat (→ 6 % share) does **not** reach 12 % by doubling penetration to 40 %. The extra 20 % of penetration is made of **late, marginal entrants**, whose repeat is **much lower** than the first cohort's. So the realised effect ranges from almost nothing (6 % → 6.5 %) to at most ~6 % → 9 %, never 12 %.

The trap is that the multiplicative formula _looks_ like it treats Trial and Repeat as independent. They are **not**: pushing penetration recruits lower-repeat buyers, so the marginal penetration carries **marginal repeat**, not average repeat. Multiplying "new penetration × old repeat" is exactly the error — the same `(would overestimate)` mechanism as the cohort refinement, here applied to a _decision_ instead of a _calculation_.

> [!key] Repeat is written before launch Penetration can be bought (distribution, advertising, promotion) almost at will. **Repeat cannot be bought after launch** — it is set at the **product-development stage**, and by the time the product reaches the market there is very limited room to move it. _Half the model is written before the product exists, and no budget can shift it._ This implies a **structural ceiling** on a product's share that post-launch marketing cannot break through.

### Promotions and buyer selection

- **Price cuts** easily lift penetration but add little to share: the buyers gained were prepared to be in the market **only at the reduced price**, so their RBR is **even lower** than normal, and they drop out when the price returns.
- **Other promotions (Brand G/H)** added penetration _and_ kept a healthy repeat. This is **not** a contradiction of the theorem: the variable is **which buyers a promotion selects**, not "promotion yes/no". A price cut **adversely selects** deal-seekers (low repeat by construction); other promotions select more normal buyers who, _if the product is accepted_, repurchase. Repeat always tracks **product acceptance**, never the marketing instrument.

> [!warning] Don't over-generalise Parfitt **explicitly refuses** to conclude "price cuts bad, other promotions good" — there aren't enough cases, and in highly price-elastic markets it could reverse. The robust conclusion is only: it is **easier to move penetration than ultimate share**, and how much share _follows_ penetration depends on the **method** used to raise it.

> [!note] Elasticity caveat Low repeat can sometimes be a **price-point** effect rather than rejection: shoppers may not repurchase at _this_ price but would at a lower one. Elasticity bounds the generalisation; it is not the explanation of the deal-seeker effect.

### Psychological segments (sweet spot)

There are psychological shopper traits affecting repeat:

- Those **least** willing to try novelties also have the **lowest** repeat if they do try.
- Those **most** willing to try novelties also tend to have **low** repeat — though, being eager triers, they move a lot of volume.
- The **sweet spot** is shoppers **moderately** willing to try novelties: they combine willingness to try with the **best repeat**.

> [!tip] Trade implication → deal-seekers Promotions improve share **when they do not attract deal-seekers**; in that case repeat holds. The recurring law: **repeat stays tied to product acceptance**, and the marketing instrument only determines _who_ you recruit.

### Use beyond new launches

The analysis also applies to **established brands** — e.g. to judge how much a promotion really contributes to share (analysis usually starts at the promotion). And it can be used during **product tests (SPE)** to assess a product's probability of success before launch.

---

## 8. One-paragraph synthesis

With **penetration** you learn the _maximum_ share you could reach; with **repeat** you learn that you would keep that share only if your buyers (everyone who at least tried) kept buying _only_ you; with the **buying index** you learn how heavy your repeaters are as category buyers. The buying index is read on the **latest window**, where the brand's buyers are almost all repeaters and very few are fresh triers. Early entrants carry the highest repeat and matter most for ultimate share; later entrants — especially those bought with price cuts — carry the lowest. So a brand has, in effect, a **structural share ceiling**, set largely by product acceptance before launch, that marketing can approach but not break.

---

## References

- Parfitt, J. H. & Collins, B. J. K. (1968). _Use of Consumer Panels for Brand-Share Prediction._ Journal of Marketing Research, 5(2), 131–145.
- Fourt, L. A. & Woodlock, J. W. (1960). _Early Prediction of Market Success for New Grocery Products._
- Gilchrist, W. G. (1967). _Methods of Estimation Involving Discounting._ JRSS Series B.
- [Why launch forecasting still needs its old equations: Parfitt-Collins, Fourt-Woodlock and Bass in the age of always-on marketing](https://marketing-mix.net/en/why-launch-forecasting-still-needs-its-old-equations-parfitt-collins-fourt-woodlock-and-bass-in-the-age-of-always-on-marketing/)


---
Stefano Villata | 2026-06-27, sa