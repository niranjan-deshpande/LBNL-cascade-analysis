# My Ideas
## Rough Model Design
- Seems like the model should be:
    - Use a naive ABM to simulate the old DFAX threshold in the old serial regime (or bound reallocation to a specific POI, since that's what we empirically looked at). Model decision to stay based on NPV using this threshold/rule. Then, simulate a world where we add the underfunded-upgrade deposit pool.
    - Need to figure out a way to model the shared POI-level shocks, which are the dominant drivers of withdrawals---the DiD only found an effect of 2-3 pp to 6-8pp (depending on how strict the control is). 
    - It's a little risky to bind reallocation to the POI---the actual DFAX metric isn't necessarily like this, but that's what we assumed in the empirical model. 
- For the NPV decision rule:
    - Have some cost tolerance heterogeneity---developers should sample from a distribution to decide whether or not to stay maybe? Should each developer have the same or diffferent distributions?
    - Include sunk costs already incurred, opportunity cost of capital
    - Include some stochasticity---simulates outside shocks to projects and investors
    - Restudy lag of 12-18 months, matching the DiD result found in our project.

## One-POI Toy Case
To develop our model, we start with a toy case: a single POI with 6 projects. We have the following specifications:
- ***Each project i has a state $\tilde{s_i}$.*** The components of $\tilde{s_i}$ are: allocated cost share $c_i$, sunk cost $s_i$, remaining expected future revenue $R_i$, idiosyncratic exit hazard $\lambda_i$, status $\in$ \{active, withdrawn\}.

A project stays (stay_i = 1) if $E[R_i] - C_i - U_i > \theta_i$, where $C_i$ are remaining capex costs and $U_i$ are assigned network upgrade costs and $\theta_i$ is a parameter that captures the value of leaving the queue. 

We can fold $E[R_i] C_i - \theta_i$ into a single term $H_i$, the "headroom" for each project. So the rule becomes: stay iff $U_i < H_i$. This is quite simple and lets us model most things quite easily. it also lets us insert POI-level differences into the model later quite easily, by shifting $H_i$ for a given POI. 

Then, we'll introduce some stochasticity since we don't want "knife-edge" behavior: let $H_{it} = H_i + \epsilon_{it}$, where $\epsilon$ is Gaussian in some appropriate way. 

We would later want to make $H_i$ a time-varying variable: as projects progress, their $C_i$ decreases and $E[R_i]$ grows (since we discount less). So we'd want to find a way to work that in. 

We want to model the 12-18 month lag somehow. The cleanest way to do this seems to be: when a project withdraws at $t$, $U_i$ for $i \in$ \{same POI} increases at time $t + \tau$ for some parameter $\tau$ (this could in principle vary across POIs, but need not).

One thought is that it would be important to build in some "delay" discounting---projects closer to completion have a higher PDV of the same nominal $R$. But we could probably just build this into the way the function $C_i$ decreases over time.

In our one-POI case, we'd probably just decide that there is initially some total amount of network upgrades $X$. The projects can share this cost proportional to their megawatts. This is actually different from DFAX weighting (they're correlated but not identical), but it's probably defensible here. We'd generate these projects by sampling from our data (bootstrapping), which seems quite defensible. Meanwhile, we'd also define $X$ by sampling from our data (relevantly; and we'd need to make sure that the MW totals and $X$ fit together coherently).

We should pick the headroom $H_i$ so that it matches the empirical ~27% completion rate---the projects that withdraw and the projects that succeed reveal some preferences around $H_i$. 

# How Cost Allocation Works in PJM---BROAD
1. DFAX cost-causation rule (the core allocation mechanism)
Network upgrade costs are split among projects whose distribution factor on the violated facility exceeds 3%. Share is proportional to DFAX-weighted MW contribution. This is the rule that makes withdrawal cascades mechanical: when a co-allocated project leaves, its share redistributes to remaining peers above threshold.
2. Two regimes coexist

Old serial regime (queue windows AD2 and earlier with executed ISA): first-come-first-served, unbounded reallocation across the queue. This is what your LBNL data captures.
New cycle regime (TC1, TC2, New Cycle 1+): cluster-based, first-ready-first-served, with explicit anti-cascade structure. This is the policy-relevant target for counterfactuals.

3. Within-cycle reallocation only
Under the new rules, cost reallocation is bounded to the cluster. A withdrawal in TC1 cannot reallocate onto a TC2 project. This is a hard structural break from the old regime.
4. Three-phase, three-decision-point cycle structure

Application: Study Deposit ($75K–$400K, 10% non-refundable) + RD1 ($4,000/MW) + 1-year site control
DP1 (post-Phase I): RD2 = 10% of allocated upgrade cost − RD1
DP2 (post-Phase II): RD3 (escalating)
DP3 (post-Phase III): RD4 = 100% security on allocated upgrades + 3-year full site control

5. Underfunded-upgrade deposit pool (the key anti-cascade mechanism)
Forfeited deposits from withdrawing projects are pooled and applied first to fill upgrade-cost gaps before any reallocation hits remaining peers. Residual is refunded pro rata. This is the structural buffer the new regime adds; its effectiveness depends on deposit-to-gap ratio.
6. Suspension rights eliminated
Under old rules, projects could suspend up to 3 years, creating overhang uncertainty. New rules force binary continue/withdraw at each DP.
7. Site control gates
Escalating site control requirements (1 yr → 1 yr + 50% interconnection facilities → 3 yr + 100%) act as a non-cost filter for speculative entry.
8. Generator-funded vs. regional cost split
Interconnection upgrades = generator-funded via DFAX (cascade-prone). RTEP regional upgrades = 50% solution-based DFAX + 50% postage stamp to load (no cascade). Policy proposals to shift more costs to the regional bucket would eliminate the cascade by removing the reallocation channel entirely.
9. Open Order 2023 compliance items
PJM filed compliance October 2025; further FERC-directed revisions pending on study cost allocation, network upgrade definitions, and study delay penalties. These could shift what counts as a cascade-eligible cost.
# How Cost Allocation Works in PJM---DETAILED


## 1. Two regimes operate in parallel right now

PJM is in the middle of a multi-year transition. The cost allocation rules differ depending on which regime your project sits in:

**Old "Rules" (serial, first-come-first-served)** — Projects in queue windows AD2 and earlier (submitted before April 1, 2018) that already had a Facilities Study or executable Interconnection Service Agreement (ISA) before the Transition Date continue under the legacy serial process. Existing projects in the queue that have received a facilities study or an executable interconnection agreement before the effective date of the PJM proposal will not be subject to the new queue reform and will proceed to interconnection under existing rules on a project-specific basis. Most of your LBNL Queued Up data (2000–2023) sits under this regime.

**New "Cycle" Rules (cluster-based, first-ready-first-served)** — Codified in Tariff Part VIII and PJM Manual 14H. Projects in queue windows AE1–AH1 that had not received an ISA by the Transition Date were swept into one of two Transition Cycles (TC1, TC2), and all post-October 2021 applications were withdrawn and must resubmit into New Cycle 1 (the first "true" cycle, with applications opening late 2025/2026). Late 2026 – Cluster Study Begins: PJM studies projects submitted in the intake. 2027–2028 – Deliverability Assessments & Upgrades Identified: Developers receive cost allocation results. 2029+ – Interconnection Agreements Executed.

This split matters for your ABM: the contagion mechanism you identified is a property of the **old** regime. The new regime is explicitly engineered to suppress it. So a useful framing is "ABM as a tool to evaluate whether the reforms actually kill the cascade, and what residual cascade exists."

## 2. How cost allocation works — the technical mechanics

### 2a. What gets allocated

PJM distinguishes several categories of cost (Manual 14H, definitions):

- **Attachment Facilities** — equipment between the generator's interconnection point and the transmission owner's system (e.g., the gen-tie). Sole responsibility of the requesting project.
- **Direct Connection Network Upgrades / Transmission Owner Interconnection Facilities (TOIF)** — facilities at the POI substation. Allocated to projects sharing the POI.
- **Network Upgrades / System Reinforcements** — broader transmission system modifications (line reconductoring, substation upgrades, even new lines hundreds of miles away) needed to maintain reliability post-interconnection. **This is where the cascade lives** — these are the costs that get reallocated when peers withdraw, and that are the main driver behind these increases has been broader network upgrade costs. For complete projects, network upgrade costs averaged $71/kW; for withdrawn projects, $563/kW.

### 2b. The allocation rule (cost causation via DFAX)

PJM uses a **flow-based, cost-causation methodology** for assigning network upgrade costs to interconnecting generators. From the project study reports: PJM shall identify the New Service Requests in the Cycle contributing to the need for the required Network Upgrades within the Cycle. All New Service Requests that contribute to the need for a Network Upgrade will receive cost allocation for that upgrade pursuant to each New Service Requests contribution to the reliability violation identified on the transmission system.

Concretely, the test for whether a project gets allocated cost for a given violated facility is a **distribution factor (DFAX) threshold**: that has ≥3% distribution factor (absolute value) for all facilities. If your project's incremental injection drives ≥3% of the additional flow on the overloaded line (in absolute value), you're "responsible" for some share of the upgrade. The share is proportional to your DFAX-weighted contribution.

This rule is what makes the contagion mechanical. If projects A, B, C all share ≥3% DFAX on a given overloaded line, and the upgrade costs $50M, the cost is split among them in proportion to DFAX-weighted MW. If A withdraws, B and C's shares mechanically grow — and the restudy triggered by A's withdrawal is what formally implements the reallocation.

Note also: There will be no inter-Cycle cost allocation for Interconnection Facilities or Network Upgrades identified in the System Impact Study costs identified in a Cycle; all such costs shall be allocated to New Service Requests in that Cycle. Within the new cycle regime, cost reallocation is **bounded to the cycle** — a withdrawal in TC1 cannot reallocate costs onto a TC2 project. This is a major structural change from the old serial regime, where a withdrawal could reallocate to any later-queued project.

### 2c. What "shared" looks like in practice

From the actual study reports for a real cluster of projects (AG1-389, AG1-390, AG1-392 series): AG1-392, AG1-392A, AG1-392B, and AG1-392C are behind the same POI. All four projects are in Transition Cycle 1 and will share costs for the required Transmission Owner Interconnection Facilities and physical interconnection Network Upgrades.

And the explicit cost-reallocation warning baked into every study report: For Project Developers with System Reinforcements listed: If this project presents cost allocation to a System Reinforcement indicates $0, then please be aware that as changes to the interconnection process occur, such as other projects withdrawing, reducing in size, etc, the cost responsibilities can change and a cost allocation may be assigned to this project.

This is the cascade mechanism made textually explicit. Even projects that initially have $0 allocated to a given upgrade are sitting in a "contingent cost" pool that activates when peers withdraw.

## 3. The reform: deposits, decision points, and the underfunded-upgrade pool

This is the most important structural shift for your modeling.

### 3a. Cycle structure

Under Manual 14H, every project moves through three Phases and three Decision Points:

- **Application** — Study Deposit ($75K–$400K depending on MW; 10% non-refundable). Plus **Readiness Deposit #1** = $4,000/MW. Plus 100% site control for at least one year.
- **Phase I System Impact Study** (~120 days) → **Decision Point I**. Continue requires Readiness Deposit No. 2 = 10% of cost allocation for network upgrades determined in Phase I, less Readiness Deposit No. 1. Site control required – 100% of the generating or merchant transmission facility for an additional one-year term AND 50% for interconnection.
- **Phase II** (~180 days) → **Decision Point II**. Readiness Deposit #3 (likely escalating share of allocated cost, plus expanded site control).
- **Phase III** (~180 days) → **Decision Point III**. Continuing customers must submit Readiness Deposit No. 4 = a security deposit equal to 100% of the network upgrades allocated to the customer. Site control required – 100% of the generating or merchant transmission facility AND 100% for interconnection facilities for a three-year term.

At each Decision Point, deposits become progressively at-risk. Withdrawing/terminated customers – all Readiness Deposits are considered at-risk, so may or may not be refunded. A customer would be reimbursed up to 90% of their study deposit, less actual allocated study costs.

### 3b. The underfunded-upgrade pool — the explicit anti-cascade mechanism

This is the mechanism most directly relevant to your ABM:

PJM will use retained deposits to fund underfunded network upgrades caused by withdrawn customers, as determined in Phase III. After all underfunded network upgrades are made whole, PJM will reimburse any remaining amounts on a pro-rata share.

Equivalently from Modo: Over successive phases, deposits become non-refundable. If a developer withdraws their project, PJM pools their non-refundable deposits to offset underfunded network upgrades related to late-stage withdrawals.

So under the new rules, the cascade mechanism is partially absorbed by the deposit pool: when a peer withdraws, their forfeited deposits go first toward filling the upgrade-cost gap before any reallocation to remaining peers occurs. Whether this fully neutralizes the cascade depends on (a) whether deposits are large enough relative to allocated upgrade costs, and (b) the timing of withdrawals (early-phase withdrawals forfeit less).

The structure of "carrot-and-stick" deposit forfeiture is described as: Developers must post Readiness Deposits ("RD") with their applications that increase in amount at the end of each phase. The amount of these deposits at risk of forfeiture if projects withdraw from the queue also increases to incentivize non-viable projects to exit the queue earlier in the process.

### 3c. Suspension rights eliminated

A subtle but important change: because of the cluster study approach, where a withdrawal or delay could trigger the need for restudies and there is a need to provide certainty in network upgrade cost allocation for remaining projects, PJM has proposed to eliminate the right currently available under PJM interconnection agreements of the project to suspend its application for up to three years. Under the old rules, projects could effectively park themselves and create uncertainty for peers; the new rules force a binary continue-or-withdraw decision.

## 4. The transmission cost allocation methodology more broadly (DFAX vs. postage stamp)

Worth distinguishing **interconnection-driven** allocation (above) from **RTEP/regional** allocation, since policy proposals sometimes target the latter:

- **Generator-funded interconnection upgrades**: cost-causation via DFAX (described above) — paid by the interconnecting customer.
- **Regional/baseline transmission upgrades** (RTEP under Schedule 12): hybrid allocation. For projects >= 500 kV: 100% of project cost is regionally allocated to PJM zones based on load-ratio share. For projects < 500 kV: Use the DFAX (Distribution Factor) Method. After the FERC remand, PJM adopted **solution-based DFAX**: FERC conditionally accepted a hybrid approach proposed by the PJM transmission owners for allocation of costs associated with regional facilities and necessary lower-voltage facilities. The hybrid approach is composed of allocations based on the solution-based DFAX method for 50 percent of the costs and a postage stamp basis for the remaining 50 percent.

This distinction matters because a major class of policy proposals — "shift more upgrade costs to load via postage stamp/regional allocation" — would essentially move costs out of the generator-funded bucket entirely, eliminating the cascade by removing the reallocation channel.

## 5. Open compliance items (as of late 2025/early 2026)

Order 2023 compliance is still being litigated. FERC approved several elements of PJM's existing process, including its three-stage cluster study design, interconnection application windows, site control rules, study deposit structure, readiness deposit framework, withdrawal penalties, and transition process but FERC rejected PJM's proposal to rely on "reasonable efforts" for reviewing interconnection requests and study timelines, instead requiring stricter accountability measures and study delay penalty structures. Additional directives included adopting FERC's required definitions for network upgrades, changing the allocation of study costs, allowing surety bonds, and ensuring transparency of interconnection data.

PJM filed compliance on October 22, 2025; further revisions are pending. The "definition of network upgrades" item in particular could shift what is and isn't subject to the cascade-prone cost-causation allocation.

There's also an open FERC proceeding on **Transmission Owner Self-Funding**, which is orthogonal to the cascade question but worth flagging: the Commission preliminarily finds that the existing tariffs of MISO, PJM, SPP and ISO-NE are unjust and unreasonable because they include a TO Self-Funding option. According to the Commission, the ability to elect to self-fund network upgrades increases the costs of interconnection service without improvements to that service and may do so in a manner that raises barriers to entry and creates opportunities for undue discrimination.

## 6. Synthesis for your ABM

The status quo gives you several distinct, calibratable mechanisms to model:

1. **DFAX-based cost-causation allocation with a 3% threshold** — the underlying allocation rule. Maps onto a network where peers share an upgrade if their DFAX-weighted contributions both exceed threshold; a withdrawal redistributes the absent project's share among remaining peers in proportion to their DFAX weights.

2. **Cluster boundaries** (new regime) — within-cycle reallocation only; no inter-cycle spillover. A binary parameter in the ABM: cascade ON within cluster, OFF across.

3. **Phase-gated, escalating deposit-at-risk schedule** — RD1 ($4K/MW), RD2 (10% of allocated upgrade cost net of RD1), RD3, RD4 (100% security). Withdrawal cost depends on phase at withdrawal.

4. **Underfunded-upgrade pool** — forfeited deposits absorb peer-withdrawal cost shocks before reallocation hits remaining peers. This is the central anti-cascade buffer in the new regime; its effectiveness depends on the ratio of pooled deposits to underfunded amounts.

5. **Restudy timing** — Phase III is when underfunded amounts get determined and reallocated. This roughly aligns with your DiD's 4–8 quarter delayed-positive finding, since cycle phases are ~6 months each.

6. **Site control / readiness gates** — independent of cost allocation but affect entry/exit rates and thus the cross-sectional clustering you observe.

The cleanest counterfactual experiments to run, given this status quo:
- **Vary the deposit-at-risk schedule** (e.g., what if RD2 were 50% of allocated cost instead of 10%?)
- **Vary the cluster boundary** (serial / cluster / cluster-with-spillover)
- **Vary the share of network upgrades funded via cost-causation vs. postage stamp**
- **Vary the DFAX threshold** (3% → 5% → 10%)
- **Add proactive transmission buildout** that reduces the depth of allocation per project

Want me to look up any specific element in more detail — e.g., the exact Phase III underfunded-upgrade restudy mechanics, or the New Cycle 1 application window specifics?