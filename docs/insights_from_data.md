# Insights from LBNL Interconnection Queue Data

## Executive Summary

The planned regression/survival analysis of post-ISA queue outcomes is **feasible but significantly constrained** by data availability. The main bottleneck is not total sample size but the sparse and uneven reporting of `ia_date` across entities, which is the field required to compute the key dependent variable (`post_ISA_time`). PJM—originally the intended focus—reports essentially zero IA dates. Interconnection cost data covers a completely different set of BAs than those with IA dates, making the cost regressor unusable.

---

## 1. The Queued Up Dataset (`LBNL_queued_up.csv`)

### Overall Size
- **36,441 rows** (interconnection queue requests)
- Covers 55+ entities across 9 regions

### Identifying Post-ISA Projects
Using `IA_status_clean ∈ {IA Executed, Operational, Construction, Suspended}` yields **7,116 post-ISA projects**. Their current `q_status`:

| q_status     | Count |
|-------------|-------|
| operational | 3,441 |
| active      | 2,123 |
| withdrawn   | 1,184 |
| suspended   |   368 |

This is the broadest definition of "signed an ISA." However, only a subset have the dates needed to compute durations.

### The `ia_date` Availability Problem

Only **2,358 of 7,116 post-ISA projects** have a non-NA `ia_date`. Availability varies drastically by entity:

| Entity     | Post-ISA Total | Has ia_date | %   |
|-----------|---------------|-------------|-----|
| PJM       | 1,676         | 3           | 0%  |
| SPP       | 435           | 1           | 0%  |
| NVE       | 184           | 0           | 0%  |
| ISO-NE    | 172           | 7           | 4%  |
| Duke      | 166           | 0           | 0%  |
| MISO      | 1,172         | 339         | 29% |
| SOCO      | 249           | 89          | 36% |
| CAISO     | 462           | 321         | 69% |
| PacifiCorp| 443           | 335         | 76% |
| ERCOT     | 1,067         | 993         | 93% |
| IP        | 258           | 251         | 97% |

**PJM is completely unusable** for any analysis requiring `ia_date`. This is a data reporting limitation—PJM simply does not provide IA execution dates in its public queue data.

### Computable `post_ISA_time` Sample

To compute `post_ISA_time = (on_date or wd_date) - ia_date`, a project needs both `ia_date` and an outcome date. This yields:

- **912 resolved projects** (have ia_date + outcome date)
  - 731 operational (reached COD)
  - 149 withdrawn
  - 14 suspended
  - 18 active (with dates recorded)
- **884 right-censored projects** (active with ia_date but no outcome yet)
- **1,796 total** usable for a survival/hazard model

The 884 right-censored observations are usable in a Cox PH model—they contribute information about duration without withdrawal/COD.

### Entity Composition of Computable Sample

| Entity | Resolved |
|--------|----------|
| ERCOT  | 537      |
| CAISO  | 175      |
| SOCO   | 89       |
| IP     | 66       |
| MISO   | 35       |
| ISO-NE | 7        |
| PJM    | 2        |
| SPP    | 1        |

The computable sample is **dominated by ERCOT** (59% of resolved outcomes). The model is effectively "mostly an ERCOT model."

### Key Covariate Availability (Within Computable Sample)

| Planned Regressor           | Available? | Notes |
|----------------------------|-----------|-------|
| Resource type (`type_clean`)| Yes       | Good coverage. Solar (296), Wind (245), Gas (141), Other (100), etc. |
| Capacity (`mw1`)           | Yes       | 100% populated. Median ~100 MW, range -671 to 16,875 MW. |
| POI (`poi_name`)           | Partial   | 5,138 unique POIs across post-ISA, but many are "NA" or "Unknown." Very high cardinality—would need grouping. |
| State                      | Yes       | Good coverage. |
| Queue entry year (`q_year`)| Yes       | Good coverage, 2000–2024. |
| Service type (`service`)   | **Poor**  | **70% NA** in the computable sample. ERIS/NRIS distinction is mostly missing. |
| Interconnection cost       | **No**    | Cost data covers different BAs (see below). |

### Date Format Notes
- `q_date` is in ISO format: `YYYY-MM-DD`
- `ia_date`, `on_date`, `wd_date` are **Excel serial date numbers** (e.g., `42326` = 2015-11-18). These need conversion: `datetime(1899, 12, 30) + timedelta(days=serial_number)`.
- `q_year` is an integer year extracted from `q_date`.

---

## 2. Interconnection Costs Dataset (`ba_costs_2024_clean_data.xlsx`)

### Source
- LBNL Generator Interconnection Costs report (February 2026)
- Authors: Seel, Manderlink, Mulvaney Kemp, Rand, Gorman, Wiser (LBNL); Cotton, Porter (Exeter Associates)

### Structure
- **2,104 project-level cost records** in the "Project Cost Data" sheet
- Key columns: `Project #`, `Project # in Queued Up` (merge key), `Queue Date`, `BA`, `State`, `Fuel`, `Nameplate MW`, `Request Status`, `Service Type`, `$2024 POI Cost/kW`, `$2024 Network Cost/kW`, `$2024 Total Cost/kW`
- 1,841 projects have a `Project # in Queued Up` for merging with the queue dataset

### BA Coverage (Cost Data Only)
| BA  | Count |
|-----|-------|
| PAC (PacifiCorp) | 1,117 |
| DEP (Duke Energy Progress) | 300 |
| BPA | 296 |
| DEF (Duke Energy Florida) | 204 |
| DEC (Duke Energy Carolinas) | 187 |

**Critical finding:** The cost dataset covers only 5 non-ISO BAs. It does **not** include PJM, MISO, ERCOT, CAISO, SPP, or any ISO. This means interconnection cost data is fundamentally unavailable for the entities that provide IA dates.

### Merge with Post-ISA Projects
- **521 post-ISA projects** have matching cost data
- These are concentrated in PacifiCorp (293) and IP (167), with some Duke (107) and CAISO (96)
- Almost no overlap with the ERCOT-dominated `ia_date` sample

---

## 3. The Full Queue XLSX (`lbnl_ix_queue_data_file_thru2024_v2.xlsx`)

### Structure
- 38 sheets including raw data, codebook, and pre-computed analyses
- Key sheets:
  - `03. Complete Queue Data`: Same 31 columns as the CSV, plus the row data
  - `04. Data Codebook`: Field descriptions (see below)
  - `17. Cap. with IA`: Capacity of requests with signed/draft IAs by region
  - `25. Post-IA Completion`: Pre-computed post-IA status breakdowns by IA year (2000–2021). Shows 3,805 total ISA projects with resolved status: 1,878 operational, 509 active, 39 suspended, 1,379 withdrawn.
  - Sheets 26–38: Duration analyses (IR to WD, IR to IA, IA to COD, IR to COD) by region, type, size, service

### Codebook (Sheet 04)
| Field | Description |
|-------|-------------|
| `q_id` | Queue ID number |
| `q_status` | Current queue status (active, withdrawn, suspended, operational) |
| `q_date` | Interconnection request date (date project entered queue) |
| `prop_date` | Proposed online date from interconnection application |
| `on_date` | Date project became operational (if applicable) |
| `wd_date` | Date project withdrawn from queue (if applicable) |
| `ia_date` | Date of signed interconnection agreement (if applicable) |
| `IA_status_raw` | Interconnection study phase / status from queue (raw) |
| `IA_status_clean` | Standardized interconnection study phase / status |
| `poi_name` | Point of interconnection name |
| `region` | Region (ISO or non-ISO region) |
| `entity` | Transmission provider entity name (ISO or utility) |
| `service` | Interconnection service type (e.g., ERIS or NRIS) |
| `type_clean` | Resource type—standardized |
| `mw1/mw2/mw3` | Capacity of each resource type (MW) |
| `q_year` | Year project entered queue |

---

## 4. Feasibility Assessment for Planned Analyses

### Regression/Survival Model

**Feasible with constraints.** A Cox proportional hazards model with ~1,796 observations (912 resolved + 884 censored) and ~149 withdrawal events can support **4–5 covariates** comfortably (rule of thumb: 10–20 events per covariate degree of freedom).

Realistic regressor set:
1. Resource type (Solar/Wind/Gas/Other → 3 df)
2. Capacity (mw1, continuous → 1 df)
3. Entity (ERCOT/CAISO/SOCO/Other → 3 df)
4. Queue entry year (continuous or binned → 1–2 df)
5. State (high cardinality—would need grouping or dropping)

**Must drop from plan:**
- Interconnection cost (no overlap between cost BAs and ia_date BAs)
- Service type / ERIS vs NRIS (70% NA in sample)
- POI/zone (too high cardinality for 149 events; could try POI cluster count as a simpler proxy)

**Cannot restrict to PJM.** PJM has 3 ia_dates total. A PJM-only analysis is impossible with these data.

### Agent-Based Model / POI Cluster Withdrawal Correlations

**More promising.** This analysis doesn't require `ia_date`—it uses `q_status`, `cluster`, and `poi_name` fields, which are available for all 7,116 post-ISA projects (or even the full 36,441 dataset). Showing correlations between withdrawal rates within POI clusters is feasible with this data volume.

### Conditioning Strategy

Rather than restricting to one ISO, the recommended approach is:
- Pool all entities that report `ia_date` (primarily ERCOT, CAISO, PacifiCorp, IP, MISO, SOCO)
- Include entity as a stratification variable in the Cox model (allows baseline hazard to differ by entity)
- Report entity-specific results where sample permits

Alternatively, for a simpler logistic regression predicting withdrawal (yes/no) rather than time-to-event, the full 7,116 post-ISA sample is usable since it doesn't require `ia_date`—only `q_status`.

---

## 5. Data Quality Notes

- **Date formats are mixed:** `q_date` is ISO-8601 (`YYYY-MM-DD`), but `ia_date`, `on_date`, and `wd_date` are Excel serial numbers. Must convert consistently.
- **Negative capacity values exist:** `mw1` has values as low as -671 MW (likely load or derating entries). Filter or investigate.
- **`q_year` has anomalies:** 56 projects show `q_year = 1970`, and 101 show `q_year = NA`.
- **POI names are messy:** Many NA/Unknown entries, inconsistent naming conventions across entities. Not suitable as a regression variable without significant cleaning.
- **`IA_status_clean` vs `q_status`:** These track different things. `IA_status_clean` reflects the furthest interconnection study phase reached. `q_status` reflects the current overall queue status. A project can be `IA_status_clean = "IA Executed"` but `q_status = "withdrawn"` (signed ISA then withdrew).
