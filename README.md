# Music's Effect on Plant Growth 

> Numerical analysis using **Sobel Edge Detection**, **Finite Differences**, **Nonlinear Optimization**, and **Newton–Raphson Root-Finding**


---

## Overview

Can music make plants grow faster? This project analyses two published experiments using three core numerical methods applied to real biological data — plant photographs and time-series measurements across multiple species.

The full pipeline is implemented in a single Python script (`main.py`) and produces one comprehensive output figure.

---

## Methods

| Method | Application |
|--------|-------------|
| **Sobel Edge Detection** (finite differences on images) | Measuring green pixel area from plant photos |
| **Central Finite Differences** | Computing growth rate dN/dt and acceleration d²N/dt² |
| **Nonlinear Least Squares** (Levenberg–Marquardt) | Fitting logistic growth model to time-series data |
| **Newton–Raphson Root-Finding** | Locating peak growth day from logistic inflection point |

### Logistic Growth Model

$$h(t) = \frac{K}{1 + e^{-r(t - t_0)}}$$

Parameters fitted per dataset:
- **K** — carrying capacity (maximum plant size)
- **r** — intrinsic growth rate
- **t₀** — inflection point (day of peak growth rate)

---

## Datasets

| # | Species | Source | Data Type |
|---|---------|--------|-----------|
| 1 | **Duckweed** (*Landoltia punctata*) | Wang et al. 2022 (PMC9839374) | Photos + frond count over 7 days |
| 2 | **Radish** | Science Fair Project | Height (mm) over 14 days |
| 3 | **Chickpea** | Chowdhury & Gupta, Gaia Campus 2021 | Germination count at days 2, 4, 6 |
| 4 | **Marigold** | Chowdhury & Gupta | Photos (music vs. meditation vs. noise) |
| 5 | **Bok Choy** (*Brassica rapa*) | Yeoh et al. 2024 | Photos (control vs. classical vs. rock) |

---

## Key Results

### Logistic Model — Fitted Parameters

| Group | K | r | t₀ | Peak Day |
|-------|---|---|-----|----------|
| Duckweed CK (control) | 58.2 | 0.3019 | 5.64 | 5.64 |
| Duckweed Music | 61.4 | 0.3281 | 5.49 | 5.49 |
| Radish Control | 948.8 mm | 0.4646 | 10.38 | 10.38 |
| Radish Music | 1187.3 mm | 0.3977 | 10.35 | 10.35 |

**Highlights:**
- Music-exposed plants reach a **higher carrying capacity K** in both species (+5.4% duckweed, +25% radish).
- Peak growth day is nearly identical across groups — music shifts growth **magnitude**, not timing.
- Chickpea germination rate was **2.6× higher** in the music group (6.5 vs 2.5 saplings/day, days 2–4).
- Newton–Raphson converged in **4 iterations** for all four datasets.

### Image Analysis — Green Pixel Area (×10³ pixels)

| Group | Day 0 | Day 4 | Day 7 |
|-------|-------|-------|-------|
| Duckweed CK | 1.07 | 1.70 | 1.78 |
| Duckweed Music | 1.38 | 2.23 | 2.50 |

---

## Output Figure

The script generates `plant_growth_analysis.png` containing 8 panels:

1. Duckweed photos (CK and Music, days 0/4/7) with Sobel edge overlay
2. Marigold photos (Indian music, meditation, noise) with Sobel overlay
3. Logistic growth curves — duckweed and radish
4. Chickpea germination bar chart
5. Growth rate dN/dt curves with Newton–Raphson peak-day markers
6. Green-area bar chart from photo analysis
7. Growth acceleration d²N/dt² from duckweed data
8. Summary table of fitted logistic parameters

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

> The duckweed figure is downloaded automatically from PubMed Central (PMC9839374) on first run. No other external files are required.

---

## References

1. Wang, J., Xu, C., Durrani, S., et al. (2022). Evidence for the role of sound on the growth and signal response in duckweed. *Plant Signaling & Behavior*, 18(1). https://pmc.ncbi.nlm.nih.gov/articles/PMC9839374/
2. All Science Fair Projects. *Classical Music and Radish Growth*. https://www.all-science-fair-projects.com/project1301_details.html
3. Chowdhury, A. R. & Gupta, A. (2021). *Effect of Music on Plants*. Gaia Campus. https://gaiacampus.com/effect-of-music-on-plants/
4. Yeoh, J. P. S., Zhang, Z., Koh, K. S., et al. (2024). Music for plants? An investigation into the impact of exposure to acoustic stimulus in bok choy (*Brassica rapa*). *ESI Culture*. https://doi.org/10.70082/esiculture.vi.677
