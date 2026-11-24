# Published Biasi (2014) T700/M21 Benchmark Dataset (`data/biasi_t700m21/`)

Digitized Air-atmosphere TGA dataset for Torayca T700 / Hexcel M21 CFRP from:

> V. Biasi, *Characterization and modeling of composite materials under thermal degradation and fire*, PhD Thesis, ISAE-ENSMA, Chapter 4, Fig. IV.6 (p. 103), 2014.

- **`experimental_5K.csv`:** Remaining mass fraction across 4 heating rates in Air (`2, 5, 10, 20 K/min`) on a shared `5 K` grid from `505 K` to `1250 K` (`150` observation points per curve; `600` points total).
- **`coverage.csv`:** Digitized temperature-window coverage summary per heating rate.
- **Reference fiber mass fraction:** `0.656` (from Biasi 2014 Table IV.2, p. 102).
- **Delayed observation start handling:** Models integrate forward from initial unreacted state at `300 K` and compare predicted mass against experimental observations across the `505-1250 K` window.
