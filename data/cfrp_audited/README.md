# CFRP TGA Input Data Schema (`data/cfrp_audited/`)

Raw experimental CFRP TGA curves are omitted from the public repository release. All derived CFRP comparison tables, parameter uncertainty summaries, synthetic calibration benchmarks, and published Biasi (2014) T700/M21 experimental curves (`data/biasi_t700m21/`) remain included for verification and comparison.

## Expected Schema When Running on Your Own CFRP Dataset

To run `run_cfrp_solid_state.jl` on local audited CFRP data (via `--audit-folder=/path/to/cfrp_data`), supply a directory containing:

1. **`measurement_quality.csv`** — required columns:
   - `condition` (string run ID, matching `<condition>_observations.csv`)
   - `gas` (`"Air"` or `"Nitrogen"`)
   - `rate_C_min` (`2.5`, `5.0`, `10.0`, `20.0`)
   - `source` (path or identifier)
   - `source_sha256` (file SHA-256 hex digest)

2. **Ten `<condition>_observations.csv` files** (6 `Air`, 4 `Nitrogen` across `2.5, 5.0, 10.0, 20.0` °C/min) — required columns:
   - `condition`, `gas`, `rate` (must match `measurement_quality.csv`)
   - `time_min` (strictly increasing elapsed time in minutes)
   - `temperature_K` (positive temperature in Kelvin)
   - `mass_fraction` (remaining mass normalized to `1.0` at initial ambient state)

## Included Benchmark Dataset

For an immediately runnable end-to-end experimental dataset, see [`../biasi_t700m21/`](../biasi_t700m21/README.md) (`experimental_5K.csv`, `coverage.csv`) and `scripts/run_biasi_validation.jl`.
