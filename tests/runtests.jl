using Test, CSV, DataFrames, Statistics, ForwardDiff, SHA

const ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(ROOT, "src", "common.jl"))
include(joinpath(ROOT, "src", "uncertainty", "KineticUQ.jl"))
include(joinpath(ROOT, "src", "material_kinetics", "uq_reporting.jl"))
include(joinpath(ROOT, "src", "solid_state", "CFRPSolidState.jl"))
using .KineticUQ, .MaterialUQReporting
using StableRNGs: StableRNG

function build_synthetic_cfrp_dir(dir::AbstractString)
    spec = CFRPSolidState.cfrp_model_spec(:M2_1Mat_GeomOx)
    x_true = [-1.8, 1.2, 2.0, -2.0, 1.65, -2.1, 1.5, 0.585, 0.21]
    conds = [
        ("air_2p5_original", "Air", 2.5),
        ("air_5_original", "Air", 5.0),
        ("air_2p5_t2", "Air", 2.5),
        ("air_5_t2", "Air", 5.0),
        ("air_10_t2", "Air", 10.0),
        ("air_20_t2", "Air", 20.0),
        ("n2_2p5_t2", "Nitrogen", 2.5),
        ("n2_5_t2", "Nitrogen", 5.0),
        ("n2_10_t2", "Nitrogen", 10.0),
        ("n2_20_t2", "Nitrogen", 20.0),
    ]
    T_grid = collect(300.0:3.0:1200.0)
    mq_rows = NamedTuple[]
    for (cid, gas, rate) in conds
        time_min = (T_grid .- first(T_grid)) ./ rate
        t_sec = 60.0 .* time_min
        idx = collect(1:length(T_grid))
        r_stub = (id=cid, gas=gas, air=(gas == "Air" ? 1 : 0), rate=rate,
                  time=t_sec, temperature=T_grid, observation_indices=idx,
                  obstime=t_sec, T=T_grid, mass=ones(Float64, length(T_grid)), grids=Dict{Float64,Any}())
        mass = CFRPSolidState.predict_cfrp(spec, x_true, r_stub; step_C=0.25)
        obs_df = DataFrame(
            condition=fill(cid, length(T_grid)),
            gas=fill(gas, length(T_grid)),
            rate=fill(rate, length(T_grid)),
            time_min=time_min,
            temperature_K=T_grid,
            mass_fraction=mass,
        )
        obs_path = joinpath(dir, cid * "_observations.csv")
        CSV.write(obs_path, obs_df)
        push!(mq_rows, (
            condition=cid,
            gas=gas,
            rate_C_min=rate,
            source="synthetic/" * cid,
            source_sha256=bytes2hex(sha256(cid)),
        ))
    end
    CSV.write(joinpath(dir, "measurement_quality.csv"), DataFrame(mq_rows))
    return (spec=spec, x_true=x_true)
end

@testset "CFRP Solid-State Kinetics Standalone Suite" begin
    @testset "1. KineticUQ AR(1) thresholds, noise & bootstraps" begin
        @test abs(KineticUQ.norminvcdf(0.975) - 1.95996398454) < 1e-6
        @test abs(KineticUQ.f_quantile_1_df(1e6; level=0.95) - 3.8414588) < 1e-3

        resids = [0.01 .* sin.(collect(range(0, 6pi; length=150))) for _ in 1:4]
        thr = KineticUQ.effective_uq_thresholds(resids, 8; level=0.95)
        @test thr.n_obs == 600
        @test thr.n_eff < thr.n_obs
        @test thr.loss_ratio_f > thr.loss_ratio_naive

        rng = StableRNG(2026)
        e1 = KineticUQ.generate_ar1_noise(200, 0.90, 0.015; rng=rng)
        @test length(e1) == 200 && abs(std(e1)) > 0.005

        @test isempty(MaterialUQReporting.evaluate_guardrails(
            [2.5, 5.0, 10.0],
            (n_eff=22.0, median_lag1_autocorr=0.90, loss_ratio_f=1.35, loss_ratio_naive=1.003),
            (summary=DataFrame(parameter=["logk1", "E1"], identifiability=["identifiable", "identifiable"]),),
            (compensation=DataFrame(corr_E_log10A=[0.85]),),
            nothing; k_params=3, near_bounds=String[]
        ).alerts)
    end

    @testset "2. Synthetic 10-run CFRP schema, CLI --audit-folder & PEtab parity" begin
        mktempdir() do tmp
            synth = build_synthetic_cfrp_dir(tmp)
            runs = CFRPSolidState.load_cfrp_runs(tmp)
            @test length(runs) == 10
            @test count(r -> r.gas == "Air", runs) == 6
            @test count(r -> r.gas == "Nitrogen", runs) == 4
            @test all(r -> 1 in r.observation_indices, runs)

            for mk in (:M1_1Mat_1stOrder, :M2_1Mat_GeomOx, :M2b_2Mat1st_GeomOx, :M3_1Mat_RPMOx, :M5_2Mat_FreeOrder_Cold)
                @test !isempty(CFRPSolidState.cfrp_model_spec(mk).names)
            end

            nv = CFRPSolidState.verify_native_cfrp(synth.spec, synth.x_true, runs; step_C=0.25)
            @test nv.passed
            pv = CFRPSolidState.verify_cfrp_petab(synth.spec, synth.x_true, runs; rtol=1e-7)
            @test pv.passed && pv.max_gap_pp < 0.05

            cmd = `julia --project=$(ROOT) -O1 $(joinpath(ROOT, "run_cfrp_solid_state.jl")) --audit-folder=$(tmp) --check-inputs`
            out = read(cmd, String)
            @test occursin("Runs loaded:  10", out)
        end
    end

    @testset "3. Biasi (2014) T700/M21 loader, fixed_fiber & delayed-start PEtab" begin
        runs = CFRPSolidState.load_biasi_as_cfrp_runs()
        @test length(runs) == 4
        @test [r.rate for r in runs] == [2.0, 5.0, 10.0, 20.0]
        @test all(r -> !(1 in r.observation_indices), runs)
        @test all(r -> first(r.temperature) == 300.0 && first(r.T) == 505.0 && last(r.T) == 1250.0, runs)

        spec_fix = CFRPSolidState.cfrp_model_spec(:M2_1Mat_GeomOx; fixed_fiber=0.656)
        spec_free = CFRPSolidState.cfrp_model_spec(:M2_1Mat_GeomOx)
        @test length(spec_fix.names) == 8 && !("fiber_fraction" in spec_fix.names)
        @test_throws ErrorException CFRPSolidState.cfrp_model_spec(:M2_1Mat_GeomOx; fixed_fiber=0.0)

        x_fix = [-2.0, 1.15, 1.8, -2.0, 1.65, -2.1, 1.55, 0.35]
        x_free = [-2.0, 1.15, 1.8, -2.0, 1.65, -2.1, 1.55, 0.656, 0.35]
        @test maximum(maximum(abs.(
            CFRPSolidState.predict_cfrp(spec_fix, x_fix, r; step_C=0.25) .-
            CFRPSolidState.predict_cfrp(spec_free, x_free, r; step_C=0.25)
        )) for r in runs) < 1e-13

        pv = CFRPSolidState.verify_cfrp_petab(spec_fix, x_fix, runs; rtol=1e-7)
        @test pv.passed && pv.max_gap_pp < 0.05
    end

    @testset "4. Comparison CSVs, figures & omitted raw CFRP check" begin
        raw_csvs = filter(f -> endswith(f, ".csv"), readdir(joinpath(ROOT, "data", "cfrp_audited")))
        @test isempty(raw_csvs)
        @test isfile(joinpath(ROOT, "data", "cfrp_audited", "README.md"))
        @test isfile(joinpath(ROOT, "data", "biasi_t700m21", "experimental_5K.csv"))

        cfrp_sel = CSV.read(joinpath(ROOT, "results", "cfrp_models", "cfrp_model_selection_and_map.csv"), DataFrame)
        @test nrow(cfrp_sel) == 4
        @test !("pooled_with_resin" in names(cfrp_sel))
        @test all(v -> !occursin("Pooled", v), cfrp_sel.variant)

        biasi_lit = CSV.read(joinpath(ROOT, "results", "biasi_comparison", "biasi_vs_literature_and_cfrp.csv"), DataFrame)
        @test nrow(biasi_lit) >= 7
        @test all(s -> startswith(s, "Biasi") || startswith(s, "Free-order") || startswith(s, "CFRP"), biasi_lit.source)

        for fig in [
            joinpath(ROOT, "results", "cfrp_models", "figures", "Fig02_Synthetic_ground_truth_recovery.png"),
            joinpath(ROOT, "results", "cfrp_models", "figures", "Fig03_CFRP_solid_state_fits.png"),
            joinpath(ROOT, "results", "cfrp_models", "figures", "Fig05_CFRP_M2_predictive_uncertainty_bands.png"),
            joinpath(ROOT, "results", "biasi_comparison", "figures", "Fig06_Biasi_solid_state_fits_and_cv.png"),
            joinpath(ROOT, "results", "biasi_comparison", "figures", "Fig08_Biasi_predictive_uncertainty_bands.png"),
        ]
            @test isfile(fig)
        end
    end
end
