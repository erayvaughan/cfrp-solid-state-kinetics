# Production CFRP solid-state Arrhenius + Gaussian MAP prior runner (Air + Nitrogen).
# Edit SETTINGS below or pass CLI overrides, then run:
#   julia --project=. -O1 run_cfrp_solid_state.jl --check-inputs
#   julia --project=. -O1 run_cfrp_solid_state.jl

import Pkg
Pkg.activate(@__DIR__)

SETTINGS = (
    audit_folder = joinpath(@__DIR__, "data", "cfrp_audited"),
    model = :M2_MAP, # :M2_MAP, :M2b_MAP, :M2_Unconstrained, :M3_RPM
    mu_E_char = 1.60, # 160 kJ/mol Gaussian prior mean on char oxidation E (units of 100 kJ/mol)
    sigma_E_char = 0.35, # 35 kJ/mol Gaussian prior std on char oxidation E (units of 100 kJ/mol)
    prior_weight = nothing, # auto-calibrated from AR(1) pilot data_loss / n_eff when nothing for *_MAP
    verify_petab = true,
    run_cv = true,
    run_uq = true,
    uq_boot = 30,
    seed = 20261122,
    output_folder = nothing,
)

using CSV, DataFrames, Statistics, Printf, TOML, SHA

const MODEL_ALIASES = Dict{Symbol,NamedTuple{(:kind, :k_params, :use_prior),Tuple{Symbol,Int,Bool}}}(
    :M2_MAP => (kind=:M2_1Mat_GeomOx, k_params=9, use_prior=true),
    :M2b_MAP => (kind=:M2b_2Mat1st_GeomOx, k_params=12, use_prior=true),
    :M2_Unconstrained => (kind=:M2_1Mat_GeomOx, k_params=9, use_prior=false),
    :M3_RPM => (kind=:M3_1Mat_RPMOx, k_params=11, use_prior=false),
)

function parse_cli_options(args, defaults)
    check_only = false
    model = defaults.model
    verify_petab = defaults.verify_petab
    run_cv = defaults.run_cv
    run_uq = defaults.run_uq
    uq_boot = defaults.uq_boot
    prior_weight = defaults.prior_weight
    output_folder = defaults.output_folder
    audit_folder = defaults.audit_folder

    for a in args
        if a == "--check-inputs"
            check_only = true
        elseif a == "--cv"
            run_cv = true
        elseif a == "--no-cv"
            run_cv = false
        elseif a == "--uq"
            run_uq = true
        elseif a == "--no-uq"
            run_uq = false
        elseif a == "--petab"
            verify_petab = true
        elseif a == "--no-petab"
            verify_petab = false
        elseif startswith(a, "--model=")
            m = Symbol(split(a, '='; limit=2)[2])
            haskey(MODEL_ALIASES, m) || error("Unknown --model=$m. Choose from M2_MAP, M2b_MAP, M2_Unconstrained, M3_RPM")
            model = m
        elseif startswith(a, "--uq-boot=")
            b = tryparse(Int, split(a, '='; limit=2)[2])
            (b === nothing || b < 4) && error("--uq-boot requires an integer >= 4")
            uq_boot = b
        elseif startswith(a, "--prior-weight=")
            w = tryparse(Float64, split(a, '='; limit=2)[2])
            (w === nothing || !isfinite(w) || w < 0) && error("--prior-weight requires a finite nonnegative number")
            prior_weight = w
        elseif startswith(a, "--audit-folder=")
            audit_folder = abspath(split(a, '='; limit=2)[2])
        elseif startswith(a, "--output=")
            output_folder = abspath(split(a, '='; limit=2)[2])
        else
            error("Unknown flag: $a. Supported: --check-inputs, --model=M2_MAP|M2b_MAP|M2_Unconstrained|M3_RPM, --cv/--no-cv, --uq/--no-uq, --uq-boot=N, --petab/--no-petab, --prior-weight=W, --audit-folder=PATH, --output=PATH")
        end
    end

    haskey(MODEL_ALIASES, model) || error("Unknown SETTINGS.model=$model")
    isdir(audit_folder) || error("Audit folder not found: $(audit_folder)")
    isfile(joinpath(audit_folder, "measurement_quality.csv")) ||
        error("measurement_quality.csv missing in $(audit_folder)")
    isfinite(defaults.mu_E_char) && defaults.mu_E_char > 0 || error("mu_E_char must be positive")
    isfinite(defaults.sigma_E_char) && defaults.sigma_E_char > 0 || error("sigma_E_char must be positive")
    (prior_weight === nothing || (isfinite(prior_weight) && prior_weight >= 0)) || error("prior_weight must be nonnegative or nothing")
    uq_boot >= 4 || error("uq_boot must be at least 4")

    (; check_only, model, spec_info=MODEL_ALIASES[model], verify_petab, run_cv, run_uq,
       uq_boot, prior_weight, output_folder,
       audit_folder=audit_folder,
       mu_E_char=Float64(defaults.mu_E_char),
       sigma_E_char=Float64(defaults.sigma_E_char),
       seed=Int(defaults.seed))
end

function check_audit_runs_light(audit_folder)
    metadata = CSV.read(joinpath(audit_folder, "measurement_quality.csv"), DataFrame)
    summary_runs = NamedTuple[]
    for row in eachrow(metadata)
        path = joinpath(audit_folder, row.condition * "_observations.csv")
        isfile(path) || error("Missing audited observations file: $path")
        obs = CSV.read(path, DataFrame)
        all(obs.condition .== row.condition) && all(obs.gas .== row.gas) && all(obs.rate .== row.rate_C_min) ||
            error("Audit labels differ for $(row.condition)")
        all(isfinite, obs.temperature_K) && minimum(obs.temperature_K) > 0 && all(isfinite, obs.mass_fraction) ||
            error("Invalid observations in $(row.condition)")
        all(diff(Float64.(obs.time_min)) .> 0) || error("Nonchronological audit in $(row.condition)")
        push!(summary_runs, (id=String(row.condition), gas=String(row.gas), rate=Float64(row.rate_C_min)))
    end
    length(summary_runs) == 10 && Set(r.rate for r in summary_runs) == Set([2.5, 5.0, 10.0, 20.0]) ||
        error("Expected the ten audited CFRP-only curves")
    summary_runs
end

opts = parse_cli_options(ARGS, SETTINGS)
light_runs = check_audit_runs_light(opts.audit_folder)

println("CFRP Solid-State Production Runner")
println("  Audit folder: ", opts.audit_folder)
println("  Runs loaded:  ", length(light_runs), " (rates: ", sort(unique(r.rate for r in light_runs)), " °C/min; gases: ", sort(unique(r.gas for r in light_runs)), ")")
println("  Model preset: ", opts.model, " -> ", opts.spec_info.kind, " (k=", opts.spec_info.k_params, ", prior=", opts.spec_info.use_prior, ")")
if opts.spec_info.use_prior
    pw_desc = opts.prior_weight === nothing ? "auto-calibrated from AR(1) pilot (data_loss / n_eff)" : @sprintf("%.5e (explicit)", opts.prior_weight)
    println("  Char prior:   E_char ~ N(", round(100opts.mu_E_char; digits=1), ", ", round(100opts.sigma_E_char; digits=1), "^2) kJ/mol, prior_weight=", pw_desc)
end
println("  Options:      verify_petab=", opts.verify_petab, ", run_cv=", opts.run_cv, ", run_uq=", opts.run_uq, " (uq_boot=", opts.uq_boot, ")")

if opts.check_only
    println("Input check passed. No optimization or output files written.")
    exit(0)
end

using LinearAlgebra, ForwardDiff
import Fides

include(joinpath(@__DIR__, "src", "solid_state", "CFRPSolidState.jl"))
include(joinpath(@__DIR__, "src", "uncertainty", "KineticUQ.jl"))
include(joinpath(@__DIR__, "src", "material_kinetics", "uq_reporting.jl"))
using .KineticUQ, .MaterialUQReporting

ENV["GKSwstype"] = "100"
using Plots

runs = CFRPSolidState.load_cfrp_runs(opts.audit_folder; verify_sources=false)
spec = CFRPSolidState.cfrp_model_spec(opts.spec_info.kind)

function make_output_dir(explicit_path)
    if explicit_path !== nothing
        ispath(explicit_path) && error("Output folder already exists: $explicit_path")
        mkpath(explicit_path)
        return explicit_path
    end
    CFRPSolidState.resultdir("cfrp_solid_state_")
end

function cfrp_fit_local(spec, runs, synth_masses, x_start; mu_E, sigma_E, prior_weight, step_C=0.5, seconds=10.0, maxiter=300)
    synth_runs = [merge(r, (mass=synth_masses[i],)) for (i, r) in enumerate(runs)]
    lf = z -> CFRPSolidState.loss_cfrp_map(spec, z, synth_runs; step_C=step_C, mu_E=mu_E, sigma_E=sigma_E, prior_weight=prior_weight)
    cfg = ForwardDiff.GradientConfig(lf, x_start)
    res = ForwardDiff.DiffResults.GradientResult(copy(x_start))
    obj = z -> begin
        ForwardDiff.gradient!(res, lf, z, cfg)
        (ForwardDiff.DiffResults.value(res), copy(ForwardDiff.DiffResults.gradient(res)))
    end
    x0 = clamp.(x_start, spec.lower .+ 1e-8 .* (spec.upper .- spec.lower),
                        spec.upper .- 1e-8 .* (spec.upper .- spec.lower))
    raw = Fides.solve(Fides.FidesProblem(obj, x0; lb=spec.lower, ub=spec.upper), Fides.BFGS();
        options=Fides.FidesOptions(maxiter=maxiter, maxtime=Float64(seconds), fatol=1e-12, frtol=1e-9, gatol=1e-7, verbose="error"))
    xc = clamp.(collect(Float64, raw.xmin), spec.lower, spec.upper)
    v, g = obj(xc)
    pg = maximum(abs.(xc .- clamp.(xc .- g, spec.lower, spec.upper)))
    (x=xc, stationary=(isfinite(v) && pg < 1e-4), pg=pg)
end

out = make_output_dir(opts.output_folder)
println("RESULT_DIRECTORY=", out); flush(stdout)

effective_prior_weight = 0.0
pilot_thr = nothing
if opts.spec_info.use_prior
    if opts.prior_weight === nothing
        println("[Step 1/4] Calibrating AR(1) prior weight via unconstrained pilot..."); flush(stdout)
        pilot = CFRPSolidState.fit_cfrp_cold(spec, runs; seed=opts.seed, ga_pop=20, ga_evals=240, lhs_starts=2, seconds_per_start=8.0)
        pilot_preds = [CFRPSolidState.predict_cfrp(spec, pilot.x, r; step_C=0.25) for r in runs]
        pilot_resids = [r.mass .- p for (r, p) in zip(runs, pilot_preds)]
        pilot_thr = KineticUQ.effective_uq_thresholds(pilot_resids, length(spec.names); level=0.95)
        effective_prior_weight = pilot.data_loss / max(pilot_thr.n_eff, 1.0)
        println(@sprintf("  Calibrated prior_weight = %.5e (pilot data_loss=%.5e, n_eff=%.2f, rmse=%.3f pp)",
            effective_prior_weight, pilot.data_loss, pilot_thr.n_eff, pilot.rmse_pp)); flush(stdout)
    else
        effective_prior_weight = Float64(opts.prior_weight)
    end
end

println("[Step 2/4] Fitting ", opts.model, " (", spec.kind, ") across ", length(runs), " CFRP runs..."); flush(stdout)
fit_res = if opts.spec_info.use_prior
    CFRPSolidState.fit_cfrp_cold_map(spec, runs; mu_E=opts.mu_E_char, sigma_E=opts.sigma_E_char,
        prior_weight=effective_prior_weight, seed=opts.seed + 10, ga_pop=24, ga_evals=360, lhs_starts=3, seconds_per_start=10.0)
else
    CFRPSolidState.fit_cfrp_cold(spec, runs; seed=opts.seed + 10, ga_pop=24, ga_evals=360, lhs_starts=3, seconds_per_start=10.0)
end

x_hat = fit_res.x
preds = [CFRPSolidState.predict_cfrp(spec, x_hat, r; step_C=0.25) for r in runs]
resids = [r.mass .- p for (r, p) in zip(runs, preds)]
thr = KineticUQ.effective_uq_thresholds(resids, length(x_hat); level=0.95)
near_bounds = [spec.names[j] for j in eachindex(x_hat)
               if min(x_hat[j] - spec.lower[j], spec.upper[j] - x_hat[j]) < 1e-4 * (spec.upper[j] - spec.lower[j])]

native_verif = CFRPSolidState.verify_native_cfrp(spec, x_hat, runs; step_C=0.25)
println(@sprintf("  Native verification passed: RMSE=%.3f pp, pg=%.2e, AD-FD gap=%.2e, refine=%.2e pp",
    fit_res.rmse_pp, fit_res.projected_gradient, native_verif.gradient_gap, native_verif.refinement_gap_pp)); flush(stdout)

petab_verif = nothing
if opts.verify_petab
    println("[Step 3/4] Running independent Catalyst/PEtab cross-check..."); flush(stdout)
    petab_verif = CFRPSolidState.verify_cfrp_petab(spec, x_hat, runs; rtol=1e-7)
    CSV.write(joinpath(out, "petab_check.csv"), DataFrame([petab_verif]))
    println(@sprintf("  PEtab verification passed (%s): max_gap=%.3e pp", petab_verif.solver, petab_verif.max_gap_pp)); flush(stdout)
else
    println("[Step 3/4] PEtab cross-check skipped (--no-petab)."); flush(stdout)
end

# Save kinetic parameters, curve predictions, per-condition scores
kt = CFRPSolidState.cfrp_kinetic_table(spec, x_hat)
kt[!, :preset] .= string(opts.model)
kt[!, :prior_weight] .= effective_prior_weight
CSV.write(joinpath(out, "kinetic_parameters.csv"), kt)

score_rows = NamedTuple[]
pred_rows = NamedTuple[]
for (i, r) in enumerate(runs)
    e = resids[i]
    push!(score_rows, (condition=r.id, gas=r.gas, rate_C_min=r.rate,
        rmse_pp=100.0 * sqrt(mean(abs2, e)), max_error_pp=100.0 * maximum(abs, e)))
    for j in eachindex(r.mass)
        push!(pred_rows, (condition=r.id, gas=r.gas, rate_C_min=r.rate,
            time_min=r.obstime[j] / 60.0, temperature_C=r.T[j] - 273.15,
            observed_percent=100.0 * r.mass[j], predicted_percent=100.0 * preds[i][j],
            residual_pp=100.0 * e[j]))
    end
end
scores_df = DataFrame(score_rows)
CSV.write(joinpath(out, "scores.csv"), scores_df)
CSV.write(joinpath(out, "predictions.csv"), DataFrame(pred_rows))

# Two-panel fit + residuals figure
cols = Dict(2.5 => :dodgerblue3, 5.0 => :darkorange2, 10.0 => :seagreen4, 20.0 => :purple)
p_fit = plot(xlabel="Temperature (°C)", ylabel="Remaining mass (%)",
    title="$(opts.model) ($(spec.kind)): Joint 10-Curve Fit ($(round(fit_res.rmse_pp; digits=3)) pp)",
    legend=:bottomleft, size=(1250, 720))
p_res = plot(xlabel="Temperature (°C)", ylabel="Observed - Predicted (pp)",
    title="$(opts.model): Residuals (AR1 n_eff=$(round(thr.n_eff; digits=1)))", legend=:topleft)
for (i, r) in enumerate(runs)
    c = get(cols, r.rate, :black)
    ls = r.gas == "Air" ? :solid : :dash
    Tc = r.T .- 273.15
    scatter!(p_fit, Tc, 100.0 .* r.mass; color=c, ms=1.4, msw=0, alpha=0.6,
        label=occursin("original", r.id) || r.gas == "Nitrogen" && r.rate == 2.5 ? "" : "$(r.gas) $(r.rate) °C/min")
    plot!(p_fit, Tc, 100.0 .* preds[i]; color=c, linestyle=ls, lw=2.0, label="")
    plot!(p_res, Tc, 100.0 .* resids[i]; color=c, linestyle=ls, lw=1.6,
        label=i <= 6 ? "$(r.id)" : "")
end
hline!(p_res, [0.0]; color=:gray, linestyle=:dot, label="")
fig_fit = plot(p_fit, p_res; layout=(1, 2), size=(1450, 620), margin=6Plots.mm)
savefig(fig_fit, joinpath(out, "fit_and_residuals.png"))

# Optional excluded-rate CV (hold out 10 and 20 C/min both gases)
cv_df = nothing
if opts.run_cv
    println("[Step 4a] Running excluded-rate CV (10.0 and 20.0 °C/min held out)..."); flush(stdout)
    cv_scores = NamedTuple[]
    cv_preds_rows = NamedTuple[]
    for hold_rate in (10.0, 20.0)
        tr = filter(r -> r.rate != hold_rate, runs)
        va = filter(r -> r.rate == hold_rate, runs)
        ffit = if opts.spec_info.use_prior
            CFRPSolidState.fit_cfrp_cold_map(spec, tr; mu_E=opts.mu_E_char, sigma_E=opts.sigma_E_char,
                prior_weight=effective_prior_weight, seed=opts.seed + Int(hold_rate),
                ga_pop=18, ga_evals=216, lhs_starts=2, seconds_per_start=8.0)
        else
            CFRPSolidState.fit_cfrp_cold(spec, tr; seed=opts.seed + Int(hold_rate),
                ga_pop=18, ga_evals=216, lhs_starts=2, seconds_per_start=8.0)
        end
        for vr in va
            vp = CFRPSolidState.predict_cfrp(spec, ffit.x, vr; step_C=0.25)
            ve = vr.mass .- vp
            push!(cv_scores, (excluded_rate=hold_rate, condition=vr.id, gas=vr.gas,
                rmse_pp=100.0 * sqrt(mean(abs2, ve)), stationary=ffit.stationary))
            for j in eachindex(vr.mass)
                push!(cv_preds_rows, (excluded_rate=hold_rate, condition=vr.id, gas=vr.gas,
                    temperature_C=vr.T[j] - 273.15, observed_percent=100.0 * vr.mass[j],
                    predicted_percent=100.0 * vp[j]))
            end
        end
    end
    cv_df = DataFrame(cv_scores)
    CSV.write(joinpath(out, "cv_scores.csv"), cv_df)
    CSV.write(joinpath(out, "cv_predictions.csv"), DataFrame(cv_preds_rows))
    println(@sprintf("  Excluded-rate CV RMSE: %.3f pp (all stationary=%s)",
        sqrt(mean(abs2, cv_df.rmse_pp)), string(all(cv_df.stationary)))); flush(stdout)
end

# Optional KineticUQ
uq_res = nothing
if opts.run_uq
    println("[Step 4b] Running AR(1) Profile Likelihoods & Bootstrap UQ (n_boot=", opts.uq_boot, ")..."); flush(stdout)
    uq_dir = joinpath(out, "uq"); mkpath(uq_dir)
    loss_uq = z -> CFRPSolidState.loss_cfrp_map(spec, z, runs; step_C=0.5, mu_E=opts.mu_E_char, sigma_E=opts.sigma_E_char, prior_weight=effective_prior_weight)
    w_loss_uq = (z, w) -> CFRPSolidState.loss_cfrp_map_weighted(spec, z, runs, w; step_C=0.5, mu_E=opts.mu_E_char, sigma_E=opts.sigma_E_char, prior_weight=effective_prior_weight)
    pred_fn = z -> [CFRPSolidState.predict_cfrp(spec, z, r; step_C=0.25) for r in runs]
    derived_fn = z -> CFRPSolidState.cfrp_derived_triplets(spec, z)
    derived_prof = [(name="log10_A$(j)_s_inv", logk_idx=spec.logk_idx[j], E_idx=spec.E_idx[j], ref_K=spec.refs[j], rate_unit=:s_inv)
                    for j in eachindex(spec.refs)]

    prof_ar1 = KineticUQ.profile_likelihood_uq(loss_uq, x_hat, spec.lower, spec.upper, spec.names, thr.loss_ratio_f;
        derived_profiles=derived_prof, max_steps=12, seconds_per_step=3.5, maxiter=180)
    prof_naive = KineticUQ.profile_likelihood_uq(loss_uq, x_hat, spec.lower, spec.upper, spec.names, thr.loss_ratio_naive;
        derived_profiles=derived_prof, max_steps=8, seconds_per_step=2.5, maxiter=140)

    bb = KineticUQ.bayesian_curve_bootstrap(w_loss_uq, pred_fn, derived_fn, x_hat, spec.lower, spec.upper, spec.names, length(runs);
        n_boot=opts.uq_boot, seed=opts.seed + 101, seconds_per_rep=6.0, maxiter=250)
    mb = KineticUQ.moving_block_bootstrap(
        (m, x0) -> cfrp_fit_local(spec, runs, m, x0; mu_E=opts.mu_E_char, sigma_E=opts.sigma_E_char, prior_weight=effective_prior_weight),
        [r.mass for r in runs], preds, pred_fn, derived_fn, x_hat, spec.names;
        n_boot=opts.uq_boot, seed=opts.seed + 201)

    bb_ok = size(bb.replicates, 2)
    mb_ok = size(mb.replicates, 2)

    sum_df = MaterialUQReporting._combine_summary("CFRP", string(opts.model), prof_ar1.summary, prof_naive.summary, bb.summary, mb.summary)
    CSV.write(joinpath(uq_dir, "uq_summary.csv"), sum_df)

    d1 = copy(bb.compensation); d1[!, :preset] .= string(opts.model)
    d2 = copy(mb.compensation); d2[!, :preset] .= string(opts.model)
    comp_df = reduce(vcat, [d1, d2]; cols=:union)
    CSV.write(joinpath(uq_dir, "compensation_correlations.csv"), comp_df)

    b1 = MaterialUQReporting._band_summary("CFRP", string(opts.model), "bayesian_curve_bootstrap", runs, bb.bands)
    b2 = MaterialUQReporting._band_summary("CFRP", string(opts.model), "moving_block_bootstrap", runs, mb.bands)
    bands_df = reduce(vcat, [b1, b2]; cols=:union)
    CSV.write(joinpath(uq_dir, "predictive_bands_summary.csv"), bands_df)

    guardrails = MaterialUQReporting.evaluate_guardrails([r.rate for r in runs], thr, prof_ar1, bb, mb;
        k_params=length(x_hat), near_bounds=near_bounds)
    open(joinpath(uq_dir, "uq_guardrails.toml"), "w") do io
        TOML.print(io, Dict{String,Any}(
            "model" => string(opts.model),
            "status" => guardrails.status,
            "n_rates" => guardrails.n_rates,
            "n_runs" => guardrails.n_runs,
            "k_params" => guardrails.k_params,
            "n_eff" => guardrails.n_eff,
            "rho1" => guardrails.rho1,
            "ci_inflation_vs_naive" => guardrails.ci_inflation,
            "identifiable_count" => guardrails.identifiable_count,
            "total_profiled" => guardrails.total_params,
            "near_bounds" => guardrails.near_bounds,
            "non_identifiable" => guardrails.non_identifiable,
            "max_comp_corr" => isfinite(guardrails.max_comp_corr) ? guardrails.max_comp_corr : "unavailable",
            "bb_converged" => bb_ok,
            "mb_converged" => mb_ok,
            "n_boot" => opts.uq_boot,
            "alerts" => [Dict("severity" => a.severity, "code" => a.code, "message" => a.message) for a in guardrails.alerts],
        ); sorted=true)
    end

    cfrp_runs_plot = [merge(r, (observation_temperature=r.T,)) for r in runs]
    MaterialUQReporting._plot_profiles(prof_ar1, thr, spec, "CFRP / $(opts.model)", joinpath(uq_dir, "uq_profiles.png"))
    MaterialUQReporting._plot_predictive_bands(cfrp_runs_plot, preds, bb, mb, "CFRP / $(opts.model)", joinpath(uq_dir, "uq_predictive_bands.png"))

    uq_res = (mod_id=string(opts.model), thr=thr, summary_df=sum_df, comp_df=comp_df, bands_df=bands_df,
              guardrails=guardrails, bb_ok=bb_ok, mb_ok=mb_ok, n_boot=opts.uq_boot)
    println(@sprintf("  KineticUQ done: identifiable=%d/%d, BB=%d/%d, MB=%d/%d, status=%s",
        guardrails.identifiable_count, guardrails.total_params, bb_ok, opts.uq_boot, mb_ok, opts.uq_boot, guardrails.status)); flush(stdout)
end

upk = CFRPSolidState.unpack(spec, x_hat)
run_meta = Dict{String,Any}(
    "preset" => string(opts.model),
    "spec_kind" => string(spec.kind),
    "k_params" => length(x_hat),
    "names" => spec.names,
    "x" => x_hat,
    "rmse_pp" => fit_res.rmse_pp,
    "data_loss" => fit_res.data_loss,
    "projected_gradient" => fit_res.projected_gradient,
    "stationary" => fit_res.stationary,
    "near_bounds" => near_bounds,
    "fiber_fraction" => upk.fiber,
    "char_yields" => collect(upk.yields),
    "prior_active" => opts.spec_info.use_prior,
    "prior_weight" => effective_prior_weight,
    "mu_E_char_kJ_mol" => 100.0 * opts.mu_E_char,
    "sigma_E_char_kJ_mol" => 100.0 * opts.sigma_E_char,
    "n_obs" => thr.n_obs,
    "n_eff" => thr.n_eff,
    "nu_eff" => thr.nu_eff,
    "median_lag1_autocorr" => thr.median_lag1_autocorr,
    "petab_verified" => opts.verify_petab,
    "petab_max_gap_pp" => petab_verif === nothing ? "skipped" : petab_verif.max_gap_pp,
    "audit_folder" => opts.audit_folder,
)
open(joinpath(out, "run.toml"), "w") do io
    TOML.print(io, run_meta; sorted=true)
end

report_path = joinpath(out, "report.md")
open(report_path, "w") do io
    println(io, "# CFRP Solid-State Arrhenius / MAP Report (`$(opts.model)` -> `$(spec.kind)`)\n")
    println(io, @sprintf("- **Dataset:** %d audited runs (%s; rates %s °C/min) from `%s`\n- **Joint fit RMSE:** `%.3f pp` (`k = %d`, stationary = `%s`, projected gradient = `%.2e`, near bounds = `%s`)",
        length(runs), join(sort(unique(r.gas for r in runs)), " + "), join(sort(unique(r.rate for r in runs)), ", "), opts.audit_folder,
        fit_res.rmse_pp, length(x_hat), string(fit_res.stationary), fit_res.projected_gradient, isempty(near_bounds) ? "none" : join(near_bounds, ", ")))
    if opts.spec_info.use_prior
        println(io, @sprintf("- **Char Oxidation Gaussian MAP Prior:** `E_char ~ N(%.1f, %.1f^2) kJ/mol`, `prior_weight = %.5e`",
            100.0 * opts.mu_E_char, 100.0 * opts.sigma_E_char, effective_prior_weight))
    else
        println(io, "- **Prior:** Unconstrained (no char-oxidation MAP prior applied)")
    end
    println(io, @sprintf("- **AR(1) Residual Structure:** `n_obs = %d`, `n_eff = %.2f`, `nu_eff = %.2f`, median lag-1 autocorrelation `rho1 = %.4f`",
        thr.n_obs, thr.n_eff, thr.nu_eff, thr.median_lag1_autocorr))
    if petab_verif !== nothing
        println(io, @sprintf("- **PEtab Cross-Check:** Passed (`%s`, max curve gap = `%.3e pp`)", petab_verif.solver, petab_verif.max_gap_pp))
    end
    println(io, "\n## Estimated Kinetic Parameters\n\n| Reaction | Law | A (s^-1) | E (kJ/mol) | Shape | Value | Fixed | Ref (K) | Char Yield |")
    println(io, "|---|---|---:|---:|---|---:|---|---:|---:|")
    for row in eachrow(kt)
        println(io, @sprintf("| %s | %s | %.4e | %.2f | %s | %.4f | %s | %.0f | %.4f |",
            row.reaction, row.law, row.A_s_inv, row.E_kJ_mol, row.shape_param, row.shape_value,
            string(row.shape_fixed), row.ref_K, row.char_yield))
    end
    println(io, @sprintf("\nEstimated carbon-fiber mass fraction: **`%.4f`** (`%.2f%%`).\n", upk.fiber, 100.0 * upk.fiber))
    println(io, "## Condition Scores\n\n| Condition | Gas | Rate (°C/min) | RMSE (pp) | Max Error (pp) |")
    println(io, "|---|---|---:|---:|---:|")
    for row in eachrow(scores_df)
        println(io, @sprintf("| %s | %s | %.1f | %.3f | %.3f |", row.condition, row.gas, row.rate_C_min, row.rmse_pp, row.max_error_pp))
    end
    println(io, "\n![Joint fit and residuals](fit_and_residuals.png)\n")
    if cv_df !== nothing
        println(io, "## Excluded-Rate Cross-Validation\n\n| Excluded Rate | Condition | Gas | RMSE (pp) | Stationary |")
        println(io, "|---:|---|---|---:|---|")
        for row in eachrow(cv_df)
            println(io, @sprintf("| %.1f | %s | %s | %.3f | %s |", row.excluded_rate, row.condition, row.gas, row.rmse_pp, string(row.stationary)))
        end
        println(io)
    end
end

if uq_res !== nothing
    MaterialUQReporting.append_uq_markdown(report_path, uq_res)
end

println("Finished. RESULT_DIRECTORY=", out)
