# External Validation of Single-Method Solid-State + MAP + KineticUQ
# Framework on Independent Published Biasi (2014) T700/M21 Experimental Data.

import Pkg
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(ROOT)

using CSV, DataFrames, Statistics, LinearAlgebra, ForwardDiff, Printf, TOML, Serialization, Plots
import Fides
using StableRNGs: StableRNG
ENV["GKSwstype"] = "100"

include(joinpath(ROOT, "src", "solid_state", "CFRPSolidState.jl"))
include(joinpath(ROOT, "src", "uncertainty", "KineticUQ.jl"))
include(joinpath(ROOT, "src", "material_kinetics", "uq_reporting.jl"))
using .KineticUQ

const OUT = joinpath(ROOT, "results", "biasi_comparison")
const BACKEND = joinpath(ROOT, "results", "biasi_comparison", "_checkpoints")
mkpath(OUT)
mkpath(joinpath(OUT, "figures"))
mkpath(BACKEND)

const MU_E_CHAR = 1.60
const SIGMA_E_CHAR = 0.35
const TRUE_BIASI_FIBER = 0.656

function _load_or_run(fn::Function, fname::String)
    ckpt = joinpath(BACKEND, fname)
    if isfile(ckpt)
        println("  [checkpoint] Loading cached ", fname); flush(stdout)
        return deserialize(ckpt)
    end
    res = fn()
    serialize(ckpt, res)
    res
end
_load_or_run(fname::String, fn::Function) = _load_or_run(fn, fname)

function combine_uq_table(dataset::String, model::String, prof_ar1::DataFrame, prof_naive::DataFrame,
                          bb::DataFrame, mb::DataFrame)
    rows = NamedTuple[]
    for r in eachrow(prof_ar1)
        pname = r.parameter
        pn = first(eachrow(filter(q -> q.parameter == pname, prof_naive)))
        brows = filter(q -> q.parameter == pname, bb)
        mrows = filter(q -> q.parameter == pname, mb)
        br = isempty(brows) ? (ci_low=NaN, ci_high=NaN, ci_width=NaN, boot_median=NaN) : first(eachrow(brows))
        mr = isempty(mrows) ? (ci_low=NaN, ci_high=NaN, ci_width=NaN, boot_median=NaN) : first(eachrow(mrows))
        push!(rows, (
            dataset=dataset, model=model, parameter=pname,
            estimate=r.estimate,
            ar1_prof_low=r.ci_low, ar1_prof_high=r.ci_high, ar1_prof_width=r.ci_width,
            identifiability=r.identifiability, bound_active=r.bound_active,
            naive_prof_low=pn.ci_low, naive_prof_high=pn.ci_high, naive_prof_width=pn.ci_width,
            ar1_to_naive_ratio=r.ci_width / max(pn.ci_width, 1e-12),
            block_boot_median=mr.boot_median, block_boot_low=mr.ci_low, block_boot_high=mr.ci_high, block_boot_width=mr.ci_width,
            bayes_boot_median=br.boot_median, bayes_boot_low=br.ci_low, bayes_boot_high=br.ci_high, bayes_boot_width=br.ci_width,
        ))
    end
    DataFrame(rows)
end

function band_summary(dataset::String, method::String, runs, bands_df::DataFrame)
    rows = NamedTuple[]
    for (i, r) in enumerate(runs)
        sub = filter(q -> q.run_idx == i, bands_df)
        isempty(sub) && continue
        push!(rows, (
            dataset=dataset, method=method, run=r.id,
            rate_K_min=r.rate, gas=r.gas,
            mean_width_pp=mean(sub.width_pp), max_width_pp=maximum(sub.width_pp),
        ))
    end
    DataFrame(rows)
end

function cfrp_fit_local_map(spec, runs, synth_masses, x_start; step_C=0.25, mu_E=MU_E_CHAR, sigma_E=SIGMA_E_CHAR, prior_weight=0.0, seconds=8.0, maxiter=350)
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

function run_biasi_cv(variant::String, spec, runs; prior_weight=0.0, seed=20261123)
    fold_rows = NamedTuple[]
    fold_preds = Dict{Float64,Vector{Float64}}()
    for (fold_idx, va_run) in enumerate(runs)
        hold_rate = va_run.rate
        tr_runs = filter(r -> r.rate != hold_rate, runs)
        ffit = if prior_weight > 0.0
            CFRPSolidState.fit_cfrp_cold_map(spec, tr_runs; mu_E=MU_E_CHAR, sigma_E=SIGMA_E_CHAR, prior_weight=prior_weight,
                seed=seed + 100 * fold_idx + Int(round(hold_rate)), ga_pop=18, ga_evals=216, lhs_starts=2, seconds_per_start=8.0)
        else
            CFRPSolidState.fit_cfrp_cold(spec, tr_runs;
                seed=seed + 100 * fold_idx + Int(round(hold_rate)), ga_pop=18, ga_evals=216, lhs_starts=2, seconds_per_start=8.0)
        end
        vp = CFRPSolidState.predict_cfrp(spec, ffit.x, va_run; step_C=0.25)
        ve = vp .- va_run.mass
        rmse_pp = 100.0 * sqrt(mean(abs2, ve))
        fold_preds[hold_rate] = vp
        push!(fold_rows, (
            variant=variant, base_model=string(spec.kind), fixed_fiber=(spec.fiber_idx == 0),
            excluded_rate_K_min=hold_rate, condition=va_run.id,
            cv_rmse_pp=rmse_pp, max_abs_err_pp=100.0 * maximum(abs, ve),
            train_rmse_pp=ffit.rmse_pp, stationary=ffit.stationary,
        ))
        println(@sprintf("    [%s CV] hold=%.0f K/min -> CV RMSE=%.3f pp (train=%.3f, stationary=%s)",
            variant, hold_rate, rmse_pp, ffit.rmse_pp, string(ffit.stationary))); flush(stdout)
    end
    df = DataFrame(fold_rows)
    (; df, fold_preds,
       cv_rms_pp=sqrt(mean(abs2, df.cv_rmse_pp)),
       cv_mean_pp=mean(df.cv_rmse_pp),
       all_stationary=all(df.stationary))
end

function fit_and_summarize_variant(variant::String, kind::Symbol, runs; fixed_fiber=TRUE_BIASI_FIBER, prior_weight=0.0, seed=20261123)
    spec = CFRPSolidState.cfrp_model_spec(kind; fixed_fiber=fixed_fiber)
    println("  Fitting ", variant, " (", spec.kind, ", k=", length(spec.names), ", fixed_fiber=", fixed_fiber, ", prior_weight=", @sprintf("%.3e", prior_weight), ")..."); flush(stdout)
    fit_obj = if prior_weight > 0.0
        CFRPSolidState.fit_cfrp_cold_map(spec, runs; mu_E=MU_E_CHAR, sigma_E=SIGMA_E_CHAR, prior_weight=prior_weight,
            seed=seed, ga_pop=24, ga_evals=360, lhs_starts=3, seconds_per_start=10.0)
    else
        CFRPSolidState.fit_cfrp_cold(spec, runs;
            seed=seed, ga_pop=24, ga_evals=360, lhs_starts=3, seconds_per_start=10.0)
    end
    cv_obj = run_biasi_cv(variant, spec, runs; prior_weight=prior_weight, seed=seed + 1000)
    x_hat = fit_obj.x
    preds = [CFRPSolidState.predict_cfrp(spec, x_hat, r; step_C=0.25) for r in runs]
    resids = [r.mass .- p for (r, p) in zip(runs, preds)]
    ic = CFRPSolidState.information_criteria(resids, length(x_hat))
    thr = KineticUQ.effective_uq_thresholds(resids, length(x_hat); level=0.95)
    nv = CFRPSolidState.verify_native_cfrp(spec, x_hat, runs; step_C=0.25)
    pv = CFRPSolidState.verify_cfrp_petab(spec, x_hat, runs; rtol=1e-7)
    jac = CFRPSolidState.scaled_jacobian_diagnostics(
        z -> vcat([CFRPSolidState.predict_cfrp(spec, z, r; step_C=0.25) ./ sqrt(length(r.mass) * length(runs)) for r in runs]...),
        x_hat, spec.lower, spec.upper,
    )
    near = [spec.names[j] for j in eachindex(x_hat)
            if min(x_hat[j] - spec.lower[j], spec.upper[j] - x_hat[j]) < 1e-4 * (spec.upper[j] - spec.lower[j])]
    upk = CFRPSolidState.unpack(spec, x_hat)
    e_matrix = 100.0 * x_hat[spec.E_idx[1]]
    e_char = 100.0 * x_hat[spec.E_idx[spec.nresin + 1]]
    e_fiber = 100.0 * x_hat[spec.E_idx[spec.nresin + 2]]
    kt = CFRPSolidState.cfrp_kinetic_table(spec, x_hat)
    kt[!, :variant] .= variant
    kt[!, :fixed_fiber] .= (fixed_fiber !== nothing)

    row = (
        variant=variant, base_model=string(spec.kind), k_params=length(x_hat),
        fixed_fiber=(fixed_fiber !== nothing), prior_active=(prior_weight > 0.0), prior_weight=prior_weight,
        joint_rmse_pp=ic.rmse_pp, cv_rms_pp=cv_obj.cv_rms_pp, cv_mean_pp=cv_obj.cv_mean_pp,
        n_obs=ic.n_obs, n_eff=ic.n_eff, nu_eff=thr.nu_eff, median_rho1=ic.median_lag1_autocorr,
        aic_nominal=ic.aic_nominal, bic_nominal=ic.bic_nominal, aicc_eff=ic.aicc_eff, bic_eff=ic.bic_eff,
        fiber_fraction=upk.fiber, fiber_abs_err=abs(upk.fiber - TRUE_BIASI_FIBER), char_yield_eff=upk.yields[1],
        E_matrix_kJ_mol=e_matrix, E_char_kJ_mol=e_char, E_fiber_kJ_mol=e_fiber,
        stationary=fit_obj.stationary, cv_all_stationary=cv_obj.all_stationary,
        bound_hits=isempty(near) ? "none" : join(near, ", "),
        jac_cond=jac.condition_number, ad_fd_gap=nv.gradient_gap, petab_max_gap_pp=pv.max_gap_pp,
    )
    println(@sprintf("  [%s] RMSE=%.3f pp, CV_mean=%.3f pp (CV_rms=%.3f pp), AICc_eff=%.2f, E_ch=%.1f, E_fib=%.1f, f_F=%.4f, PEtab=%.2e pp",
        variant, row.joint_rmse_pp, row.cv_mean_pp, row.cv_rms_pp, row.aicc_eff,
        row.E_char_kJ_mol, row.E_fiber_kJ_mol, row.fiber_fraction, row.petab_max_gap_pp)); flush(stdout)

    (; variant, spec, fit_obj, cv_obj, x_hat, preds, resids, ic, thr, nv, pv, jac, near, upk, kt, row)
end

function run_biasi_uq(item, runs; prior_weight=0.0, seed=20262000, n_boot=40)
    spec = item.spec
    x_hat = item.x_hat
    preds = item.preds
    thr = item.thr
    label = item.variant
    @printf("  [UQ %s] n_obs=%d, n_eff=%.2f, nu_eff=%.2f, rho1=%.4f, AR1_ratio=%.4f\n",
        label, thr.n_obs, thr.n_eff, thr.nu_eff, thr.median_lag1_autocorr, thr.loss_ratio_f)
    flush(stdout)

    loss_fn = z -> CFRPSolidState.loss_cfrp_map(spec, z, runs; step_C=0.25, mu_E=MU_E_CHAR, sigma_E=SIGMA_E_CHAR, prior_weight=prior_weight)
    derived_prof = [(name="log10_A$(j)_s_inv", logk_idx=spec.logk_idx[j], E_idx=spec.E_idx[j], ref_K=spec.refs[j], rate_unit=:s_inv)
                    for j in eachindex(spec.refs)]

    prof_ar1 = KineticUQ.profile_likelihood_uq(loss_fn, x_hat, spec.lower, spec.upper, spec.names, thr.loss_ratio_f;
        derived_profiles=derived_prof, max_steps=14, seconds_per_step=3.5, maxiter=180)
    prof_naive = KineticUQ.profile_likelihood_uq(loss_fn, x_hat, spec.lower, spec.upper, spec.names, thr.loss_ratio_naive;
        derived_profiles=derived_prof, max_steps=8, seconds_per_step=2.0, maxiter=140)

    pred_fn = z -> [CFRPSolidState.predict_cfrp(spec, z, r; step_C=0.25) for r in runs]
    derived_fn = z -> CFRPSolidState.cfrp_derived_triplets(spec, z)
    w_loss = (z, w) -> CFRPSolidState.loss_cfrp_map_weighted(spec, z, runs, w; step_C=0.25, mu_E=MU_E_CHAR, sigma_E=SIGMA_E_CHAR, prior_weight=prior_weight)

    bb = KineticUQ.bayesian_curve_bootstrap(w_loss, pred_fn, derived_fn, x_hat, spec.lower, spec.upper, spec.names, length(runs);
        n_boot=n_boot, seed=seed + 1, seconds_per_rep=12.0, maxiter=500)
    mb = KineticUQ.moving_block_bootstrap(
        (m, x0) -> cfrp_fit_local_map(spec, runs, m, x0; step_C=0.25, mu_E=MU_E_CHAR, sigma_E=SIGMA_E_CHAR, prior_weight=prior_weight, seconds=12.0, maxiter=500),
        [r.mass for r in runs], preds, pred_fn, derived_fn, x_hat, spec.names;
        n_boot=n_boot, seed=seed + 2)

    bb_ok = size(bb.replicates, 2)
    mb_ok = size(mb.replicates, 2)
    guardrails = MaterialUQReporting.evaluate_guardrails([r.rate for r in runs], thr, prof_ar1, bb, mb;
        k_params=length(x_hat), near_bounds=item.near)

    println(@sprintf("  [UQ %s] BB=%d/%d, MB=%d/%d, identifiable=%d/%d, status=%s",
        label, bb_ok, n_boot, mb_ok, n_boot, guardrails.identifiable_count, guardrails.total_params, guardrails.status)); flush(stdout)

    tbl = combine_uq_table("Biasi_T700M21", label, prof_ar1.summary, prof_naive.summary, bb.summary, mb.summary)
    (; label, thr, prof_ar1, prof_naive, bb, mb, bb_ok, mb_ok, n_boot, guardrails, tbl)
end

function main()
    println("=== PHASE 6: EXTERNAL VALIDATION ON BIASI (2014) T700/M21 BENCHMARK ==="); flush(stdout)
    runs = CFRPSolidState.load_biasi_as_cfrp_runs()

    # Block 1: Fixed-composition model selection (fiber_fraction = 0.656)
    blk1 = _load_or_run("blk1_fixed_fiber.jls") do
        println("\n[Block 1] Fitting B1..B6 with fixed_fiber = 0.656 across 4 Air heating rates (2, 5, 10, 20 K/min)...")
        b1 = fit_and_summarize_variant("B1_M1_1stOrder", :M1_1Mat_1stOrder, runs; fixed_fiber=TRUE_BIASI_FIBER, prior_weight=0.0, seed=2026601)
        b2 = fit_and_summarize_variant("B2_M2_GeomOx", :M2_1Mat_GeomOx, runs; fixed_fiber=TRUE_BIASI_FIBER, prior_weight=0.0, seed=2026602)

        pw_biasi = b2.fit_obj.data_loss / max(b2.ic.n_eff, 1.0)
        @printf("  Calibrated Biasi AR(1) char prior_weight = %.5e (B2 data_loss=%.5e, n_eff=%.2f)\n",
            pw_biasi, b2.fit_obj.data_loss, b2.ic.n_eff); flush(stdout)

        b3 = fit_and_summarize_variant("B3_M2_MAP_CharPrior", :M2_1Mat_GeomOx, runs; fixed_fiber=TRUE_BIASI_FIBER, prior_weight=pw_biasi, seed=2026603)
        b4 = fit_and_summarize_variant("B4_M2b_2Mat1st_GeomOx", :M2b_2Mat1st_GeomOx, runs; fixed_fiber=TRUE_BIASI_FIBER, prior_weight=0.0, seed=2026604)
        b5 = fit_and_summarize_variant("B5_M3_RPMOx", :M3_1Mat_RPMOx, runs; fixed_fiber=TRUE_BIASI_FIBER, prior_weight=0.0, seed=2026605)
        b6 = fit_and_summarize_variant("B6_M5_FreeOrder_Cold", :M5_2Mat_FreeOrder_Cold, runs; fixed_fiber=TRUE_BIASI_FIBER, prior_weight=0.0, seed=2026606)
        (; pw_biasi, items=[b1, b2, b3, b4, b5, b6])
    end

    # Block 2: Blind fiber_fraction recovery (fixed_fiber = nothing, Air only)
    blk2 = _load_or_run("blk2_blind_fiber.jls") do
        println("\n[Block 2] Blind composition recovery (fiber_fraction free in [0.1, 0.9], true = 0.656)...")
        b2_blind = fit_and_summarize_variant("B2_BlindFiber", :M2_1Mat_GeomOx, runs; fixed_fiber=nothing, prior_weight=0.0, seed=2026607)
        b3_blind = fit_and_summarize_variant("B3_MAP_BlindFiber", :M2_1Mat_GeomOx, runs; fixed_fiber=nothing, prior_weight=blk1.pw_biasi, seed=2026608)
        (; items=[b2_blind, b3_blind])
    end

    all_items = vcat(blk1.items, blk2.items)
    model_sel_df = DataFrame([it.row for it in all_items])
    cv_df = reduce(vcat, [it.cv_obj.df for it in all_items]; cols=:union)
    triplets_df = reduce(vcat, [it.kt for it in all_items]; cols=:union)
    CSV.write(joinpath(OUT, "biasi_model_selection.csv"), model_sel_df)
    CSV.write(joinpath(OUT, "biasi_cv_by_rate.csv"), cv_df)
    CSV.write(joinpath(OUT, "biasi_triplets.csv"), triplets_df)

    # Block 3: Comparison against Biasi (2014) Table IV.2 & Reference CFRP M2_MAP_CharPrior
gulf_comp = CSV.read(joinpath(ROOT, "results", "cfrp_models", "cfrp_model_selection_and_map.csv"), DataFrame)
gulf_kt = CSV.read(joinpath(ROOT, "results", "cfrp_models", "cfrp_prior_triplets.csv"), DataFrame)
g_m2 = only(eachrow(filter(r -> r.variant == "M2_MAP_CharPrior", gulf_comp)))
g_kt_m2 = filter(r -> r.variant == "M2_MAP_CharPrior", gulf_kt)

lit_rows = NamedTuple[]
# Biasi (2014) Table IV.2 Air parameters
push!(lit_rows, (
    source="Biasi (2014) Table IV.2 Air", dataset="T700/M21 (4 Air)", k_params=missing,
    joint_rmse_pp=missing, cv_mean_pp=missing,
    E_matrix_kJ_mol=105.5, A_matrix_s_inv=9.03e5, n_matrix=1.8, char_yield_nu=0.45,
    E_char_kJ_mol=203.3, A_char_s_inv=5.62e10, n_char=1.8,
    E_fiber_kJ_mol=167.7, A_fiber_s_inv=2.0e5, n_fiber=0.5, fiber_fraction=0.656,
    notes="Published Table IV.2 air parameters (p.102); char normalization ambiguous",
))
# Historical Free-Order Power-Law Baseline Refits on experimental_5K.csv
for (src_name, kpar, jrmse, cvm, emx, amx, nmx, nuv, ech, ach, nch, efib, afib, nfib, note) in [
    ("Free-order 3-route refit", 10, 1.031, 1.267, 193.91, 9.766996726213537e12, 1.2300371806147228, 0.34085075077315723,
     154.64, 5.2179923721454846e7, 0.9613490007339006, 133.56, 2911.441628860731, 0.5015772851196069,
     "Reference 3-route power-law refit on experimental_5K.csv (free reaction orders)"),
    ("Free-order 4-route refit", 13, 0.820, 1.033, 433.51, 3.3075071284266194e34, 4.918268454032833, 0.3125731274627358,
     148.45, 2.2750374494359753e7, 1.0108176709391103, 133.73, 2979.1313638201297, 0.5020816117025369,
     "Reference 4-route early-loss refit (unconstrained high E_matrix; early_fraction bound active)"),
]
    push!(lit_rows, (
        source=src_name, dataset="T700/M21 (4 Air)", k_params=kpar,
        joint_rmse_pp=jrmse, cv_mean_pp=cvm,
        E_matrix_kJ_mol=emx, A_matrix_s_inv=amx, n_matrix=nmx, char_yield_nu=nuv,
        E_char_kJ_mol=ech, A_char_s_inv=ach, n_char=nch,
        E_fiber_kJ_mol=efib, A_fiber_s_inv=afib, n_fiber=nfib, fiber_fraction=0.656,
        notes=note,
    ))
end
# Biasi solid-state variants
    for it in [blk1.items[2], blk1.items[3], blk1.items[4], blk2.items[2]]
        k1 = it.kt[1, :]
        kc = it.kt[it.spec.nresin + 1, :]
        kf = it.kt[it.spec.nresin + 2, :]
        push!(lit_rows, (
            source="Biasi $(it.variant)", dataset="T700/M21 (4 Air)", k_params=it.row.k_params,
            joint_rmse_pp=it.row.joint_rmse_pp, cv_mean_pp=it.row.cv_mean_pp,
            E_matrix_kJ_mol=k1.E_kJ_mol, A_matrix_s_inv=k1.A_s_inv, n_matrix=k1.shape_value, char_yield_nu=it.upk.yields[1],
            E_char_kJ_mol=kc.E_kJ_mol, A_char_s_inv=kc.A_s_inv, n_char=kc.shape_value,
            E_fiber_kJ_mol=kf.E_kJ_mol, A_fiber_s_inv=kf.A_s_inv, n_fiber=kf.shape_value, fiber_fraction=it.upk.fiber,
            notes="Single-method GA+BFGS + solid-state geom Ox" * (it.row.prior_active ? " + AR(1) char prior" : "") * (it.row.fixed_fiber ? " (fixed f_F=0.656)" : " (blind f_F)"),
        ))
    end
    # CFRP (8552S/AS4) M2_MAP_CharPrior
    push!(lit_rows, (
        source="CFRP (8552S/AS4) M2_MAP_CharPrior", dataset="CFRP 8552S/AS4 (10 Air+N2)", k_params=g_m2.k_params,
        joint_rmse_pp=g_m2.cfrp_rmse_pp, cv_mean_pp=g_m2.cfrp_cv_rmse_pp,
        E_matrix_kJ_mol=g_kt_m2.E_kJ_mol[1], A_matrix_s_inv=g_kt_m2.A_s_inv[1], n_matrix=g_kt_m2.shape_value[1], char_yield_nu=g_m2.char_yield_eff,
        E_char_kJ_mol=g_kt_m2.E_kJ_mol[2], A_char_s_inv=g_kt_m2.A_s_inv[2], n_char=g_kt_m2.shape_value[2],
        E_fiber_kJ_mol=g_kt_m2.E_kJ_mol[3], A_fiber_s_inv=g_kt_m2.A_s_inv[3], n_fiber=g_kt_m2.shape_value[3], fiber_fraction=g_m2.fiber_fraction,
        notes="Standalone CFRP M2_MAP_CharPrior benchmark (2.5-20 C/min, Air+N2)",
    ))
    lit_df = DataFrame(lit_rows)
    CSV.write(joinpath(OUT, "biasi_vs_literature_and_cfrp.csv"), lit_df)

    # Block 4: Full KineticUQ + Guardrails
    blk4 = _load_or_run("blk4_biasi_uq.jls") do
        println("\n[Block 4] Running full KineticUQ (AR(1)/Naive profiles + 40 BB + 40 MB + Guardrails)...")
        uq_b2 = run_biasi_uq(blk1.items[2], runs; prior_weight=0.0, seed=2026701, n_boot=40)
        uq_b3 = run_biasi_uq(blk1.items[3], runs; prior_weight=blk1.pw_biasi, seed=2026702, n_boot=40)
        uq_b3_blind = run_biasi_uq(blk2.items[2], runs; prior_weight=blk1.pw_biasi, seed=2026703, n_boot=40)
        uq_b4 = run_biasi_uq(blk1.items[4], runs; prior_weight=0.0, seed=2026704, n_boot=40)
        (; uq_b2, uq_b3, uq_b3_blind, uq_b4)
    end

    CSV.write(joinpath(OUT, "biasi_b2_m2_uq_summary.csv"), blk4.uq_b2.tbl)
    CSV.write(joinpath(OUT, "biasi_b3_m2_map_uq_summary.csv"), blk4.uq_b3.tbl)
    CSV.write(joinpath(OUT, "biasi_b3_blind_fiber_uq_summary.csv"), blk4.uq_b3_blind.tbl)
    CSV.write(joinpath(OUT, "biasi_b4_m2b_uq_summary.csv"), blk4.uq_b4.tbl)

    comp_parts = DataFrame[]
    band_parts = DataFrame[]
    for uq_it in [blk4.uq_b2, blk4.uq_b3, blk4.uq_b3_blind, blk4.uq_b4]
        d1 = copy(uq_it.bb.compensation); d1[!, :variant] .= uq_it.label; push!(comp_parts, d1)
        d2 = copy(uq_it.mb.compensation); d2[!, :variant] .= uq_it.label; push!(comp_parts, d2)
        push!(band_parts, band_summary(uq_it.label, "bayesian_curve_bootstrap", runs, uq_it.bb.bands))
        push!(band_parts, band_summary(uq_it.label, "moving_block_bootstrap", runs, uq_it.mb.bands))
    end
    CSV.write(joinpath(OUT, "biasi_compensation_correlations.csv"), reduce(vcat, comp_parts; cols=:union))
    CSV.write(joinpath(OUT, "biasi_predictive_bands_summary.csv"), reduce(vcat, band_parts; cols=:union))

    # --- FIGURES ---
    default(background_color=:white, guidefontsize=10, tickfontsize=8, legendfontsize=8, titlefontsize=10, margin=5Plots.mm)

    b3 = blk1.items[3]
    b3_blind = blk2.items[2]
    b4 = blk1.items[4]
    rate_cols = [:dodgerblue3, :darkorange2, :seagreen4, :purple]

    # Figure 1: 2x2 Fits, CV Model Selection, Residuals, Activation Energies
    p_fit = plot(xlabel="Temperature (K)", ylabel="Remaining mass (%)",
        title="Biasi T700/M21: B3_M2_MAP_CharPrior (RMSE=$(round(b3.row.joint_rmse_pp; digits=3)) pp)",
        legend=:bottomleft, ylims=(-2, 104))
    p_res = plot(xlabel="Temperature (K)", ylabel="Obs - Pred (pp)",
        title="B3_M2_MAP Residuals (n_eff=$(round(b3.ic.n_eff; digits=1)))", legend=:topleft)
    for (i, r) in enumerate(runs)
        c = rate_cols[i]
        scatter!(p_fit, r.T, 100.0 .* r.mass; color=c, ms=1.6, msw=0, alpha=0.65, label="$(Int(r.rate)) K/min data")
        plot!(p_fit, r.T, 100.0 .* b3.preds[i]; color=c, lw=2.0, label="$(Int(r.rate)) K/min fit")
        plot!(p_res, r.T, 100.0 .* b3.resids[i]; color=c, lw=1.7, label="$(Int(r.rate)) K/min")
    end
    hline!(p_res, [0.0]; color=:gray, linestyle=:dot, label="")

    cv_labels = ["B1_M1", "B2_M2", "B3_MAP", "B4_M2b", "B5_M3", "B6_M5", "B3_Blind", "Ref_3rt", "Ref_4rt"]
    joint_vals = vcat(model_sel_df.joint_rmse_pp[[1,2,3,4,5,6,8]], [1.031, 0.820])
    cv_vals = vcat(model_sel_df.cv_mean_pp[[1,2,3,4,5,6,8]], [1.267, 1.033])
    xidx = 1:length(cv_labels)
    p_cv = bar(xidx .- 0.18, joint_vals; bar_width=0.35, color=:dodgerblue3, label="Joint RMSE",
        xticks=(xidx, cv_labels), xrotation=30, ylabel="RMSE (pp)",
        ylims=(0, max(maximum(cv_vals), 2.0) * 1.18), title="Model Selection: Joint vs Leave-One-Rate-Out CV")
    bar!(p_cv, xidx .+ 0.18, cv_vals; bar_width=0.35, color=:darkorange2, label="Mean CV RMSE")

    e_src = ["Biasi IV.2", "B2_M2", "B3_MAP", "B3_Blind", "CFRP 8552S"]
    b2 = blk1.items[2]
    em = [105.5, b2.row.E_matrix_kJ_mol, b3.row.E_matrix_kJ_mol, b3_blind.row.E_matrix_kJ_mol, g_m2.char_yield_eff > 0 ? g_kt_m2.E_kJ_mol[1] : NaN]
    ec = [203.3, b2.row.E_char_kJ_mol, b3.row.E_char_kJ_mol, b3_blind.row.E_char_kJ_mol, g_kt_m2.E_kJ_mol[2]]
    ef = [167.7, b2.row.E_fiber_kJ_mol, b3.row.E_fiber_kJ_mol, b3_blind.row.E_fiber_kJ_mol, g_kt_m2.E_kJ_mol[3]]
    exi = 1:length(e_src)
    p_e = bar(exi .- 0.25, em; bar_width=0.24, color=:steelblue, label="E_matrix",
        xticks=(exi, e_src), xrotation=20, ylabel="Activation energy (kJ/mol)",
        title="Activation Energies vs Biasi (2014) & CFRP (8552S/AS4)", ylims=(0, max(maximum(ec), 230) * 1.15))
    bar!(p_e, exi, ec; bar_width=0.24, color=:firebrick3, label="E_char")
    bar!(p_e, exi .+ 0.25, ef; bar_width=0.24, color=:seagreen4, label="E_fiber (n=0.5)")

    fig1 = plot(p_fit, p_cv, p_res, p_e; layout=(2, 2), size=(1450, 960), margin=6Plots.mm,
        plot_title="External Validation: Biasi (2014) T700/M21 Solid-State Models", plot_titlefontsize=12)
    savefig(fig1, joinpath(OUT, "figures", "Fig06_Biasi_solid_state_fits_and_cv.png"))

    # Figure 2: AR(1) vs Naive Profile Likelihoods (B3_M2_MAP_CharPrior + Blind fiber_fraction)
    uq_b3 = blk4.uq_b3
    uq_bl = blk4.uq_b3_blind
    prof_specs = [
        ("B3_MAP", "E1", "E1_matrix (×100 kJ/mol)", uq_b3, nothing),
        ("B3_MAP", "n1", "n1_matrix", uq_b3, nothing),
        ("B3_MAP", "E2", "E2_char (×100 kJ/mol)", uq_b3, 1.60),
        ("B3_MAP", "log10_A2_s_inv", "log10 A_char (s^-1)", uq_b3, nothing),
        ("B3_MAP", "E3", "E3_fiber (×100 kJ/mol)", uq_b3, nothing),
        ("B3_MAP", "log10_A3_s_inv", "log10 A_fiber (s^-1)", uq_b3, nothing),
        ("B3_MAP", "char_yield1", "char_yield1", uq_b3, 0.45),
        ("B3_Blind", "fiber_fraction", "Blind fiber_fraction (true=0.656)", uq_bl, TRUE_BIASI_FIBER),
    ]
    psubs = Any[]
    for (tag, pname, ttl, uqobj, refval) in prof_specs
        sub = filter(r -> r.parameter == pname, uqobj.prof_ar1.curves)
        srow = first(eachrow(filter(r -> r.parameter == pname, uqobj.prof_ar1.summary)))
        ycap = max(3.0 * uqobj.thr.loss_ratio_f, 3.0)
        plt = plot(sub.target, clamp.(sub.loss_ratio, 0.98, ycap); lw=2.0, color=:dodgerblue3, label="",
            title="$tag: $pname ($(srow.identifiability))", xlabel=ttl, ylabel="L / L_min",
            ylims=(0.98, min(maximum(clamp.(sub.loss_ratio, 0.98, ycap)) * 1.10 + 0.05, ycap)),
            titlefontsize=9, guidefontsize=8, tickfontsize=7)
        hline!(plt, [uqobj.thr.loss_ratio_f]; color=:firebrick3, linestyle=:dash, lw=1.4, label="")
        hline!(plt, [uqobj.thr.loss_ratio_naive]; color=:gray45, linestyle=:dot, lw=1.2, label="")
        vline!(plt, [srow.estimate]; color=:black, linestyle=:dashdot, lw=1.0, label="")
        refval !== nothing && vline!(plt, [refval]; color=:seagreen4, linestyle=:dash, lw=1.5, label="")
        push!(psubs, plt)
    end
    fig2 = plot(psubs...; layout=(2, 4), size=(1550, 760), margin=5Plots.mm,
        plot_title="Biasi T700/M21 Profile Likelihoods: AR(1) 95% (red) vs Naive (gray) & True f_F=0.656 (green)", plot_titlefontsize=12)
    savefig(fig2, joinpath(OUT, "figures", "Fig07_Biasi_profile_likelihoods.png"))

    # Figure 3: 2x2 Predictive Uncertainty Bands + Genuine Held-Out CV Curves
    ps_band = Any[]
    for (i, r) in enumerate(runs)
        c = rate_cols[i]
        sub_mb = sort(filter(q -> q.run_idx == i, uq_b3.mb.bands), :point_idx)
        sub_bb = sort(filter(q -> q.run_idx == i, uq_b3.bb.bands), :point_idx)
        cv_pred = b3.cv_obj.fold_preds[r.rate]
        cv_err = only(eachrow(filter(q -> q.excluded_rate_K_min == r.rate, b3.cv_obj.df))).cv_rmse_pp
        plt = plot(xlabel="Temperature (K)", ylabel="Remaining mass (%)",
            title=@sprintf("Biasi %g K/min: B3_M2_MAP + 95%% UQ & Excluded-Rate CV (%.3f pp)", r.rate, cv_err),
            legend=:bottomleft, ylims=(-2, 104), titlefontsize=10)
        scatter!(plt, r.T, 100.0 .* r.mass; color=:black, ms=1.8, msw=0, alpha=0.7, label="Experimental data")
        plot!(plt, r.T, 100.0 .* sub_bb.baseline;
            ribbon=(100.0 .* (sub_bb.baseline .- sub_bb.p025), 100.0 .* (sub_bb.p975 .- sub_bb.baseline)),
            fillalpha=0.18, color=:darkorange2, lw=0.0, label="BB 95% band")
        plot!(plt, r.T, 100.0 .* sub_mb.baseline;
            ribbon=(100.0 .* (sub_mb.baseline .- sub_mb.p025), 100.0 .* (sub_mb.p975 .- sub_mb.baseline)),
            fillalpha=0.28, color=c, lw=2.0, label="Joint fit ± MB 95% band")
        plot!(plt, r.T, 100.0 .* cv_pred; color=:firebrick3, linestyle=:dash, lw=1.8, label="Held-out CV prediction")
        push!(ps_band, plt)
    end
    fig3 = plot(ps_band...; layout=(2, 2), size=(1450, 920), margin=6Plots.mm,
        plot_title="Biasi T700/M21 95% Predictive Uncertainty Bands & Leave-One-Rate-Out CV", plot_titlefontsize=12)
    savefig(fig3, joinpath(OUT, "figures", "Fig08_Biasi_predictive_uncertainty_bands.png"))

    # Write clean Report.md
    blind_f_row = only(eachrow(filter(r -> r.parameter == "fiber_fraction", blk4.uq_b3_blind.tbl)))
    open(joinpath(OUT, "Report.md"), "w") do io
        println(io, "# External Validation: Published Biasi (2014) T700/M21 Benchmark\n")
        println(io, "Applied our unified thesis methodology (**Hybrid `GA + ForwardDiff-BFGS` + Solid-State `M2_MAP` + `KineticUQ` + `Catalyst/PEtab` verification**) to the independent published T700/M21 Air dataset (`2, 5, 10, 20 K/min`, `505-1250 K`, true fiber mass fraction `0.656`).\n")
        println(io, "## 1. Model Selection, Leave-One-Rate-Out CV & Blind Composition Recovery\n")
        println(io, "| Variant | Model | k | Fixed Fiber | Joint RMSE (pp) | Mean CV RMSE (pp) | CV RMS (pp) | n_eff | AICc_eff | E_char (kJ/mol) | E_fiber (kJ/mol) | Fiber Frac | PEtab Gap (pp) | Bounds |")
        println(io, "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in eachrow(model_sel_df)
            println(io, @sprintf("| %s | %s | %d | %s | %.3f | %.3f | %.3f | %.2f | %.2f | %.2f | %.2f | %.4f | %.2e | %s |",
                r.variant, r.base_model, r.k_params, string(r.fixed_fiber), r.joint_rmse_pp, r.cv_mean_pp, r.cv_rms_pp,
                r.n_eff, r.aicc_eff, r.E_char_kJ_mol, r.E_fiber_kJ_mol, r.fiber_fraction, r.petab_max_gap_pp, r.bound_hits))
        end
        println(io)
        println(io, "## 2. Blind `fiber_fraction` Recovery (Air-Only, Uninformed Initial Composition)\n")
        println(io, @sprintf("- **`B3_MAP_BlindFiber`:** Recovered `fiber_fraction = %.4f` (true `0.6560`, absolute error `%.4f` / `%.2f pp`); AR(1) 95%% profile CI `[%.4f, %.4f]` (%s); MB bootstrap `[%.4f, %.4f]`, BB bootstrap `[%.4f, %.4f]`.\n",
            blind_f_row.estimate, abs(blind_f_row.estimate - TRUE_BIASI_FIBER), 100.0 * abs(blind_f_row.estimate - TRUE_BIASI_FIBER),
            blind_f_row.ar1_prof_low, blind_f_row.ar1_prof_high, blind_f_row.identifiability,
            blind_f_row.block_boot_low, blind_f_row.block_boot_high, blind_f_row.bayes_boot_low, blind_f_row.bayes_boot_high))

        println(io, "## 3. Cross-Benchmark Literature & CFRP Alignment\n")
        println(io, "| Source | k | Joint RMSE (pp) | Mean CV (pp) | E_matrix (kJ/mol) | E_char (kJ/mol) | E_fiber (kJ/mol) | Fiber Frac |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in eachrow(lit_df)
            ks = ismissing(r.k_params) ? "-" : string(r.k_params)
            js = ismissing(r.joint_rmse_pp) ? "-" : @sprintf("%.3f", r.joint_rmse_pp)
            cs = ismissing(r.cv_mean_pp) ? "-" : @sprintf("%.3f", r.cv_mean_pp)
            println(io, @sprintf("| %s | %s | %s | %s | %.2f | %.2f | %.2f | %.4f |",
                r.source, ks, js, cs, r.E_matrix_kJ_mol, r.E_char_kJ_mol, r.E_fiber_kJ_mol, r.fiber_fraction))
        end
        println(io)

        println(io, "## 4. `B3_M2_MAP_CharPrior` (k = 8) Calibrated UQ & Guardrail Audit\n")
        println(io, "- **Guardrail Status:** `$(blk4.uq_b3.guardrails.status)` | **Identifiable:** `$(blk4.uq_b3.guardrails.identifiable_count)/$(blk4.uq_b3.guardrails.total_params)` | **BB:** `$(blk4.uq_b3.bb_ok)/$(blk4.uq_b3.n_boot)` | **MB:** `$(blk4.uq_b3.mb_ok)/$(blk4.uq_b3.n_boot)`\n")
        for a in blk4.uq_b3.guardrails.alerts
            println(io, "- **[$(a.severity)] (`$(a.code)`):** $(a.message)")
        end
        println(io, "\n| Parameter | Estimate | AR(1) 95% Profile CI | Block Boot 95% CI | Bayes Curve Boot 95% CI | Identifiability |")
        println(io, "|---|---:|---|---|---|---|")
        for r in eachrow(blk4.uq_b3.tbl)
            println(io, @sprintf("| `%s` | %.4f | [%.4f, %.4f] | [%.4f, %.4f] | [%.4f, %.4f] | %s |",
                r.parameter, r.estimate, r.ar1_prof_low, r.ar1_prof_high, r.block_boot_low, r.block_boot_high,
                r.bayes_boot_low, r.bayes_boot_high, r.identifiability))
        end
        println(io, "\n![Biasi solid-state fits and CV comparison](figures/Fig06_Biasi_solid_state_fits_and_cv.png)")
        println(io, "\n![Biasi M2-MAP and Blind-Fiber profile likelihoods](figures/Fig07_Biasi_profile_likelihoods.png)")
        println(io, "\n![Biasi predictive uncertainty bands](figures/Fig08_Biasi_predictive_uncertainty_bands.png)")
    end

    println("\nPHASE6_OUTPUT_DIRECTORY=", OUT)
end

main()
