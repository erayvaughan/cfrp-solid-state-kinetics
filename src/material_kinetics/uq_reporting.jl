module MaterialUQReporting

using CSV, DataFrames, Statistics, LinearAlgebra, ForwardDiff, Printf, TOML, SHA
import Fides
ENV["GKSwstype"] = "100"
using Plots

if !isdefined(@__MODULE__, :KineticUQ)
    include(joinpath(@__DIR__, "..", "uncertainty", "KineticUQ.jl"))
end
using .KineticUQ

export evaluate_guardrails, material_derived_profiles, material_derived_triplets,
       run_material_uq, append_uq_markdown

function _route_coords(spec)
    if spec.family in ("contracting_cylinder", "contracting_sphere")
        return [("route_1", "logk1", "E1", "fixed_geom", spec.refs[1])]
    elseif spec.family == "random_pore"
        return [("route_1", "logk1", "E1", "psi1", spec.refs[1])]
    elseif spec.family == "avrami_erofeev"
        return [("route_1", "logk1", "E1", "m1", spec.refs[1])]
    elseif spec.family == "sestak_berggren"
        return [("route_1", "logk1", "E1", "n1", spec.refs[1])]
    end
    routes = [("route_1", "logk1", "E1", "n1", spec.refs[1])]
    spec.family != "single" && push!(routes, ("route_2", "logk2", "E2", "n2", spec.refs[2]))
    spec.family == "composite" && push!(routes, ("fiber", "logkf", "Ef", "nf", spec.refs[3]))
    spec.family == "early_sequential" && push!(routes, ("early_loss", "logk0", "E0", "fixed_first_order", spec.refs[3]))
    routes
end

function material_derived_profiles(spec)
    name_to_idx = Dict(n => i for (i, n) in enumerate(spec.names))
    ru = Symbol(get(spec, :rate_unit, "min_inv"))
    [(name="log10_A$(k)_s_inv", logk_idx=name_to_idx[rt[2]], E_idx=name_to_idx[rt[3]],
      ref_K=Float64(rt[5]), rate_unit=ru)
     for (k, rt) in enumerate(_route_coords(spec))
     if haskey(name_to_idx, rt[2]) && haskey(name_to_idx, rt[3])]
end

function material_derived_triplets(mk_mod, spec, x)
    kt = mk_mod.kinetic_table(spec, x)
    out = Pair{String,Float64}[]
    for k in 1:nrow(kt)
        push!(out, "E$(k)_kJ_mol" => Float64(kt.E_kJ_mol[k]))
        push!(out, "log10_A$(k)_s_inv" => log10(max(Float64(kt.A_per_s[k]), 1e-300)))
    end
    out
end

function _max_comp_corr(boot)
    boot === nothing && return NaN
    comp = boot.compensation
    isempty(comp) && return NaN
    vals = filter(isfinite, Float64.(comp.corr_E_log10A))
    isempty(vals) ? NaN : maximum(abs, vals)
end

function evaluate_guardrails(rates::AbstractVector{<:Real}, uq_thresh, prof, bb, mb;
                             k_params::Integer, near_bounds::AbstractVector{<:AbstractString}=String[])
    n_rates = length(unique(round.(Float64.(rates); digits=4)))
    n_runs = length(rates)
    n_eff = Float64(uq_thresh.n_eff)
    rho1 = Float64(uq_thresh.median_lag1_autocorr)
    ci_inflation = sqrt(Float64(uq_thresh.loss_ratio_f) / max(Float64(uq_thresh.loss_ratio_naive), 1e-12))

    summary_df = prof.summary
    identifiable_count = count(==("identifiable"), String.(summary_df.identifiability))
    total_params = nrow(summary_df)
    non_identifiable = [String(r.parameter) for r in eachrow(summary_df) if String(r.identifiability) != "identifiable"]

    c_bb = _max_comp_corr(bb)
    c_mb = _max_comp_corr(mb)
    comp_candidates = filter(isfinite, [c_bb, c_mb])
    max_comp_corr = isempty(comp_candidates) ? NaN : maximum(comp_candidates)

    alerts = NamedTuple{(:severity, :code, :message)}[]
    if n_rates == 1
        push!(alerts, (severity="CRITICAL", code="single_rate",
            message="Single heating rate detected (n_rates=1): Arrhenius kinetic triplets (A, E, n) are structurally unidentifiable."))
    elseif n_rates == 2
        push!(alerts, (severity="CRITICAL", code="two_rates_only",
            message="Only 2 heating rates detected (n_rates=2): parameter CIs inflate severely and reaction orders/multi-step routes are generally non-identifiable; collect >= 3 heating rates."))
    end
    if n_eff < k_params + 5
        push!(alerts, (severity="WARNING", code="low_effective_dof",
            message=@sprintf("Effective sample size n_eff=%.1f is close to k=%d free parameters (< k+5); autocorrelation leaves minimal statistical degrees of freedom.", n_eff, k_params)))
    end
    if !isempty(near_bounds)
        push!(alerts, (severity="WARNING", code="near_bounds",
            message="Fitted parameters near optimization bounds: " * join(String.(near_bounds), ", ")))
    end
    if !isempty(non_identifiable)
        push!(alerts, (severity="WARNING", code="non_identifiable_params",
            message="Parameters not classified as identifiable under AR(1) profile likelihood: " * join(non_identifiable, ", ")))
    end
    if isfinite(max_comp_corr) && max_comp_corr > 0.99
        push!(alerts, (severity="WARNING", code="arrhenius_compensation",
            message=@sprintf("Severe Arrhenius (E, log10 A) compensation correlation (max |r|=%.4f > 0.99).", max_comp_corr)))
    end

    status = any(a -> a.severity == "CRITICAL", alerts) ? "critical_warnings" :
             isempty(alerts) ? "ok" : "warnings"

    (; status, n_rates, n_runs, k_params=Int(k_params), n_eff, rho1, ci_inflation,
       identifiable_count, total_params, near_bounds=String.(near_bounds),
       non_identifiable, max_comp_corr, alerts)
end

function _mk_fit_local(mk_mod, study, spec, synth_masses, x_start; seconds=15.0, maxiter=600)
    synth_runs = [merge(r, (mass=synth_masses[i],)) for (i, r) in enumerate(study.runs)]
    synth_study = merge(study, (runs=synth_runs,))
    lf = z -> mk_mod.loss(synth_study, spec, z; level=2)
    cfg = ForwardDiff.GradientConfig(lf, x_start)
    res = ForwardDiff.DiffResults.GradientResult(copy(x_start))
    obj = z -> begin
        ForwardDiff.gradient!(res, lf, z, cfg)
        (ForwardDiff.DiffResults.value(res), copy(ForwardDiff.DiffResults.gradient(res)))
    end
    x0 = clamp.(x_start, spec.lower .+ 1e-7 .* (spec.upper .- spec.lower),
                        spec.upper .- 1e-7 .* (spec.upper .- spec.lower))
    raw = Fides.solve(Fides.FidesProblem(obj, x0; lb=spec.lower, ub=spec.upper), Fides.BFGS();
        options=Fides.FidesOptions(maxiter=maxiter, maxtime=Float64(seconds), fatol=1e-12, frtol=1e-9, gatol=1e-7, verbose="error"))
    xc = clamp.(collect(Float64, raw.xmin), spec.lower, spec.upper)
    v, g = obj(xc)
    pg = maximum(abs.(xc .- clamp.(xc .- g, spec.lower, spec.upper)))
    (x=xc, stationary=(isfinite(v) && pg < 1e-4), pg=pg)
end

function _combine_summary(material_id::String, model_id::String, prof_ar1::DataFrame, prof_naive::DataFrame,
                          bb_sum::DataFrame, mb_sum::DataFrame)
    rows = NamedTuple[]
    for r in eachrow(prof_ar1)
        pname = String(r.parameter)
        pn = first(eachrow(filter(q -> q.parameter == pname, prof_naive)))
        brows = filter(q -> q.parameter == pname, bb_sum)
        mrows = filter(q -> q.parameter == pname, mb_sum)
        br = isempty(brows) ? (ci_low=NaN, ci_high=NaN, ci_width=NaN, boot_median=NaN) : first(eachrow(brows))
        mr = isempty(mrows) ? (ci_low=NaN, ci_high=NaN, ci_width=NaN, boot_median=NaN) : first(eachrow(mrows))
        pkind = startswith(pname, "log10_A") ? "derived" : "parameter"
        push!(rows, (
            material=material_id, model=model_id, parameter=pname, kind=pkind,
            mle=r.estimate,
            prof_ar1_lo=r.ci_low, prof_ar1_hi=r.ci_high, prof_ar1_width=r.ci_width,
            identifiability=String(r.identifiability), bound_active=Bool(r.bound_active),
            prof_naive_lo=pn.ci_low, prof_naive_hi=pn.ci_high, prof_naive_width=pn.ci_width,
            ar1_to_naive_ratio=r.ci_width / max(pn.ci_width, 1e-12),
            bb_median=br.boot_median, bb_lo=br.ci_low, bb_hi=br.ci_high,
            mb_median=mr.boot_median, mb_lo=mr.ci_low, mb_hi=mr.ci_high,
        ))
    end
    DataFrame(rows)
end

function _band_summary(material_id::String, model_id::String, method::String, runs, bands_df::DataFrame)
    rows = NamedTuple[]
    for (i, r) in enumerate(runs)
        sub = filter(q -> q.run_idx == i, bands_df)
        isempty(sub) && continue
        push!(rows, (
            material=material_id, model=model_id, method=method,
            condition=r.id, rate_C_min=r.rate,
            mean_width_pp=mean(sub.width_pp), max_width_pp=maximum(sub.width_pp),
        ))
    end
    DataFrame(rows)
end

function _plot_profiles(prof_ar1, thr, spec, title_prefix::String, path::String)
    curves = prof_ar1.curves
    params = collect(prof_ar1.summary.parameter)
    np = length(params)
    ncols = min(4, max(1, ceil(Int, sqrt(np))))
    nrows = ceil(Int, np / ncols)
    y_cap = max(3.5 * thr.loss_ratio_f, 4.0)
    subplots = Any[]
    for p in params
        sub = filter(r -> r.parameter == p, curves)
        xs = sub.target
        ys = clamp.(sub.loss_ratio, 0.98, y_cap)
        srow = first(eachrow(filter(r -> r.parameter == p, prof_ar1.summary)))
        plt = plot(xs, ys; lw=2.0, color=:dodgerblue3, label="", title="$p ($(srow.identifiability))",
            xlabel=p, ylabel="L / L_min", ylims=(0.98, min(maximum(ys) * 1.12 + 0.05, y_cap)),
            titlefontsize=9, guidefontsize=8, tickfontsize=7)
        hline!(plt, [thr.loss_ratio_f]; color=:firebrick3, linestyle=:dash, lw=1.4, label="")
        hline!(plt, [thr.loss_ratio_naive]; color=:gray45, linestyle=:dot, lw=1.2, label="")
        vline!(plt, [srow.estimate]; color=:black, linestyle=:dashdot, lw=1.0, label="")
        push!(subplots, plt)
    end
    fig = plot(subplots...; layout=(nrows, ncols), size=(380 * ncols, 290 * nrows),
        plot_title="$title_prefix — AR(1) (red) vs Naive (gray) 95% Profiles", plot_titlefontsize=11,
        margin=4Plots.mm)
    savefig(fig, path)
end

function _plot_predictive_bands(runs, predictions, bb, mb, title_prefix::String, path::String)
    colors = [:dodgerblue3, :darkorange2, :seagreen4, :purple, :firebrick3, :teal]
    plt = plot(xlabel="Recorded temperature (°C)", ylabel="Remaining mass (%)",
        title="$title_prefix — 95% Predictive Uncertainty Bands",
        legend=:bottomleft, size=(1250, 780), left_margin=8Plots.mm, bottom_margin=6Plots.mm,
        titlefontsize=12, guidefontsize=11, tickfontsize=9, legendfontsize=9)
    for (i, r) in enumerate(runs)
        c = colors[mod1(i, length(colors))]
        T = r.observation_temperature .- 273.15
        scatter!(plt, T, 100.0 .* r.mass; color=c, ms=1.6, msw=0, alpha=0.65,
            label="$(r.rate) °C/min data")
        if mb !== nothing
            sub = sort(filter(q -> q.run_idx == i, mb.bands), :point_idx)
            if nrow(sub) == length(T)
                mid = 100.0 .* sub.baseline
                lo_rib = 100.0 .* (sub.baseline .- sub.p025)
                hi_rib = 100.0 .* (sub.p975 .- sub.baseline)
                plot!(plt, T, mid; ribbon=(lo_rib, hi_rib), fillalpha=0.20, color=c, lw=2.2,
                    label="$(r.rate) °C/min fit ± MB 95%")
            else
                plot!(plt, T, 100.0 .* predictions[i]; color=c, lw=2.2, label="$(r.rate) °C/min fit")
            end
        elseif bb !== nothing
            sub = sort(filter(q -> q.run_idx == i, bb.bands), :point_idx)
            if nrow(sub) == length(T)
                mid = 100.0 .* sub.baseline
                lo_rib = 100.0 .* (sub.baseline .- sub.p025)
                hi_rib = 100.0 .* (sub.p975 .- sub.baseline)
                plot!(plt, T, mid; ribbon=(lo_rib, hi_rib), fillalpha=0.20, color=c, lw=2.2,
                    label="$(r.rate) °C/min fit ± BB 95%")
            else
                plot!(plt, T, 100.0 .* predictions[i]; color=c, lw=2.2, label="$(r.rate) °C/min fit")
            end
        else
            plot!(plt, T, 100.0 .* predictions[i]; color=c, lw=2.2, label="$(r.rate) °C/min fit")
        end
    end
    savefig(plt, path)
end

function run_material_uq(mk_mod, study, spec, x, predictions, diagnostics, out;
                         n_boot::Int=30, seed::Int=20261122, level::Int=2,
                         seconds_per_rep::Real=15.0, maxiter::Int=600,
                         profile_steps::Int=12)
    uq_dir = joinpath(out, "uq")
    mkpath(uq_dir)
    mat_id = String(study.config["id"])
    mod_id = String(spec.id)
    x_hat = Float64.(x)

    resids = [Float64.(study.runs[i].mass) .- Float64.(predictions[i]) for i in eachindex(study.runs)]
    thr = KineticUQ.effective_uq_thresholds(resids, length(x_hat); level=0.95)
    derived_prof = material_derived_profiles(spec)

    loss_fn = z -> mk_mod.loss(study, spec, z; level=level)
    w_loss = (z, w) -> mk_mod.weighted_loss(study, spec, z, w; level=level)
    pred_fn = z -> mk_mod.predict(study, spec, z; level=level)
    derived_fn = z -> material_derived_triplets(mk_mod, spec, z)

    prof_ar1 = KineticUQ.profile_likelihood_uq(loss_fn, x_hat, spec.lower, spec.upper, spec.names, thr.loss_ratio_f;
        derived_profiles=derived_prof, max_steps=profile_steps, seconds_per_step=3.5, maxiter=180)
    prof_naive = KineticUQ.profile_likelihood_uq(loss_fn, x_hat, spec.lower, spec.upper, spec.names, thr.loss_ratio_naive;
        derived_profiles=derived_prof, max_steps=8, seconds_per_step=2.5, maxiter=140)

    bb = nothing
    bb_ok = 0
    if length(study.runs) >= 2
        try
            bb = KineticUQ.bayesian_curve_bootstrap(w_loss, pred_fn, derived_fn, x_hat, spec.lower, spec.upper,
                spec.names, length(study.runs); n_boot=n_boot, seed=seed + 1, seconds_per_rep=seconds_per_rep, maxiter=maxiter)
            bb_ok = size(bb.replicates, 2)
        catch err
            @warn "Bayesian curve bootstrap skipped for $mat_id/$mod_id: $(sprint(showerror, err))"
            bb = nothing
        end
    end

    mb = nothing
    mb_ok = 0
    try
        mb = KineticUQ.moving_block_bootstrap(
            (m, x0) -> _mk_fit_local(mk_mod, study, spec, m, x0; seconds=seconds_per_rep, maxiter=maxiter),
            [r.mass for r in study.runs], predictions, pred_fn, derived_fn, x_hat, spec.names;
            n_boot=n_boot, seed=seed + 2)
        mb_ok = size(mb.replicates, 2)
    catch err
        @warn "Moving-block bootstrap skipped for $mat_id/$mod_id: $(sprint(showerror, err))"
        mb = nothing
    end

    empty_sum = DataFrame(parameter=String[], ci_low=Float64[], ci_high=Float64[], ci_width=Float64[], boot_median=Float64[])
    bb_sum = bb === nothing ? empty_sum : bb.summary
    mb_sum = mb === nothing ? empty_sum : mb.summary

    summary_df = _combine_summary(mat_id, mod_id, prof_ar1.summary, prof_naive.summary, bb_sum, mb_sum)
    summary_path = joinpath(uq_dir, "uq_summary.csv")
    CSV.write(summary_path, summary_df)

    comp_parts = DataFrame[]
    if bb !== nothing && !isempty(bb.compensation)
        d1 = copy(bb.compensation); d1[!, :material] .= mat_id; d1[!, :model] .= mod_id; push!(comp_parts, d1)
    end
    if mb !== nothing && !isempty(mb.compensation)
        d2 = copy(mb.compensation); d2[!, :material] .= mat_id; d2[!, :model] .= mod_id; push!(comp_parts, d2)
    end
    comp_df = isempty(comp_parts) ? DataFrame(
        channel=Int[], E_param=String[], A_param=String[], corr_E_log10A=Float64[],
        slope_log10A_per_kJ=Float64[], T_iso_K=Float64[], E_ci_width_kJ_mol=Float64[],
        log10A_ci_width=Float64[], method=String[], n_converged=Int[], material=String[], model=String[]
    ) : reduce(vcat, comp_parts; cols=:union)
    comp_path = joinpath(uq_dir, "compensation_correlations.csv")
    CSV.write(comp_path, comp_df)

    band_parts = DataFrame[]
    if bb !== nothing && !isempty(bb.bands)
        push!(band_parts, _band_summary(mat_id, mod_id, "bayesian_curve_bootstrap", study.runs, bb.bands))
    end
    if mb !== nothing && !isempty(mb.bands)
        push!(band_parts, _band_summary(mat_id, mod_id, "moving_block_bootstrap", study.runs, mb.bands))
    end
    bands_df = isempty(band_parts) ? DataFrame(
        material=String[], model=String[], method=String[], condition=String[],
        rate_C_min=Float64[], mean_width_pp=Float64[], max_width_pp=Float64[]
    ) : reduce(vcat, band_parts; cols=:union)
    bands_path = joinpath(uq_dir, "predictive_bands_summary.csv")
    CSV.write(bands_path, bands_df)

    rates = [Float64(r.rate) for r in study.runs]
    near_b = String.(get(diagnostics, "near_bounds", String[]))
    guardrails = evaluate_guardrails(rates, thr, prof_ar1, bb, mb; k_params=length(x_hat), near_bounds=near_b)

    guardrails_path = joinpath(uq_dir, "uq_guardrails.toml")
    gtoml = Dict{String,Any}(
        "material" => mat_id,
        "model" => mod_id,
        "status" => guardrails.status,
        "n_rates" => guardrails.n_rates,
        "n_runs" => guardrails.n_runs,
        "k_params" => guardrails.k_params,
        "n_obs" => thr.n_obs,
        "n_eff" => guardrails.n_eff,
        "nu_eff" => thr.nu_eff,
        "median_lag1_autocorr" => guardrails.rho1,
        "ci_inflation_vs_naive" => guardrails.ci_inflation,
        "ar1_loss_ratio_95" => thr.loss_ratio_f,
        "naive_loss_ratio_95" => thr.loss_ratio_naive,
        "identifiable_count" => guardrails.identifiable_count,
        "total_profiled" => guardrails.total_params,
        "near_bounds" => guardrails.near_bounds,
        "non_identifiable" => guardrails.non_identifiable,
        "max_comp_corr" => isfinite(guardrails.max_comp_corr) ? guardrails.max_comp_corr : "unavailable",
        "n_boot_requested" => n_boot,
        "bb_converged" => bb_ok,
        "mb_converged" => mb_ok,
        "alerts" => [Dict("severity" => a.severity, "code" => a.code, "message" => a.message) for a in guardrails.alerts],
    )
    open(guardrails_path, "w") do io
        TOML.print(io, gtoml; sorted=true)
    end

    prof_plot_path = joinpath(uq_dir, "uq_profiles.png")
    band_plot_path = joinpath(uq_dir, "uq_predictive_bands.png")
    _plot_profiles(prof_ar1, thr, spec, "$mat_id / $mod_id", prof_plot_path)
    _plot_predictive_bands(study.runs, predictions, bb, mb, "$mat_id / $mod_id", band_plot_path)

    (; uq_dir, mat_id, mod_id, thr, prof_ar1, prof_naive, bb, mb, bb_ok, mb_ok, n_boot,
       summary_df, comp_df, bands_df, guardrails,
       summary_path, comp_path, bands_path, guardrails_path, prof_plot_path, band_plot_path)
end

function append_uq_markdown(report_path::AbstractString, uq_res)
    g = uq_res.guardrails
    t = uq_res.thr
    open(report_path, "a") do io
        println(io, "\n## Calibrated Uncertainty & Identifiability (KineticUQ)\n")
        println(io, @sprintf("Display model profiled: `%s` (`k = %d` free parameters; `n_obs = %d`, AR(1) effective sample size `n_eff = %.2f`, `nu_eff = %.2f`, median lag-1 autocorr `rho1 = %.4f`, 95%% AR(1)-to-naive CI inflation = `%.2fx`).\n",
            uq_res.mod_id, g.k_params, t.n_obs, t.n_eff, t.nu_eff, g.rho1, g.ci_inflation))
        bb_txt = g.n_runs >= 2 ? "$(uq_res.bb_ok)/$(uq_res.n_boot)" : "skipped (n_runs < 2)"
        println(io, "Bootstrap replicates converged: Bayesian curve bootstrap = `$(bb_txt)`, Moving-block bootstrap = `$(uq_res.mb_ok)/$(uq_res.n_boot)`. Identifiable coordinates under AR(1) profile likelihood: **$(g.identifiable_count)/$(g.total_params)**.\n")

        println(io, "### Guardrails & Data-Sufficiency Warnings\n")
        if isempty(g.alerts)
            println(io, "- **OK:** No critical or warning guardrails triggered (`status = ok`).\n")
        else
            for a in g.alerts
                println(io, "- **[$(a.severity)] (`$(a.code)`):** $(a.message)")
            end
            println(io)
        end

        println(io, "### Parameter Profile Likelihood & Bootstrap 95% CIs\n")
        println(io, "| Parameter | Kind | MLE | AR(1) 95% Profile CI | Naive 95% Profile CI | BB 95% CI | MB 95% CI | Identifiability |")
        println(io, "|---|---|---:|---|---|---|---|---|")
        fmt_ci(lo, hi) = (isfinite(lo) && isfinite(hi)) ? @sprintf("[%.4g, %.4g]", lo, hi) : "N/A"
        for r in eachrow(uq_res.summary_df)
            println(io, @sprintf("| `%s` | %s | %.5g | %s | %s | %s | %s | %s |",
                r.parameter, r.kind, r.mle,
                fmt_ci(r.prof_ar1_lo, r.prof_ar1_hi),
                fmt_ci(r.prof_naive_lo, r.prof_naive_hi),
                fmt_ci(r.bb_lo, r.bb_hi),
                fmt_ci(r.mb_lo, r.mb_hi),
                r.identifiability))
        end
        println(io, "\n![Profile likelihoods](uq/uq_profiles.png)\n\n![Predictive uncertainty bands](uq/uq_predictive_bands.png)\n")
        println(io, "Machine-readable UQ files: [uq_summary.csv](uq/uq_summary.csv), [compensation_correlations.csv](uq/compensation_correlations.csv), [predictive_bands_summary.csv](uq/predictive_bands_summary.csv), [uq_guardrails.toml](uq/uq_guardrails.toml).\n")
    end
end

end
