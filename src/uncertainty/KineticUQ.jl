module KineticUQ

using Statistics, LinearAlgebra, ForwardDiff, DataFrames, Printf
import Fides
using StableRNGs: StableRNG

include("../material_kinetics/model_selection.jl")

const GAS_R_UQ = 8.31446261815324

function _simpson_integral(f, a::Float64, b::Float64, panels::Int)
    b <= a && return 0.0
    n = iseven(panels) ? panels : panels + 1
    h = (b - a) / n
    s = f(a) + f(b)
    @inbounds for i in 1:(n - 1)
        s += (isodd(i) ? 4.0 : 2.0) * f(a + i * h)
    end
    s * (h / 3.0)
end

"""
Acklam inverse standard normal CDF (accuracy ~1.15e-9 across (0, 1)).
"""
function norminvcdf(p::Real)
    0.0 < p < 1.0 || error("Probability must lie in (0, 1), got $p")
    p_low = 0.02425
    p_high = 1.0 - p_low
    a1, a2, a3, a4, a5, a6 = -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
    b1, b2, b3, b4, b5 = -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01
    c1, c2, c3, c4, c5, c6 = -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
    d1, d2, d3, d4 = 7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00

    pf = Float64(p)
    if pf < p_low
        q = sqrt(-2.0 * log(pf))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elseif pf <= p_high
        q = pf - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else
        q = sqrt(-2.0 * log(1.0 - pf))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    end
end

"""
Exact F(1, df) upper quantile at confidence `level` (two-sided Student-t squared).
Supports real-valued effective degrees of freedom `df >= 1.0`.
"""
function f_quantile_1_df(df::Real; level::Real=0.95)
    nu = Float64(df)
    nu >= 1.0 || error("Degrees of freedom must be >= 1.0, got $nu")
    0.5 < level < 1.0 || error("Confidence level must lie in (0.5, 1.0), got $level")
    p_one_sided = 0.5 * (1.0 + Float64(level))
    z = norminvcdf(p_one_sided)

    if nu > 200.0
        # 4-term Cornish-Fisher expansion for t_{p, nu}
        z2 = z * z
        t = z + (z * z2 + z) / (4.0 * nu) +
            (5.0 * z * z2^2 + 16.0 * z * z2 + 3.0 * z) / (96.0 * nu^2) +
            (3.0 * z * z2^3 + 19.0 * z * z2^2 + 17.0 * z * z2 - 15.0 * z) / (384.0 * nu^3)
        return t * t
    end

    # Use substituted integral I(phi, nu) = int_0^phi sin(psi)^(nu-1) dpsi
    # via psi = phi * s^(1/nu) when 1 <= nu < 2, and direct cos^(nu-1) Simpson when nu >= 2.
    half_pi = 0.5 * pi
    if nu < 2.0
        pow_inv = 1.0 / nu
        sin_tail(phi_rem) = begin
            phi_rem <= 0.0 && return 0.0
            pref = (phi_rem^nu) / nu
            pref * _simpson_integral(s -> begin
                s <= 0.0 && return 1.0
                ang = phi_rem * s^pow_inv
                (sin(ang) / ang)^(nu - 1.0)
            end, 0.0, 1.0, 256)
        end
        denom = sin_tail(half_pi)
        _cdf_lt2(t_val) = begin
            phi = atan(t_val / sqrt(nu))
            1.0 - sin_tail(half_pi - phi) / denom
        end
        lo = 0.0
        hi = 2.0
        while _cdf_lt2(hi) < level && hi < 1e6
            hi *= 2.0
        end
        for _ in 1:64
            mid = 0.5 * (lo + hi)
            if _cdf_lt2(mid) < level
                lo = mid
            else
                hi = mid
            end
        end
        t_val = 0.5 * (lo + hi)
        return t_val * t_val
    else
        integrand = theta -> cos(theta)^(nu - 1.0)
        denom = _simpson_integral(integrand, 0.0, half_pi, 256)
        _cdf_ge2(t_val) = _simpson_integral(integrand, 0.0, atan(t_val / sqrt(nu)), 256) / denom
        lo = 0.0
        hi = max(5.0, 2.0 * z)
        while _cdf_ge2(hi) < level && hi < 1e4
            hi *= 2.0
        end
        for _ in 1:64
            mid = 0.5 * (lo + hi)
            if _cdf_ge2(mid) < level
                lo = mid
            else
                hi = mid
            end
        end
        t_val = 0.5 * (lo + hi)
        return t_val * t_val
    end
end

function effective_uq_thresholds(residuals_by_run::AbstractVector, k_params::Integer; level::Real=0.95)
    ic = information_criteria(residuals_by_run, k_params)
    nu_eff = max(ic.n_eff - k_params, 1.0)
    nu_naive = max(ic.n_obs - k_params, 1.0)
    f_eff = f_quantile_1_df(nu_eff; level)
    f_naive = f_quantile_1_df(nu_naive; level)
    chi2_crit = f_quantile_1_df(1e9; level)

    loss_ratio_f = 1.0 + f_eff / nu_eff
    loss_ratio_chi2_eff = exp(chi2_crit / max(ic.n_eff, 1.0))
    loss_ratio_naive = 1.0 + f_naive / nu_naive

    (; level=Float64(level),
       n_runs=ic.n_runs, n_obs=ic.n_obs, n_eff=ic.n_eff, k_params=Int(k_params),
       nu_eff, nu_naive, median_lag1_autocorr=ic.median_lag1_autocorr, rmse_pp=ic.rmse_pp,
       f_crit_eff=f_eff, f_crit_naive=f_naive, chi2_crit,
       loss_ratio_f, rmse_ratio_f=sqrt(loss_ratio_f),
       loss_ratio_chi2_eff, rmse_ratio_chi2_eff=sqrt(loss_ratio_chi2_eff),
       loss_ratio_naive, rmse_ratio_naive=sqrt(loss_ratio_naive))
end

_proj_grad(x, g, lo, hi) = maximum(abs.(x .- clamp.(x .- g, lo, hi)))

function _solve_reduced(f_red, lo_red, hi_red, u_start, seconds_per_step, maxiter)
    if isempty(lo_red)
        v = Float64(f_red(Float64[]))
        return (u=Float64[], loss=v, stationary=isfinite(v), status="1D_exact", pg=0.0)
    end
    cfg = ForwardDiff.GradientConfig(f_red, u_start)
    res = ForwardDiff.DiffResults.GradientResult(copy(u_start))
    obj = u -> begin
        ForwardDiff.gradient!(res, f_red, u, cfg)
        (ForwardDiff.DiffResults.value(res), copy(ForwardDiff.DiffResults.gradient(res)))
    end
    u0 = clamp.(u_start, lo_red .+ 1e-9 .* max.(1.0, hi_red .- lo_red),
                         hi_red .- 1e-9 .* max.(1.0, hi_red .- lo_red))
    u0 = clamp.(u0, lo_red, hi_red)
    try
        raw = Fides.solve(
            Fides.FidesProblem(obj, u0; lb=lo_red, ub=hi_red),
            Fides.BFGS();
            options=Fides.FidesOptions(maxiter=maxiter, maxtime=Float64(seconds_per_step),
                fatol=1e-13, frtol=1e-10, gatol=1e-8, verbose="error"),
        )
        u_sol = clamp.(collect(Float64, raw.xmin), lo_red, hi_red)
        v, g = obj(u_sol)
        pg = _proj_grad(u_sol, g, lo_red, hi_red)
        return (u=u_sol, loss=Float64(v), stationary=(isfinite(v) && pg < 1e-5), status=string(raw.retcode), pg=pg)
    catch err
        err isa InterruptException && rethrow()
        return (u=copy(u0), loss=Inf, stationary=false, status="failed", pg=Inf)
    end
end

function _bisect_crossing(eval_at, t_below::Float64, l_below::Float64, u_below::Vector{Float64},
                          t_above::Float64, l_above::Float64, loss_crit::Float64)
    tb, lb, ub = t_below, l_below, copy(u_below)
    ta, la = t_above, l_above
    mid_points = NamedTuple[]
    for bstep in 1:6
        t_mid = 0.5 * (tb + ta)
        sol = eval_at(t_mid, ub)
        bstep <= 3 && push!(mid_points, (target=t_mid, loss=sol.loss, stationary=sol.stationary, status=sol.status))
        if isfinite(sol.loss) && sol.loss < loss_crit
            tb, lb, ub = t_mid, sol.loss, sol.u
        elseif isfinite(sol.loss)
            ta, la = t_mid, sol.loss
        else
            break
        end
    end
    denom = max(la - lb, 1e-30)
    frac = clamp((loss_crit - lb) / denom, 0.0, 1.0)
    t_cross = tb + frac * (ta - tb)
    (t_cross, mid_points)
end

function _walk_profile(name::String, x0_coord::Float64, lo_coord::Float64, hi_coord::Float64,
                       init_step::Float64, base_loss::Float64, loss_crit::Float64,
                       make_subproblem, max_steps::Int, seconds_per_step::Real, maxiter::Int)
    span = hi_coord - lo_coord
    edge_tol = 1e-4 * span
    curve = [(parameter=name, target=x0_coord, delta_loss=0.0, loss=base_loss,
              loss_ratio=1.0, rmse_pp=100.0 * sqrt(2.0 * base_loss),
              rmse_ratio=1.0, stationary=true, status="mle")]

    left_active = (x0_coord - lo_coord) <= edge_tol
    right_active = (hi_coord - x0_coord) <= edge_tol

    side_results = Dict{Int,NamedTuple}()
    for (dir, is_active, bound_end) in ((-1, left_active, lo_coord), (1, right_active, hi_coord))
        if is_active
            side_results[dir] = (crossed=false, active=true, bound=bound_end, n_steps=0)
            continue
        end
        t_prev = x0_coord
        l_prev = base_loss
        sub0 = make_subproblem(x0_coord)
        u_prev = isnothing(sub0) ? Float64[] : copy(sub0.u_init)
        step_sz = clamp(init_step, 1e-5 * span, 0.20 * span)
        crossed = false
        bound_val = bound_end
        n_steps = 0

        for _ in 1:max_steps
            t_next = dir == -1 ? max(lo_coord, t_prev - step_sz) : min(hi_coord, t_prev + step_sz)
            abs(t_next - t_prev) <= 1e-11 * max(1.0, abs(span)) && break
            sub = make_subproblem(t_next)
            isnothing(sub) && break
            u_warm = length(u_prev) == length(sub.lo) ? clamp.(u_prev, sub.lo, sub.hi) : sub.u_init
            sol = _solve_reduced(sub.f, sub.lo, sub.hi, u_warm, seconds_per_step, maxiter)
            n_steps += 1
            if !isfinite(sol.loss)
                break
            end
            ratio = sol.loss / max(base_loss, 1e-30)
            push!(curve, (parameter=name, target=t_next, delta_loss=sol.loss - base_loss,
                          loss=sol.loss, loss_ratio=ratio, rmse_pp=100.0 * sqrt(2.0 * sol.loss),
                          rmse_ratio=sqrt(max(ratio, 0.0)), stationary=sol.stationary, status=sol.status))
            if sol.loss >= loss_crit
                eval_at = (tv, uw) -> begin
                    s2 = make_subproblem(tv)
                    isnothing(s2) && return (u=uw, loss=Inf, stationary=false, status="infeasible", pg=Inf)
                    uw2 = length(uw) == length(s2.lo) ? clamp.(uw, s2.lo, s2.hi) : s2.u_init
                    _solve_reduced(s2.f, s2.lo, s2.hi, uw2, seconds_per_step, maxiter)
                end
                t_cross, mids = _bisect_crossing(eval_at, t_prev, l_prev, u_prev, t_next, sol.loss, loss_crit)
                for m in mids
                    if isfinite(m.loss)
                        mr = m.loss / max(base_loss, 1e-30)
                        push!(curve, (parameter=name, target=m.target, delta_loss=m.loss - base_loss,
                                      loss=m.loss, loss_ratio=mr, rmse_pp=100.0 * sqrt(2.0 * m.loss),
                                      rmse_ratio=sqrt(max(mr, 0.0)), stationary=m.stationary, status=m.status))
                    end
                end
                crossed = true
                bound_val = t_cross
                break
            end
            bound_val = t_next
            t_prev, l_prev, u_prev = t_next, sol.loss, sol.u
            (dir == -1 && t_next <= lo_coord + 1e-12) && break
            (dir == 1 && t_next >= hi_coord - 1e-12) && break
            step_sz = min(step_sz * 1.4, 0.25 * span)
        end
        side_results[dir] = (crossed=crossed, active=false, bound=bound_val, n_steps=n_steps)
    end

    L = side_results[-1]
    R = side_results[1]
    ident = if L.crossed && R.crossed
        "identifiable"
    elseif (L.active && R.crossed) || (R.active && L.crossed)
        "bound_active"
    elseif !L.crossed && !R.crossed && !L.active && !R.active
        "practically_non_identifiable_both"
    elseif !L.crossed && !L.active
        "practically_non_identifiable_lower"
    elseif !R.crossed && !R.active
        "practically_non_identifiable_upper"
    else
        "bound_active_non_identifiable"
    end

    ci_lo = L.active ? lo_coord : L.bound
    ci_hi = R.active ? hi_coord : R.bound
    sort!(curve; by=r -> r.target)
    summary_row = (
        parameter=name, estimate=x0_coord,
        ci_low=ci_lo, ci_high=ci_hi, ci_width=ci_hi - ci_lo,
        lower_bound=lo_coord, upper_bound=hi_coord,
        ci_low_finite=L.crossed, ci_high_finite=R.crossed,
        bound_active=(L.active || R.active),
        identifiability=ident,
        steps_evaluated=L.n_steps + R.n_steps,
    )
    (summary_row, curve)
end

function profile_likelihood_uq(
    loss_fn, x_hat::AbstractVector{<:Real}, lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real},
    names::AbstractVector{<:AbstractString}, loss_ratio_crit::Real;
    derived_profiles=NamedTuple[],
    max_steps::Int=14,
    seconds_per_step::Real=4.0,
    maxiter::Int=250,
    coordinate_indices=eachindex(x_hat),
)
    x0 = Float64.(x_hat)
    lo = Float64.(lower)
    hi = Float64.(upper)
    base_loss = Float64(loss_fn(x0))
    loss_crit = base_loss * Float64(loss_ratio_crit)
    delta_loss = max(loss_crit - base_loss, 1e-12)

    # Diagonal curvature estimate via AD gradient finite differences
    g0 = ForwardDiff.gradient(loss_fn, x0)
    H_diag = ones(length(x0))
    for j in coordinate_indices
        span = hi[j] - lo[j]
        h = max(1e-4 * span, 1e-6)
        xp = copy(x0); xm = copy(x0)
        xp[j] = min(hi[j], x0[j] + h)
        xm[j] = max(lo[j], x0[j] - h)
        dh = xp[j] - xm[j]
        if dh > 1e-9
            gp = ForwardDiff.gradient(loss_fn, xp)[j]
            gm = ForwardDiff.gradient(loss_fn, xm)[j]
            H_diag[j] = max((gp - gm) / dh, 1e-12)
        end
    end

    summary_rows = NamedTuple[]
    curve_rows = NamedTuple[]

    for j in coordinate_indices
        span = hi[j] - lo[j]
        keep = [k for k in eachindex(x0) if k != j]
        lo_k = lo[keep]; hi_k = hi[keep]; u_init_k = x0[keep]
        est_radius = sqrt(2.0 * delta_loss / H_diag[j])
        init_step = clamp(0.35 * est_radius, 1e-5 * span, 0.12 * span)

        make_sub = target -> begin
            reconstruct = u -> begin
                v = similar(u, length(x0))
                @inbounds for (pos, idx) in enumerate(keep)
                    v[idx] = u[pos]
                end
                @inbounds v[j] = target + zero(eltype(u))
                v
            end
            (f=u -> loss_fn(reconstruct(u)), lo=lo_k, hi=hi_k, u_init=u_init_k)
        end

        srow, crow = _walk_profile(String(names[j]), x0[j], lo[j], hi[j], init_step,
                                   base_loss, loss_crit, make_sub, max_steps, seconds_per_step, maxiter)
        push!(summary_rows, srow)
        append!(curve_rows, crow)
    end

    # Derived Arrhenius log10_A profiles with linear logk(E) elimination
    invln10 = 1.0 / log(10.0)
    for dp in derived_profiles
        lj = Int(dp.logk_idx)
        ej = Int(dp.E_idx)
        shift = get(dp, :rate_unit, :s_inv) == :min_inv ? log10(60.0) : 0.0
        c = 1e5 / (GAS_R_UQ * Float64(dp.ref_K)) * invln10
        x0_a = x0[lj] - shift + c * x0[ej]
        lo_a = lo[lj] - shift + c * lo[ej]
        hi_a = hi[lj] - shift + c * hi[ej]
        span_a = hi_a - lo_a
        keep = [k for k in eachindex(x0) if k != lj]
        q_e = only(findall(==(ej), keep))
        lo_k_base = lo[keep]; hi_k_base = hi[keep]; u_init_k = x0[keep]

        # Propagate curvature to log10_A from fixed-E diagonal curvature and expand smoothly along the ridge
        init_step = clamp(0.5 * sqrt(2.0 * delta_loss / H_diag[lj]), 1e-3, 0.12 * span_a)
        make_sub_a = target -> begin
            a_shift = target + shift
            lo_k = copy(lo_k_base); hi_k = copy(hi_k_base)
            lo_k[q_e] = max(lo[ej], (a_shift - hi[lj]) / c)
            hi_k[q_e] = min(hi[ej], (a_shift - lo[lj]) / c)
            lo_k[q_e] + 1e-8 < hi_k[q_e] || return nothing
            u_clamped = clamp.(u_init_k, lo_k .+ 1e-8 .* (hi_k .- lo_k), hi_k .- 1e-8 .* (hi_k .- lo_k))
            reconstruct = u -> begin
                v = similar(u, length(x0))
                @inbounds for (pos, idx) in enumerate(keep)
                    v[idx] = u[pos]
                end
                @inbounds v[lj] = a_shift - c * u[q_e]
                v
            end
            (f=u -> loss_fn(reconstruct(u)), lo=lo_k, hi=hi_k, u_init=u_clamped)
        end

        srow, crow = _walk_profile(String(dp.name), x0_a, lo_a, hi_a, init_step,
                                   base_loss, loss_crit, make_sub_a, max_steps, seconds_per_step, maxiter)
        push!(summary_rows, srow)
        append!(curve_rows, crow)
    end

    (summary=DataFrame(summary_rows), curves=DataFrame(curve_rows))
end

_to_pairs(d::AbstractVector{<:Pair}) = [(String(k), Float64(v)) for (k, v) in d]
_to_pairs(d::AbstractDict) = [(String(k), Float64(d[k])) for k in sort!(collect(keys(d)))]
_to_pairs(nt::NamedTuple) = [(String(k), Float64(v)) for (k, v) in pairs(nt)]

function _summarize_bootstrap(samples::Vector{Vector{Float64}}, predict_curves_fn, derived_fn,
                              x_hat::Vector{Float64}, names::Vector{String}, n_boot::Int; kind_label::String="bootstrap")
    n_conv = length(samples)
    n_conv >= 3 || error("Too few converged bootstrap replicates ($n_conv / $n_boot)")

    mat = reduce(hcat, samples) # d x n_conv
    d0 = _to_pairs(derived_fn(x_hat))
    d_names = [k for (k, _) in d0]
    d_x0 = [v for (_, v) in d0]
    d_mat = zeros(length(d_names), n_conv)
    for b in 1:n_conv
        pairs_b = _to_pairs(derived_fn(samples[b]))
        for (row_idx, (_, val)) in enumerate(pairs_b)
            d_mat[row_idx, b] = val
        end
    end

    summary_rows = NamedTuple[]
    for j in eachindex(names)
        vals = mat[j, :]
        q025, q50, q975, q25, q75 = quantile(vals, [0.025, 0.5, 0.975, 0.25, 0.75])
        push!(summary_rows, (
            parameter=names[j], kind="parameter",
            estimate=x_hat[j], boot_median=q50, boot_mean=mean(vals),
            ci_low=q025, ci_high=q975, ci_width=q975 - q025, iqr=q75 - q25,
            n_converged=n_conv, n_boot=n_boot, method=kind_label,
        ))
    end
    for (j, dname) in enumerate(d_names)
        vals = d_mat[j, :]
        q025, q50, q975, q25, q75 = quantile(vals, [0.025, 0.5, 0.975, 0.25, 0.75])
        push!(summary_rows, (
            parameter=dname, kind="derived",
            estimate=d_x0[j], boot_median=q50, boot_mean=mean(vals),
            ci_low=q025, ci_high=q975, ci_width=q975 - q025, iqr=q75 - q25,
            n_converged=n_conv, n_boot=n_boot, method=kind_label,
        ))
    end

    # Arrhenius E vs log10(A) compensation correlations
    comp_rows = NamedTuple[]
    d_lookup = Dict(n => i for (i, n) in enumerate(d_names))
    for dname in d_names
        m = match(r"^E(\d+)_kJ_mol$", dname)
        m === nothing && continue
        ch = m.captures[1]
        aname = "log10_A$(ch)_s_inv"
        haskey(d_lookup, aname) || continue
        Es = d_mat[d_lookup[dname], :]
        As = d_mat[d_lookup[aname], :]
        var_E = var(Es)
        std_E = std(Es); std_A = std(As)
        r_EA = (std_E > 1e-12 && std_A > 1e-12) ? cor(Es, As) : NaN
        slope = var_E > 1e-12 ? cov(Es, As) / var_E : NaN
        T_iso_K = (isfinite(slope) && slope > 1e-9) ? 1000.0 / (slope * GAS_R_UQ * log(10.0)) : NaN
        push!(comp_rows, (
            channel=parse(Int, ch), E_param=dname, A_param=aname,
            corr_E_log10A=r_EA, slope_log10A_per_kJ=slope, T_iso_K=T_iso_K,
            E_ci_width_kJ_mol=quantile(Es, 0.975) - quantile(Es, 0.025),
            log10A_ci_width=quantile(As, 0.975) - quantile(As, 0.025),
            method=kind_label, n_converged=n_conv,
        ))
    end

    # Pointwise predictive envelopes across all runs
    base_preds = predict_curves_fn(x_hat)
    n_runs = length(base_preds)
    curve_mats = [zeros(length(base_preds[r]), n_conv) for r in 1:n_runs]
    for b in 1:n_conv
        pb = predict_curves_fn(samples[b])
        for r in 1:n_runs
            @inbounds curve_mats[r][:, b] .= pb[r]
        end
    end

    band_rows = NamedTuple[]
    for r in 1:n_runs
        M = curve_mats[r]
        bcurve = base_preds[r]
        for k in eachindex(bcurve)
            vals = @view M[k, :]
            p025, p500, p975 = quantile(vals, [0.025, 0.5, 0.975])
            push!(band_rows, (
                run_idx=r, point_idx=k, baseline=Float64(bcurve[k]),
                p025=p025, p500=p500, p975=p975, width_pp=100.0 * (p975 - p025),
            ))
        end
    end

    (summary=DataFrame(summary_rows),
     compensation=DataFrame(comp_rows),
     bands=DataFrame(band_rows),
     replicates=mat)
end

function bayesian_curve_bootstrap(
    weighted_loss_fn, predict_curves_fn, derived_fn,
    x_hat::AbstractVector{<:Real}, lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real},
    names::AbstractVector{<:AbstractString}, n_runs::Integer;
    n_boot::Int=40, seed::Int=20261119, seconds_per_rep::Real=6.0, maxiter::Int=300,
)
    rng = StableRNG(seed)
    x0 = Float64.(x_hat); lo = Float64.(lower); hi = Float64.(upper)
    x_init = clamp.(x0, lo .+ 1e-8 .* (hi .- lo), hi .- 1e-8 .* (hi .- lo))
    samples = Vector{Float64}[]
    statuses = String[]

    for b in 1:n_boot
        # Rubin Bayesian bootstrap weights ~ Dirichlet(1, ..., 1)
        w = -log.(max.(rand(rng, n_runs), 1e-15))
        w ./= sum(w)
        f = z -> weighted_loss_fn(z, w)
        cfg = ForwardDiff.GradientConfig(f, x_init)
        res = ForwardDiff.DiffResults.GradientResult(copy(x_init))
        obj = z -> begin
            ForwardDiff.gradient!(res, f, z, cfg)
            (ForwardDiff.DiffResults.value(res), copy(ForwardDiff.DiffResults.gradient(res)))
        end
        try
            raw = Fides.solve(
                Fides.FidesProblem(obj, x_init; lb=lo, ub=hi),
                Fides.BFGS();
                options=Fides.FidesOptions(maxiter=maxiter, maxtime=Float64(seconds_per_rep),
                    fatol=1e-12, frtol=1e-9, gatol=1e-7, verbose="error"),
            )
            xc = clamp.(collect(Float64, raw.xmin), lo, hi)
            v, g = obj(xc)
            pg = _proj_grad(xc, g, lo, hi)
            if isfinite(v) && all(isfinite, xc) && pg < 1e-4
                push!(samples, xc)
                push!(statuses, string(raw.retcode))
            else
                push!(statuses, "nonstationary_pg_$(round(pg; sigdigits=2))")
            end
        catch err
            err isa InterruptException && rethrow()
            push!(statuses, "failed")
        end
    end

    out = _summarize_bootstrap(samples, predict_curves_fn, derived_fn, x0, String.(names), n_boot;
                               kind_label="bayesian_curve_bootstrap")
    merge(out, (statuses=statuses,))
end

function moving_block_bootstrap(
    fit_data_fn, observed_masses::AbstractVector, fitted_preds::AbstractVector,
    predict_curves_fn, derived_fn,
    x_hat::AbstractVector{<:Real}, names::AbstractVector{<:AbstractString};
    n_boot::Int=40, seed::Int=20261120,
)
    rng = StableRNG(seed)
    x0 = Float64.(x_hat)
    n_runs = length(observed_masses)
    length(fitted_preds) == n_runs || error("Observed and predicted run count mismatch")

    centered_res = Vector{Vector{Float64}}(undef, n_runs)
    block_lens = Vector{Int}(undef, n_runs)
    for r in 1:n_runs
        e = Float64.(observed_masses[r]) .- Float64.(fitted_preds[r])
        rho = lag1_autocorr(e)
        centered_res[r] = e .- mean(e)
        raw_b = round(Int, (1.0 + rho) / max(1.0 - rho, 0.05))
        block_lens[r] = clamp(raw_b, 5, max(5, div(length(e), 4)))
    end

    samples = Vector{Float64}[]
    statuses = String[]
    for _ in 1:n_boot
        synth_masses = Vector{Vector{Float64}}(undef, n_runs)
        for r in 1:n_runs
            e_r = centered_res[r]
            pred_r = fitted_preds[r]
            n_pts = length(e_r)
            blen = block_lens[r]
            boot_e = Vector{Float64}(undef, n_pts)
            pos = 1
            while pos <= n_pts
                start_idx = rand(rng, 1:n_pts)
                for off in 0:(blen - 1)
                    pos > n_pts && break
                    @inbounds boot_e[pos] = e_r[mod1(start_idx + off, n_pts)]
                    pos += 1
                end
            end
            synth_masses[r] = pred_r .+ boot_e
        end
        try
            res = fit_data_fn(synth_masses, x0)
            xc = Float64.(res.x)
            ok = all(isfinite, xc) && (hasproperty(res, :pg) ? res.pg < 1e-4 : Bool(res.stationary))
            if ok
                push!(samples, xc)
                push!(statuses, "converged")
            else
                push!(statuses, "nonstationary")
            end
        catch err
            err isa InterruptException && rethrow()
            push!(statuses, "failed")
        end
    end

    out = _summarize_bootstrap(samples, predict_curves_fn, derived_fn, x0, String.(names), n_boot;
                               kind_label="moving_block_bootstrap")
    merge(out, (block_lengths=block_lens, statuses=statuses))
end

"""
Generate stationary zero-mean AR(1) residuals with exact marginal std `sigma`
and lag-1 autocorrelation `rho`.
"""
function generate_ar1_noise(n::Integer, rho::Real, sigma::Real; rng=StableRNG(1))
    0.0 <= rho < 0.999 || error("rho must lie in [0, 0.999)")
    sigma >= 0.0 || error("sigma must be nonnegative")
    n <= 0 && return Float64[]
    e = Vector{Float64}(undef, n)
    inn_sd = sigma * sqrt(1.0 - rho * rho)
    e[1] = sigma * randn(rng)
    @inbounds for i in 2:n
        e[i] = rho * e[i - 1] + inn_sd * randn(rng)
    end
    e
end

end
