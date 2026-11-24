using LinearAlgebra

function lag1_autocorr(e::AbstractVector)
    length(e) < 3 && return 0.0
    ec = Float64.(e) .- mean(e)
    a = @view ec[1:end-1]
    b = @view ec[2:end]
    denom = sqrt(sum(abs2, a) * sum(abs2, b))
    denom <= 1e-30 && return 0.0
    clamp(sum(a .* b) / denom, 0.0, 0.99)
end

function information_criteria(residuals_by_run::AbstractVector, k_params::Integer)
    isempty(residuals_by_run) && error("Empty residuals")
    n_runs = length(residuals_by_run)
    n_obs = sum(length, residuals_by_run)
    balanced_mse = mean(mean(abs2, e) for e in residuals_by_run)
    rmse_pp = 100.0 * sqrt(balanced_mse)
    rhos = [lag1_autocorr(e) for e in residuals_by_run]
    n_eff = sum(length(e) * (1.0 - rho) / (1.0 + rho) for (e, rho) in zip(residuals_by_run, rhos))
    log_sigma2 = log(max(balanced_mse, 1e-30))
    aic_nominal = n_obs * log_sigma2 + 2.0 * k_params
    bic_nominal = n_obs * log_sigma2 + k_params * log(max(n_obs, 2))
    aic_eff = n_eff * log_sigma2 + 2.0 * k_params
    aicc_eff = aic_eff + 2.0 * k_params * (k_params + 1.0) / max(n_eff - k_params - 1.0, 1.0)
    bic_eff = n_eff * log_sigma2 + k_params * log(max(n_eff, 2.0))
    (; n_runs, n_obs, n_eff, k_params=Int(k_params), rmse_pp,
       median_lag1_autocorr=median(rhos), aic_nominal, bic_nominal,
       aic_eff, aicc_eff, bic_eff)
end

function scaled_jacobian_diagnostics(predict_flat_fn, x, lower, upper)
    J = ForwardDiff.jacobian(predict_flat_fn, Float64.(x)) * Diagonal(Float64.(upper .- lower))
    sv = svdvals(J)
    isempty(sv) && return (condition_number=NaN, effective_rank_1e6=0)
    (condition_number=first(sv) / max(last(sv), 1e-30),
     effective_rank_1e6=count(s -> s > 1e-6 * first(sv), sv))
end
