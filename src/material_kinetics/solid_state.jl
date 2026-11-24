using StableRNGs: StableRNG

const SB_EPS = 1e-3

function rpm_survival(H, psi)
    h = max(H, zero(H))
    exp(-h - 0.25 * psi * h^2)
end

function rpm_advance(q, dH, psi)
    resin_primal(q) <= 1e-16 && return zero(q + dH + psi)
    h = max(dH, zero(dH))
    lnq = min(log(q), zero(q))
    q * exp(-h * sqrt(1 - psi * lnq) - 0.25 * psi * h^2)
end

function ae_survival(H, m)
    resin_primal(H) <= 1e-30 && return one(H + m)
    exp(-exp(m * log(H)))
end

function sb_rate(q, n, m; eps_nuc=SB_EPS)
    resin_primal(q) <= 1e-14 && return zero(q + n + m)
    qc = min(q, one(q))
    exp(n * log(qc) + m * log(1 - qc + eps_nuc))
end

function sb_survival_grid(H_vec, save_indices, n, m; eps_nuc=SB_EPS, substeps=2)
    T = promote_type(eltype(H_vec), typeof(n), typeof(m))
    values = Vector{T}(undef, length(H_vec))
    q = one(T)
    values[1] = q
    for j in 2:length(H_vec)
        dh = (H_vec[j] - H_vec[j-1]) / substeps
        if resin_primal(dh) > 0 && resin_primal(q) > 1e-14
            for _ in 1:substeps
                k1 = -sb_rate(q, n, m; eps_nuc)
                k2 = -sb_rate(q + 0.5 * dh * k1, n, m; eps_nuc)
                k3 = -sb_rate(q + 0.5 * dh * k2, n, m; eps_nuc)
                k4 = -sb_rate(q + dh * k3, n, m; eps_nuc)
                q = max(q + (dh / 6) * (k1 + 2 * k2 + 2 * k3 + k4), zero(T))
            end
        end
        values[j] = q
    end
    values[save_indices]
end

function latin_hypercube_starts(lower, upper, n_starts; seed=20261118, margin=0.02)
    lo = Float64.(lower)
    hi = Float64.(upper)
    n_starts >= 1 || error("n_starts must be at least 1")
    !isempty(lo) && length(lo) == length(hi) && all(isfinite, lo) && all(isfinite, hi) && all(lo .< hi) ||
        error("Finite, strictly ordered bounds required for Latin Hypercube starts")
    0.0 <= margin < 0.5 || error("margin must lie in [0, 0.5)")
    rng = StableRNG(seed)
    d = length(lo)
    starts = [Vector{Float64}(undef, d) for _ in 1:n_starts]
    for j in 1:d
        perm = sortperm(rand(rng, n_starts))
        span = hi[j] - lo[j]
        for s in 1:n_starts
            u = (perm[s] - rand(rng)) / n_starts
            u_in = margin + (1.0 - 2.0 * margin) * u
            starts[s][j] = clamp(lo[j] + u_in * span, lo[j] + 1e-8 * span, hi[j] - 1e-8 * span)
        end
    end
    starts
end
