module CFRPSolidState

using CSV, DataFrames, Statistics, LinearAlgebra, ForwardDiff, Printf, TOML, SHA, Serialization
using Catalyst, ModelingToolkit, OrdinaryDiffEq, PEtab, Fides
using StableRNGs: StableRNG
import Evolutionary, Symbolics

const GAS_R = 8.31446261815324
primal(x) = x
primal(x::ForwardDiff.Dual) = ForwardDiff.value(x)
resin_primal(x) = primal(x)

include("../common.jl")
include("../material_kinetics/solid_state.jl")
include("../material_kinetics/model_selection.jl")
include("../material_kinetics/genetic_search.jl")

function nth_survival(H, n)
    d = n - 1
    1 + d * H <= 0 && return zero(H + n)
    if abs(primal(d * H)) < 1e-5
        return exp(-H + d * H^2 / 2 - d^2 * H^3 / 3 + d^3 * H^4 / 4)
    end
    exp(-log1p(d * H) / d)
end

function nth_advance(q, H, n)
    primal(q) <= 1e-14 && return zero(q + H + n)
    q * nth_survival(H * exp((n - 1) * log(q)), n)
end

function load_cfrp_runs(audit; verify_sources=false)
    metadata = CSV.read(joinpath(audit, "measurement_quality.csv"), DataFrame)
    runs = NamedTuple[]
    for row in eachrow(metadata)
        if verify_sources && isfile(row.source)
            filehash(row.source) == row.source_sha256 || error("Original source changed: $(row.condition)")
        end
        path = joinpath(audit, row.condition * "_observations.csv")
        obs = CSV.read(path, DataFrame)
        all(obs.condition .== row.condition) && all(obs.gas .== row.gas) && all(obs.rate .== row.rate_C_min) ||
            error("Audit labels differ for $(row.condition)")
        all(isfinite, obs.temperature_K) && minimum(obs.temperature_K) > 0 && all(isfinite, obs.mass_fraction) ||
            error("Invalid observations in $(row.condition)")
        t = 60.0 .* (obs.time_min .- first(obs.time_min))
        all(diff(t) .> 0) || error("Nonchronological audit in $(row.condition)")
        indices = [1]
        lastT = first(obs.temperature_K)
        for j in 2:nrow(obs)-1
            if abs(obs.temperature_K[j] - lastT) >= 3.0
                push!(indices, j)
                lastT = obs.temperature_K[j]
            end
        end
        push!(indices, nrow(obs))
        push!(runs, (
            id=String(row.condition),
            gas=String(row.gas),
            air=row.gas == "Air" ? 1 : 0,
            rate=Float64(row.rate_C_min),
            time=t,
            temperature=Vector{Float64}(obs.temperature_K),
            observation_indices=indices,
            obstime=t[indices],
            T=Vector{Float64}(obs.temperature_K[indices]),
            mass=Vector{Float64}(obs.mass_fraction[indices]),
            source=String(row.source),
            source_sha256=String(row.source_sha256),
            cache_path=path,
            cache_sha256=filehash(path),
            grids=Dict{Float64,Any}(),
        ))
    end
    length(runs) == 10 && Set(r.rate for r in runs) == Set([2.5, 5.0, 10.0, 20.0]) ||
        error("Expected the ten audited CFRP-only curves")
    runs
end

function load_biasi_as_cfrp_runs(path::AbstractString=joinpath(@__DIR__, "..", "..", "data", "biasi_t700m21", "experimental_5K.csv");
                                  spacing::Real=5.0, Tstart::Real=300.0)
    isfile(path) || error("Biasi dataset not found: $path")
    data = CSV.read(path, DataFrame)
    rates = sort(unique(Float64.(data.rate)))
    isempty(rates) && error("No heating rates found in $path")
    Tmin = Float64(spacing) * ceil(maximum(minimum(g.temperature_K) for g in groupby(data, :rate)) / Float64(spacing))
    Tmax = Float64(spacing) * floor(minimum(maximum(g.temperature_K) for g in groupby(data, :rate)) / Float64(spacing))
    Tstart < Tmin < Tmax || error("Invalid Biasi temperature window: Tstart=$Tstart, Tmin=$Tmin, Tmax=$Tmax")
    forcing = collect(Float64(Tstart):0.5:Tmax)
    obsT = collect(Tmin:Float64(spacing):Tmax)
    indices = [something(findfirst(T_val -> isapprox(T_val, target; atol=1e-10), forcing)) for target in obsT]

    runs = NamedTuple[]
    sha = filehash(path)
    for rate in rates
        g = sort(filter(r -> Float64(r.rate) == rate, data), :temperature_K)
        mass = map(obsT) do target
            j = clamp(searchsortedlast(g.temperature_K, target), 1, nrow(g) - 1)
            z = (target - g.temperature_K[j]) / (g.temperature_K[j + 1] - g.temperature_K[j])
            Float64((1.0 - z) * g.mass_fraction[j] + z * g.mass_fraction[j + 1])
        end
        t = 60.0 .* (forcing .- Float64(Tstart)) ./ rate
        all(diff(t) .> 0) || error("Nonchronological ramp in Biasi run at $rate K/min")
        push!(runs, (
            id="biasi_$(Int(round(rate)))",
            gas="Air",
            air=1,
            rate=Float64(rate),
            time=t,
            temperature=copy(forcing),
            observation_indices=copy(indices),
            obstime=t[indices],
            T=Vector{Float64}(forcing[indices]),
            mass=Vector{Float64}(mass),
            source=abspath(path),
            source_sha256=sha,
            grids=Dict{Float64,Any}(),
        ))
    end
    runs
end

function cfrp_model_spec(kind::Symbol; order_bounds=(0.2, 4.0), psi_bounds=(0.0, 35.0), fixed_fiber::Union{Nothing,Real}=nothing)
    kind in (:M1_1Mat_1stOrder, :M2_1Mat_GeomOx, :M2b_2Mat1st_GeomOx, :M3_1Mat_RPMOx, :M4_2Mat_RPMOx, :M5_2Mat_FreeOrder_Cold) ||
        error("Unknown CFRP solid-state model kind: $kind")
    if fixed_fiber !== nothing
        ff = Float64(fixed_fiber)
        isfinite(ff) && 0.0 < ff < 1.0 || error("fixed_fiber must lie in (0, 1), got $fixed_fiber")
    end

    names = String[]
    lo = Float64[]
    hi = Float64[]

    if kind == :M1_1Mat_1stOrder
        nresin = 1
        refs = [630.0, 850.0, 1050.0]
        labels = ["Matrix decomposition (n=1)", "Char oxidation (n=1)", "Carbon-fiber oxidation (n=1)"]
        laws = [:ro, :ro, :ro]
        fixed_shapes = Dict(1 => 1.0, 2 => 1.0, 3 => 1.0)
        free_shape_names = Dict{Int,String}()
        logk_hi = 1.5
    elseif kind == :M2_1Mat_GeomOx
        nresin = 1
        refs = [630.0, 850.0, 1050.0]
        labels = ["Matrix decomposition (free n)", "Char oxidation (R3, n=2/3)", "Carbon-fiber oxidation (R2, n=1/2)"]
        laws = [:ro, :ro, :ro]
        fixed_shapes = Dict(2 => 2.0 / 3.0, 3 => 0.5)
        free_shape_names = Dict(1 => "n1")
        logk_hi = 1.5
    elseif kind == :M2b_2Mat1st_GeomOx
        nresin = 2
        refs = [585.0, 660.0, 850.0, 1050.0]
        labels = [
            "Matrix population 1 decomposition (n=1)",
            "Matrix population 2 decomposition (n=1)",
            "Char oxidation (R3, n=2/3)",
            "Carbon-fiber oxidation (R2, n=1/2)",
        ]
        laws = [:ro, :ro, :ro, :ro]
        fixed_shapes = Dict(1 => 1.0, 2 => 1.0, 3 => 2.0 / 3.0, 4 => 0.5)
        free_shape_names = Dict{Int,String}()
        logk_hi = 1.5
    elseif kind == :M3_1Mat_RPMOx
        nresin = 1
        refs = [630.0, 850.0, 1050.0]
        labels = ["Matrix decomposition (free n)", "Char oxidation (Random Pore)", "Carbon-fiber oxidation (Random Pore)"]
        laws = [:ro, :rpm, :rpm]
        fixed_shapes = Dict{Int,Float64}()
        free_shape_names = Dict(1 => "n1", 2 => "psi2", 3 => "psi3")
        logk_hi = 1.5
    elseif kind == :M4_2Mat_RPMOx
        nresin = 2
        refs = [585.0, 660.0, 850.0, 1050.0]
        labels = [
            "Matrix population 1 decomposition (n=1)",
            "Matrix population 2 decomposition (n=1)",
            "Char oxidation (Random Pore)",
            "Carbon-fiber oxidation (Random Pore)",
        ]
        laws = [:ro, :ro, :rpm, :rpm]
        fixed_shapes = Dict(1 => 1.0, 2 => 1.0)
        free_shape_names = Dict(3 => "psi3", 4 => "psi4")
        logk_hi = 1.5
    else
        nresin = 2
        refs = [585.0, 660.0, 850.0, 1050.0]
        labels = [
            "Matrix population 1 decomposition (free n)",
            "Matrix population 2 decomposition (free n)",
            "Char oxidation (free n)",
            "Carbon-fiber oxidation (free n)",
        ]
        laws = [:ro, :ro, :ro, :ro]
        fixed_shapes = Dict{Int,Float64}()
        free_shape_names = Dict(1 => "n1", 2 => "n2", 3 => "n3", 4 => "n4")
        logk_hi = 0.0
    end

    for j in eachindex(refs)
        append!(names, ["logk$j", "E$j"])
        append!(lo, [-7.0, 0.3])
        append!(hi, [logk_hi, 4.5])
        if haskey(free_shape_names, j)
            sname = free_shape_names[j]
            push!(names, sname)
            if laws[j] == :rpm
                push!(lo, psi_bounds[1])
                push!(hi, psi_bounds[2])
            else
                push!(lo, order_bounds[1])
                push!(hi, order_bounds[2])
            end
        end
    end

    if fixed_fiber === nothing
        push!(names, "fiber_fraction"); push!(lo, 0.1); push!(hi, 0.9)
    end
    if nresin == 2
        push!(names, "matrix_split"); push!(lo, 0.05); push!(hi, 0.95)
    end
    for j in 1:nresin
        push!(names, "char_yield$j"); push!(lo, 0.0); push!(hi, 0.8)
    end

    nch = length(refs)
    name_to_idx = Dict(n => i for (i, n) in enumerate(names))
    logk_idx = Tuple(name_to_idx["logk$j"] for j in 1:nch)
    E_idx = Tuple(name_to_idx["E$j"] for j in 1:nch)
    shape_idx = Tuple(haskey(free_shape_names, j) ? name_to_idx[free_shape_names[j]] : 0 for j in 1:nch)
    fixed_shape_vals = Tuple(Float64(get(fixed_shapes, j, 0.0)) for j in 1:nch)
    is_rpm = Tuple(laws[j] == :rpm for j in 1:nch)
    inv_refs = Tuple(1.0 / refs[j] for j in 1:nch)
    fiber_idx = fixed_fiber === nothing ? name_to_idx["fiber_fraction"] : 0
    fixed_fiber_val = fixed_fiber === nothing ? NaN : Float64(fixed_fiber)
    split_idx = nresin == 2 ? name_to_idx["matrix_split"] : 0
    yield_idx = nresin == 1 ? (name_to_idx["char_yield1"], 0) : (name_to_idx["char_yield1"], name_to_idx["char_yield2"])

    (; kind, nresin, labels, refs, laws, fixed_shapes, free_shape_names, names, lower=lo, upper=hi,
       logk_idx, E_idx, shape_idx, fixed_shape_vals, is_rpm, inv_refs, fiber_idx, fixed_fiber_val, split_idx, yield_idx)
end

function unpack(spec, x)
    T_el = eltype(x)
    channels = [begin
        shape = spec.shape_idx[j] > 0 ? x[spec.shape_idx[j]] : convert(T_el, spec.fixed_shape_vals[j])
        (logk=x[spec.logk_idx[j]], E=x[spec.E_idx[j]], law=spec.laws[j], shape=shape, ref=spec.refs[j])
    end for j in eachindex(spec.refs)]
    fiber = spec.fiber_idx > 0 ? x[spec.fiber_idx] : convert(T_el, spec.fixed_fiber_val)
    split = spec.nresin == 2 ? x[spec.split_idx] : one(fiber)
    weights = spec.nresin == 1 ? [1 - fiber] : [(1 - fiber) * split, (1 - fiber) * (1 - split)]
    yields = [x[spec.yield_idx[j]] for j in 1:spec.nresin]
    (; channels, fiber, weights, yields)
end

const LOG10_CONST = log(10.0)
const E_OVER_R = 1e5 / GAS_R

rate_s(logk, E, T, ref) = exp(LOG10_CONST * logk - E * E_OVER_R * (1 / T - 1 / ref))

@inline survive_law(H, is_rpm::Bool, shape) = is_rpm ? rpm_survival(H, shape) : nth_survival(H, shape)
@inline advance_law(q_norm, dH, is_rpm::Bool, shape) = is_rpm ? rpm_advance(q_norm, dH, shape) : nth_advance(q_norm, dH, shape)

@inline function simpson_exp(dt6, i0, im, i1, ln10_logk, Ea_R, inv_ref)
    r0 = exp(ln10_logk - Ea_R * (i0 - inv_ref))
    rm = exp(ln10_logk - Ea_R * (im - inv_ref))
    r1 = exp(ln10_logk - Ea_R * (i1 - inv_ref))
    dt6 * (r0 + 4 * rm + r1)
end

@inline function channel_params(spec, x, j::Int, ::Type{T_el}) where {T_el}
    ln10_logk = LOG10_CONST * x[spec.logk_idx[j]]
    Ea_R = E_OVER_R * x[spec.E_idx[j]]
    shape = spec.shape_idx[j] > 0 ? x[spec.shape_idx[j]] : convert(T_el, spec.fixed_shape_vals[j])
    (ln10_logk, Ea_R, spec.is_rpm[j], shape, spec.inv_refs[j])
end

function make_grid(r, step_C)
    step_C > 0 || error("Positive integration step required")
    obs_set = Set(r.observation_indices)
    t = [0.0]; T = [first(r.temperature)]
    saves = 1 in obs_set ? Int[1] : Int[]
    for j in 2:length(r.time)
        subdivisions = max(1, ceil(Int, abs(r.temperature[j] - r.temperature[j-1]) / step_C))
        for k in 1:subdivisions
            a = k / subdivisions
            push!(t, (1 - a) * r.time[j-1] + a * r.time[j])
            push!(T, (1 - a) * r.temperature[j-1] + a * r.temperature[j])
        end
        j in obs_set && push!(saves, length(t))
    end
    nsteps = length(t) - 1
    dt6 = Vector{Float64}(undef, nsteps)
    invT0 = Vector{Float64}(undef, nsteps)
    invTm = Vector{Float64}(undef, nsteps)
    invT1 = Vector{Float64}(undef, nsteps)
    @inbounds for j in 1:nsteps
        dt6[j] = (t[j + 1] - t[j]) / 6.0
        T0 = T[j]
        T1 = T[j + 1]
        invT0[j] = 1.0 / T0
        invTm[j] = 2.0 / (T0 + T1)
        invT1[j] = 1.0 / T1
    end
    (time=t, temperature=T, saves=saves, dt6=dt6, invT0=invT0, invTm=invTm, invT1=invT1)
end

function _integrate_cfrp(spec, x::AbstractVector{T_el}, r, grid, physical, savedstates) where {T_el}
    saves = grid.saves
    dt6 = grid.dt6
    invT0 = grid.invT0
    invTm = grid.invTm
    invT1 = grid.invT1
    m = length(grid.time)
    nsaves = length(saves)
    N = spec.nresin
    is_two = (N == 2)
    is_air = (r.air == 1)

    lk1, e1, rpm1, s1, ir1 = channel_params(spec, x, 1, T_el)
    lk2, e2, rpm2, s2, ir2 = is_two ? channel_params(spec, x, 2, T_el) : (zero(T_el), zero(T_el), false, zero(T_el), 0.0)
    lkc, ec, rpmc, sc, irc = channel_params(spec, x, N + 1, T_el)
    lkf, ef, rpmf, sf, irf = channel_params(spec, x, N + 2, T_el)

    is_neat = hasproperty(r, :neat_resin) && r.neat_resin
    fiber = is_neat ? zero(T_el) : (spec.fiber_idx > 0 ? x[spec.fiber_idx] : convert(T_el, spec.fixed_fiber_val))
    matrix = 1 - fiber
    split = is_two ? x[spec.split_idx] : one(T_el)
    w1 = is_two ? matrix * split : matrix
    w2 = is_two ? matrix * (1 - split) : zero(T_el)
    y1 = x[spec.yield_idx[1]]
    y2 = is_two ? x[spec.yield_idx[2]] : zero(T_el)

    q1 = one(T_el)
    q2 = one(T_el)
    H1 = zero(T_el)
    H2 = zero(T_el)
    F = one(T_el)
    HF = zero(T_el)
    C = zero(T_el)

    r_mass = r.mass
    sse = zero(T_el)
    saveid = 1
    if nsaves >= 1 && saves[1] == 1
        if physical === nothing
            @inbounds sse = abs2(one(T_el) - r_mass[1])
        else
            @inbounds physical[1] = one(T_el)
        end
        if savedstates !== nothing
            @inbounds begin
                savedstates[1, 1] = w1
                if is_two
                    savedstates[1, 2] = w2
                end
                savedstates[1, N + 1] = zero(T_el)
                savedstates[1, N + 2] = fiber
                savedstates[1, N + 3] = zero(T_el)
            end
        end
        saveid = 2
    end
    @inbounds for j in 2:m
        step = j - 1
        dt6_s = dt6[step]
        i0 = invT0[step]
        im = invTm[step]
        i1 = invT1[step]

        e_char = zero(T_el)
        if is_air
            e_char = simpson_exp(dt6_s, i0, im, i1, lkc, ec, irc)
            C = matrix * advance_law(C / matrix, e_char / 2, rpmc, sc)
            HF += simpson_exp(dt6_s, i0, im, i1, lkf, ef, irf)
            F = survive_law(HF, rpmf, sf)
        end

        H1 += simpson_exp(dt6_s, i0, im, i1, lk1, e1, ir1)
        newq1 = survive_law(H1, rpm1, s1)
        C += y1 * w1 * (q1 - newq1)
        q1 = newq1

        if is_two
            H2 += simpson_exp(dt6_s, i0, im, i1, lk2, e2, ir2)
            newq2 = survive_law(H2, rpm2, s2)
            C += y2 * w2 * (q2 - newq2)
            q2 = newq2
        end

        if is_air
            C = matrix * advance_law(C / matrix, e_char / 2, rpmc, sc)
        end

        if saveid <= nsaves && j == saves[saveid]
            mass = is_two ? (w1 * q1 + w2 * q2 + C + fiber * F) : (w1 * q1 + C + fiber * F)
            if physical === nothing
                sse += abs2(mass - r_mass[saveid])
            else
                physical[saveid] = mass
            end
            if savedstates !== nothing
                savedstates[saveid, 1] = w1 * q1
                if is_two
                    savedstates[saveid, 2] = w2 * q2
                end
                savedstates[saveid, N + 1] = C
                savedstates[saveid, N + 2] = fiber * F
                savedstates[saveid, N + 3] = 1 - mass
            end
            saveid += 1
        end
    end
    saveid == nsaves + 1 || error("Observation/grid mismatch")
    physical === nothing ? sse / (2 * nsaves) : zero(T_el)
end

function predict_cfrp(spec, x, r; step_C=0.25, states=false)
    grid = get!(r.grids, Float64(step_C)) do
        make_grid(r, step_C)
    end
    T_el = eltype(x)
    physical = Vector{T_el}(undef, length(grid.saves))
    savedstates = states ? zeros(T_el, length(grid.saves), spec.nresin + 3) : nothing
    _integrate_cfrp(spec, x, r, grid, physical, savedstates)
    states ? (mass=physical, states=savedstates) : physical
end

function loss_cfrp(spec, x, runs; step_C=0.25)
    total = zero(eltype(x))
    for r in runs
        grid = get!(r.grids, Float64(step_C)) do
            make_grid(r, step_C)
        end
        total += _integrate_cfrp(spec, x, r, grid, nothing, nothing)
    end
    total / length(runs)
end

function loss_cfrp_weighted(spec, x, runs, weights; step_C=0.25)
    length(runs) == length(weights) || error("Run/weight length mismatch")
    wsum = sum(weights)
    wsum > 0 || error("Positive weight sum required")
    total = zero(eltype(x))
    for (r, w) in zip(runs, weights)
        w == 0 && continue
        grid = get!(r.grids, Float64(step_C)) do
            make_grid(r, step_C)
        end
        total += w * _integrate_cfrp(spec, x, r, grid, nothing, nothing)
    end
    total / wsum
end

projected_grad(spec, x, g) = maximum(abs.(x .- clamp.(x .- g, spec.lower, spec.upper)))

char_prior_penalty(spec, x; mu_E=1.60, sigma_E=0.35, prior_weight=0.0) =
    prior_weight * abs2((x[spec.E_idx[spec.nresin + 1]] - mu_E) / sigma_E)

function loss_cfrp_map(spec, x, runs; step_C=0.25, mu_E=1.60, sigma_E=0.35, prior_weight=0.0)
    loss_cfrp(spec, x, runs; step_C=step_C) +
        char_prior_penalty(spec, x; mu_E=mu_E, sigma_E=sigma_E, prior_weight=prior_weight)
end

function loss_cfrp_map_weighted(spec, x, runs, weights; step_C=0.25, mu_E=1.60, sigma_E=0.35, prior_weight=0.0)
    loss_cfrp_weighted(spec, x, runs, weights; step_C=step_C) +
        char_prior_penalty(spec, x; mu_E=mu_E, sigma_E=sigma_E, prior_weight=prior_weight)
end

function _fit_cfrp_core(spec, runs, make_loss; seed=20261118, ga_pop=24, ga_evals=360, lhs_starts=3, refine_candidates=2, seconds_per_start=10.0, max_iterations=350, fit_step_C=1.0, report_step_C=0.25)
    fit_started = time()
    lhs = latin_hypercube_starts(spec.lower, spec.upper, lhs_starts; seed=seed)
    coarse_loss = make_loss(fit_step_C)
    fine_loss = make_loss(report_step_C)
    println("    [", spec.kind, "] GA (pop=", ga_pop, ", evals=", ga_evals, ", runs=", length(runs), ")..."); flush(stdout)
    ga_started = time()
    ga = bounded_genetic_search(
        coarse_loss,
        spec.lower,
        spec.upper;
        options=GeneticOptions(seed=seed + 1, population=ga_pop, max_evaluations=ga_evals, refine_candidates=refine_candidates),
        starts=lhs[1:min(lhs_starts, ga_pop)],
        refine_objective=fine_loss,
    )
    all_starts = vcat(lhs, ga.starts)
    ga_best_pp = isempty(ga.starts) ? NaN : minimum(x0 -> 100sqrt(2 * loss_cfrp(spec, x0, runs; step_C=report_step_C)), ga.starts)
    println("    [", spec.kind, "] GA done in ", round(time() - ga_started; digits=1), "s (data best ", round(ga_best_pp; digits=2), " pp); running ", length(all_starts), " local BFGS starts..."); flush(stdout)

    coarse_cfg = ForwardDiff.GradientConfig(coarse_loss, spec.lower)
    fine_cfg = ForwardDiff.GradientConfig(fine_loss, spec.lower)
    coarse_res = ForwardDiff.DiffResults.GradientResult(copy(spec.lower))
    fine_res = ForwardDiff.DiffResults.GradientResult(copy(spec.lower))
    coarse_obj = x -> begin
        ForwardDiff.gradient!(coarse_res, coarse_loss, x, coarse_cfg)
        (ForwardDiff.DiffResults.value(coarse_res), copy(ForwardDiff.DiffResults.gradient(coarse_res)))
    end
    fine_obj = x -> begin
        ForwardDiff.gradient!(fine_res, fine_loss, x, fine_cfg)
        (ForwardDiff.DiffResults.value(fine_res), copy(ForwardDiff.DiffResults.gradient(fine_res)))
    end

    attempts = NamedTuple[]
    endpoints = Vector{Float64}[]
    for (i, x0) in enumerate(all_starts)
        label = i <= lhs_starts ? "lhs_$i" : "ga_$(i - lhs_starts)"
        started = time()
        try
            raw = Fides.solve(
                Fides.FidesProblem(coarse_obj, clamp.(x0, spec.lower .+ 1e-7, spec.upper .- 1e-7); lb=spec.lower, ub=spec.upper),
                Fides.BFGS();
                options=Fides.FidesOptions(maxiter=max_iterations, maxtime=Float64(seconds_per_start), fatol=1e-12, frtol=1e-9, gatol=1e-8, verbose="error"),
            )
            xc = collect(raw.xmin)
            v, g = fine_obj(xc)
            dloss = loss_cfrp(spec, xc, runs; step_C=report_step_C)
            pg = projected_grad(spec, xc, g)
            push!(endpoints, xc)
            push!(attempts, (start=label, objective=v, rmse_pp=100sqrt(2 * dloss), projected_gradient=pg, stationary=pg < 1e-6, seconds=time() - started, status=string(raw.retcode), error=""))
            println("      start ", label, ": RMSE=", round(100sqrt(2 * dloss); digits=3), " pp, pg=", @sprintf("%.2e", pg), " (", round(time() - started; digits=1), "s)"); flush(stdout)
        catch err
            err isa InterruptException && rethrow()
            push!(endpoints, copy(x0))
            push!(attempts, (start=label, objective=Inf, rmse_pp=Inf, projected_gradient=Inf, stationary=false, seconds=time() - started, status="failed", error=sprint(showerror, err)))
            println("      start ", label, ": failed (", sprint(showerror, err), ")"); flush(stdout)
        end
    end

    eligible = findall(a -> isfinite(a.objective), attempts)
    isempty(eligible) && error("All cold starts failed for $(spec.kind)")
    best = eligible[argmin([attempts[j].objective for j in eligible])]
    x = copy(endpoints[best])

    for pass in 1:2
        started = time()
        try
            raw = Fides.solve(
                Fides.FidesProblem(fine_obj, clamp.(x, spec.lower .+ 1e-8, spec.upper .- 1e-8); lb=spec.lower, ub=spec.upper),
                Fides.BFGS();
                options=Fides.FidesOptions(maxiter=max(800, max_iterations), maxtime=15.0, fatol=1e-14, frtol=1e-11, gatol=1e-9, verbose="error"),
            )
            cand = collect(raw.xmin)
            v, g = fine_obj(cand)
            dloss = loss_cfrp(spec, cand, runs; step_C=report_step_C)
            pg = projected_grad(spec, cand, g)
            push!(endpoints, cand)
            push!(attempts, (start="polish_$pass", objective=v, rmse_pp=100sqrt(2 * dloss), projected_gradient=pg, stationary=pg < 1e-6, seconds=time() - started, status=string(raw.retcode), error=""))
            if v <= first(fine_obj(x))
                x = cand
            end
            pg < 1e-6 && break
        catch err
            err isa InterruptException && rethrow()
            push!(endpoints, copy(x))
            push!(attempts, (start="polish_$pass", objective=Inf, rmse_pp=Inf, projected_gradient=Inf, stationary=false, seconds=time() - started, status="failed", error=sprint(showerror, err)))
            break
        end
    end

    v, g = fine_obj(x)
    data_loss = loss_cfrp(spec, x, runs; step_C=report_step_C)
    pg = projected_grad(spec, x, g)
    println("    [", spec.kind, "] Finished in ", round(time() - fit_started; digits=1), "s -> RMSE=", round(100sqrt(2 * data_loss); digits=3), " pp, pg=", @sprintf("%.2e", pg)); flush(stdout)
    (; x, objective=v, data_loss, rmse_pp=100sqrt(2 * data_loss), projected_gradient=pg, stationary=pg < 1e-6, attempts, ga_metadata=ga.metadata)
end

function fit_cfrp_cold(spec, runs; seed=20261118, ga_pop=24, ga_evals=360, lhs_starts=3, refine_candidates=2, seconds_per_start=10.0, max_iterations=350, fit_step_C=1.0, report_step_C=0.25)
    _fit_cfrp_core(spec, runs, sc -> (z -> loss_cfrp(spec, z, runs; step_C=sc));
        seed=seed, ga_pop=ga_pop, ga_evals=ga_evals, lhs_starts=lhs_starts,
        refine_candidates=refine_candidates, seconds_per_start=seconds_per_start,
        max_iterations=max_iterations, fit_step_C=fit_step_C, report_step_C=report_step_C)
end

function fit_cfrp_cold_map(spec, runs; mu_E=1.60, sigma_E=0.35, prior_weight=0.0, seed=20261121, ga_pop=24, ga_evals=360, lhs_starts=3, refine_candidates=2, seconds_per_start=10.0, max_iterations=350, fit_step_C=1.0, report_step_C=0.25)
    res = _fit_cfrp_core(spec, runs, sc -> (z -> loss_cfrp_map(spec, z, runs; step_C=sc, mu_E=mu_E, sigma_E=sigma_E, prior_weight=prior_weight));
        seed=seed, ga_pop=ga_pop, ga_evals=ga_evals, lhs_starts=lhs_starts,
        refine_candidates=refine_candidates, seconds_per_start=seconds_per_start,
        max_iterations=max_iterations, fit_step_C=fit_step_C, report_step_C=report_step_C)
    (; x=res.x, map_objective=res.objective, data_loss=res.data_loss, rmse_pp=res.rmse_pp,
       projected_gradient=res.projected_gradient, stationary=res.stationary, attempts=res.attempts, ga_metadata=res.ga_metadata)
end

function load_neat_resin_as_cfrp_runs(study)
    runs = NamedTuple[]
    for r in study.runs
        t = 60.0 .* (Float64.(r.observation_time) .- first(r.observation_time))
        T_K = Float64.(r.observation_temperature)
        m_all = Float64.(r.mass)
        all(diff(t) .> 0) || error("Nonchronological neat resin run $(r.id)")
        indices = [1]
        lastT = first(T_K)
        for j in 2:(length(T_K) - 1)
            if abs(T_K[j] - lastT) >= 3.0
                push!(indices, j)
                lastT = T_K[j]
            end
        end
        length(T_K) > 1 && push!(indices, length(T_K))
        push!(runs, (
            id=String(r.id),
            gas="Air",
            air=1,
            rate=Float64(r.rate),
            neat_resin=true,
            time=t,
            temperature=T_K,
            observation_indices=indices,
            obstime=t[indices],
            T=T_K[indices],
            mass=m_all[indices],
            grids=Dict{Float64,Any}(),
        ))
    end
    runs
end

function verify_native_cfrp(spec, x, runs; step_C=0.25)
    v = loss_cfrp(spec, x, runs; step_C)
    ad = ForwardDiff.gradient(z -> loss_cfrp(spec, z, runs; step_C), x)
    h = 1e-5
    fd = zeros(length(x))
    for j in eachindex(x)
        plus, minus = copy(x), copy(x)
        if x[j] - h >= spec.lower[j] && x[j] + h <= spec.upper[j]
            plus[j] += h; minus[j] -= h
            fd[j] = (loss_cfrp(spec, plus, runs; step_C) - loss_cfrp(spec, minus, runs; step_C)) / (2h)
        elseif x[j] + 2h <= spec.upper[j]
            plus[j] += h; minus[j] += 2h
            fd[j] = (-3v + 4 * loss_cfrp(spec, plus, runs; step_C) - loss_cfrp(spec, minus, runs; step_C)) / (2h)
        else
            plus[j] -= h; minus[j] -= 2h
            fd[j] = (3v - 4 * loss_cfrp(spec, plus, runs; step_C) + loss_cfrp(spec, minus, runs; step_C)) / (2h)
        end
    end
    gradient_gap = maximum(abs.(ad .- fd))
    refinement = 0.0; balance = 0.0; min_state = Inf
    for r in runs
        a = predict_cfrp(spec, x, r; step_C, states=true)
        b = predict_cfrp(spec, x, r; step_C=step_C / 2)
        refinement = max(refinement, maximum(abs.(a.mass .- b)))
        balance = max(balance, maximum(abs.(vec(sum(a.states; dims=2)) .- 1)))
        min_state = min(min_state, minimum(a.states))
    end
    gradient_gap < 2e-5 || error("CFRP AD vs FD mismatch: $gradient_gap")
    refinement < 2e-4 || error("CFRP integration refinement mismatch: $refinement")
    balance < 1e-10 && min_state >= -1e-10 || error("CFRP mass balance/nonnegativity failure")
    (; loss=v, projected_gradient=projected_grad(spec, x, ad), gradient_gap, refinement_gap_pp=100refinement, mass_balance_gap=balance, minimum_state=min_state, passed=true)
end

function cfrp_derived_triplets(spec, x)
    out = Pair{String,Float64}[]
    invln10 = 1.0 / log(10.0)
    for j in eachindex(spec.refs)
        Eval = Float64(x[spec.E_idx[j]])
        lkval = Float64(x[spec.logk_idx[j]])
        push!(out, "E$(j)_kJ_mol" => 100.0 * Eval)
        push!(out, "log10_A$(j)_s_inv" => lkval + (Eval * 1e5) / (GAS_R * spec.refs[j]) * invln10)
    end
    out
end

function cfrp_kinetic_table(spec, x)
    p = unpack(spec, x)
    near = [spec.names[j] for j in eachindex(x) if min(x[j] - spec.lower[j], spec.upper[j] - x[j]) < 1e-4 * (spec.upper[j] - spec.lower[j])]
    DataFrame([begin
        ch = p.channels[j]
        A = 10.0^ch.logk * exp(ch.E * 1e5 / (GAS_R * ch.ref))
        shape_name = ch.law == :rpm ? "psi" : "n"
        shape_fixed = !haskey(spec.free_shape_names, j)
        (model=string(spec.kind), reaction=spec.labels[j], law=string(ch.law),
         A_s_inv=A, E_kJ_mol=100 * ch.E, shape_param=shape_name, shape_value=Float64(ch.shape),
         shape_fixed=shape_fixed, ref_K=ch.ref, char_yield=j <= spec.nresin ? Float64(p.yields[j]) : 0.0,
         near_bounds=join(near, ", "))
    end for j in eachindex(p.channels)])
end

const CFRP_PETAB_HISTORIES = Any[]
function cfrp_recorded_temperature(t, id)
    r = CFRP_PETAB_HISTORIES[round(Int, primal(id))]
    i = clamp(searchsortedlast(r.time, primal(t)), 1, length(r.time) - 1)
    r.temperature[i] + (t - r.time[i]) / (r.time[i+1] - r.time[i]) * (r.temperature[i+1] - r.temperature[i])
end
ModelingToolkit.@register_symbolic cfrp_recorded_temperature(t, id)
function cfrp_recorded_slope(t, id)
    r = CFRP_PETAB_HISTORIES[round(Int, primal(id))]
    i = clamp(searchsortedlast(r.time, primal(t)), 1, length(r.time) - 1)
    (r.temperature[i+1] - r.temperature[i]) / (r.time[i+1] - r.time[i])
end
ModelingToolkit.@register_symbolic cfrp_recorded_slope(t, id)
Symbolics.derivative(::typeof(cfrp_recorded_temperature), args::NTuple{2,Any}, ::Val{1}) = cfrp_recorded_slope(args...)
Symbolics.derivative(::typeof(cfrp_recorded_temperature), args::NTuple{2,Any}, ::Val{2}) = 0

const CFRP_DEPLETION_FLOOR = 1e-8
positive_power(q, n) = max(max(q, 0), CFRP_DEPLETION_FLOOR)^n * max(q, 0) / (max(q, 0) + CFRP_DEPLETION_FLOOR)
rpm_positive_rate(q, psi) = max(q, 0) * sqrt(1 - psi * log(min(max(q, 1e-12), 1.0)))

function verify_cfrp_petab(spec, x, runs; rtol=1e-8)
    firstid = length(CFRP_PETAB_HISTORIES) + 1
    append!(CFRP_PETAB_HISTORIES, runs)
    t = Catalyst.default_t()
    ps = @parameters logk1 E1 s1 logk2 E2 s2 logk3 E3 s3 logk4 E4 s4 fiber_fraction matrix_split char_yield1 char_yield2 history_id air noise_scale
    ss = @species R1(t) R2(t) C(t) F(t) G(t)
    T = cfrp_recorded_temperature(t, history_id)
    rate(a, e, ref) = exp(log(10) * a - e * 1e5 / GAS_R * (1 / T - 1 / ref))
    chan_rate(q, a, e, law, s, ref) = rate(a, e, ref) * (law == :rpm ? rpm_positive_rate(q, s) : positive_power(q, s))

    if spec.nresin == 2
        w1 = (1 - fiber_fraction) * matrix_split
        w2 = (1 - fiber_fraction) * (1 - matrix_split)
        r1 = chan_rate(R1, logk1, E1, spec.laws[1], s1, spec.refs[1])
        r2 = chan_rate(R2, logk2, E2, spec.laws[2], s2, spec.refs[2])
        r3 = air * (1 - fiber_fraction) * chan_rate(C / (1 - fiber_fraction), logk3, E3, spec.laws[3], s3, spec.refs[3])
        r4 = air * chan_rate(F, logk4, E4, spec.laws[4], s4, spec.refs[4])
        reactions = [
            Reaction(r1, [R1], [C, G], [1], [w1 * char_yield1, w1 * (1 - char_yield1)]; only_use_rate=true),
            Reaction(r2, [R2], [C, G], [1], [w2 * char_yield2, w2 * (1 - char_yield2)]; only_use_rate=true),
            Reaction(r3, [C], [G], [1], [1]; only_use_rate=true),
            Reaction(r4, [F], [G], [1], [fiber_fraction]; only_use_rate=true),
        ]
        weights = Dict(R1 => w1, R2 => w2, C => 1, F => fiber_fraction, G => 1)
        observable = "(1-fiber_fraction)*(matrix_split*R1+(1-matrix_split)*R2)+C+fiber_fraction*F"
        initial = [R1 => 1.0, R2 => 1.0, C => 0.0, F => 1.0, G => 0.0]
    else
        ps = [logk1, E1, s1, logk2, E2, s2, logk3, E3, s3, fiber_fraction, char_yield1, history_id, air, noise_scale]
        ss = [R1, C, F, G]
        w1 = 1 - fiber_fraction
        r1 = chan_rate(R1, logk1, E1, spec.laws[1], s1, spec.refs[1])
        r2 = air * (1 - fiber_fraction) * chan_rate(C / (1 - fiber_fraction), logk2, E2, spec.laws[2], s2, spec.refs[2])
        r3 = air * chan_rate(F, logk3, E3, spec.laws[3], s3, spec.refs[3])
        reactions = [
            Reaction(r1, [R1], [C, G], [1], [w1 * char_yield1, w1 * (1 - char_yield1)]; only_use_rate=true),
            Reaction(r2, [C], [G], [1], [1]; only_use_rate=true),
            Reaction(r3, [F], [G], [1], [fiber_fraction]; only_use_rate=true),
        ]
        weights = Dict(R1 => w1, C => 1, F => fiber_fraction, G => 1)
        observable = "(1-fiber_fraction)*R1+C+fiber_fraction*F"
        initial = [R1 => 1.0, C => 0.0, F => 1.0, G => 0.0]
    end

    for reaction in reactions
        delta = sum(weights[s] * nu for (s, nu) in zip(reaction.products, reaction.prodstoich)) -
                sum(weights[s] * nu for (s, nu) in zip(reaction.substrates, reaction.substoich))
        Symbolics.value(Symbolics.simplify(delta; expand=true)) == 0 || error("Nonconservative CFRP PEtab reaction")
    end

    @named network = ReactionSystem(reactions, t, ss, ps)
    network = complete(network)

    # Map spec.names to PEtab parameter symbols (mapping free shape names to s1..s4)
    shape_alias = Dict(v => "s$k" for (k, v) in spec.free_shape_names)
    petab_names = [get(shape_alias, n, n) for n in spec.names]
    parameters = [PEtabParameter(Symbol(pn); value=v, lb=lo, ub=hi, scale=:lin)
        for (pn, v, lo, hi) in zip(petab_names, x, spec.lower, spec.upper)]

    observations = DataFrame(simulation_id=String[], obs_id=String[], time=Float64[], measurement=Float64[])
    conditions = PEtabCondition[]
    for (j, r) in enumerate(runs)
        push!(conditions, PEtabCondition(Symbol(r.id), :history_id => Float64(firstid + j - 1), :air => Float64(r.air), :noise_scale => sqrt(length(r.mass) * length(runs))))
        append!(observations, DataFrame(simulation_id=fill(r.id, length(r.mass)), obs_id=fill("mass", length(r.mass)), time=r.obstime, measurement=r.mass))
    end

    s_syms = (s1, s2, s3, s4)
    defaults = Pair[history_id => Float64(firstid), air => 1.0, noise_scale => 1.0]
    for (j, val) in spec.fixed_shapes
        push!(defaults, s_syms[j] => val)
    end
    if spec.fiber_idx == 0
        push!(defaults, fiber_fraction => spec.fixed_fiber_val)
    end
    model = PEtabModel(network, PEtabObservable(:mass, observable, :noise_scale), observations, parameters;
        simulation_conditions=conditions, speciemap=initial, parametermap=defaults)

    setup_prob = nothing; solvername = ""; success = false
    for algorithm in (Rodas5P(), QNDF())
        solvername = string(nameof(typeof(algorithm)))
        prob = PEtabODEProblem(model; odesolver=ODESolver(algorithm; abstol=rtol / 100, reltol=rtol, maxiters=Int64(200_000)),
            gradient_method=:ForwardDiff, hessian_method=:ForwardDiff)
        values = Dict(Symbol(pn) => v for (pn, v) in zip(petab_names, x))
        point = [values[n] for n in propertynames(prob.xnominal_transformed)]
        ll = prob.nllh(point)
        if isfinite(ll)
            setup_prob = prob
            success = true
            break
        end
    end
    success || error("CFRP PEtab verification failed with both Rodas5P and QNDF")

    pm = setup_prob.model_info.petab_measurements
    lookup = Dict{Tuple{String,Float64},Float64}()
    for r in runs
        pred = predict_cfrp(spec, x, r; step_C=0.125)
        for j in eachindex(r.mass)
            lookup[(r.id, r.obstime[j])] = pred[j]
        end
    end
    gap = maximum(abs(pm.simulated_values[j] - lookup[(string(pm.simulation_condition_id[j]), pm.time[j])]) for j in eachindex(pm.time))
    gap < 5e-4 || error("CFRP PEtab / native mismatch: $(100gap) pp")
    (; model=string(spec.kind), rtol, solver=solvername, max_gap_pp=100gap, mass_balance_symbolic=true, passed=true)
end

end
