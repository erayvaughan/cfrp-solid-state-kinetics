const ROOT = normpath(joinpath(@__DIR__, ".."))
const GAS_R_SI = 8.31446261815324
const LOG10_60 = log10(60.0)

logk_min_to_s(logk_min) = logk_min - LOG10_60
logk_s_to_min(logk_s) = logk_s + LOG10_60

function arrhenius_A_s_inv(logk, E_100kJ, Tref_K; rate_unit::Symbol=:s_inv)
    rate_unit in (:s_inv, :min_inv) || error("Unsupported rate_unit: $rate_unit")
    logk_s = rate_unit == :min_inv ? logk_min_to_s(logk) : logk
    10.0^logk_s * exp(E_100kJ * 1e5 / (GAS_R_SI * Tref_K))
end

function rate_s_inv(logk, E_100kJ, T_K, Tref_K; rate_unit::Symbol=:s_inv)
    rate_unit in (:s_inv, :min_inv) || error("Unsupported rate_unit: $rate_unit")
    logk_s = rate_unit == :min_inv ? logk_min_to_s(logk) : logk
    exp(log(10.0) * logk_s - E_100kJ * 1e5 / GAS_R_SI * (1 / T_K - 1 / Tref_K))
end

function standardized_triplets(df; engine="unknown", rate_unit="s_inv")
    cols = Set(string.(propertynames(df)))
    rows = NamedTuple[]
    ru = string(rate_unit)
    eng = string(engine)
    for r in eachrow(df)
        model_s = "model" in cols ? string(r.model) : eng
        reaction_s = "reaction" in cols ? string(r.reaction) : string(r.route)
        A_val = Float64("A_s_inv" in cols ? r.A_s_inv : r.A_per_s)
        E_val = Float64(r.E_kJ_mol)
        ref_T = Float64("reference_temperature_K" in cols ? r.reference_temperature_K : r.ref_K)
        if "shape_param" in cols
            law_s = "law" in cols ? string(r.law) : "unknown"
            sp = string(r.shape_param)
            sv = Float64(r.shape_value)
            sf = Bool(r.shape_fixed)
        else
            mech = "mechanism" in cols ? string(r.mechanism) : "nth_order"
            law_s = mech
            if mech == "random_pore" && "secondary_value" in cols
                sp = "psi"; sv = Float64(r.secondary_value); sf = false
            elseif mech == "avrami_erofeev" && "secondary_value" in cols
                sp = "m"; sv = Float64(r.secondary_value); sf = false
            else
                sp = "n"; sv = Float64(r.n)
                sf = "order_fixed" in cols ? Bool(r.order_fixed) : (string(r.order_status) == "fixed")
            end
        end
        push!(rows, (
            engine=eng, model=model_s, reaction=reaction_s, law=law_s,
            A_s_inv=A_val, E_kJ_mol=E_val, shape_param=sp, shape_value=sv, shape_fixed=sf,
            reference_temperature_K=ref_T, rate_unit=ru,
        ))
    end
    DataFrame(rows)
end

filehash(path) = open(io -> bytes2hex(sha256(io)), path)
function write_toml(path, value)
    open(path, "w") do io
        TOML.print(io, value; sorted=true)
    end
end
function resultdir(prefix)
    mkpath(joinpath(ROOT, "results"))
    return mktempdir(joinpath(ROOT, "results"); prefix=prefix,cleanup=false)
end
