# Optional global search adapter. Model equations, bounds and the local optimizer
# stay in their existing modules. Serial execution is required by exposure banks.
import Evolutionary
using StableRNGs: StableRNG

struct GeneticOptions
    seed::Int
    population::Int
    max_evaluations::Int
    refine_candidates::Int
    function GeneticOptions(;seed=42,population=32,max_evaluations=800,refine_candidates=3)
        values=(seed,population,max_evaluations,refine_candidates)
        all(v->v isa Integer&&!(v isa Bool)&&0<=v<=typemax(Int),values)||
            throw(ArgumentError("Genetic settings must be nonnegative, representable integers"))
        population>=4||throw(ArgumentError("population must be at least 4"))
        max_evaluations>=population||throw(ArgumentError("max_evaluations must cover the initial population"))
        1<=refine_candidates<=population||throw(ArgumentError("refine_candidates must be between 1 and population"))
        new(Int(seed),Int(population),Int(max_evaluations),Int(refine_candidates))
    end
end

# Only this private sentinel is caught around the optimizer. It prevents an
# unexpected extra package evaluation from ever exceeding the requested cap.
struct GeneticEvaluationLimit <: Exception end

function genetic_value(f,x)
    value=try
        f(x)
    catch err
        if err isa DomainError||err isa OverflowError
            return (value=Inf,reason=string(nameof(typeof(err))))
        end
        rethrow()
    end
    value isa Real||throw(ArgumentError("Genetic fitness must be a scalar real value"))
    isfinite(value) ? (value=Float64(value),reason="") : (value=Inf,reason="nonfinite_or_invalid_prediction")
end

function genetic_distinct(records,count;distance=.025)
    selected=NamedTuple[]
    for row in sort(records;by=r->r.objective)
        # The max coordinate distance is computed in the normalized free space.
        all(s->maximum(abs.(row.z.-s.z))>=distance,selected)||continue
        push!(selected,row)
        length(selected)==count&&break
    end
    selected
end

"""
    bounded_genetic_search(f, lower, upper; options=GeneticOptions(), starts=[], refine_objective=nothing)

Search a scalar bounded objective using Evolutionary.GA and a private StableRNG.
The evaluation cap includes every coarse fitness evaluation, including invalid
candidates. Whole generations fit within that cap. Optional refined re-scoring
is counted separately and cannot exceed four times `refine_candidates`.
"""
function bounded_genetic_search(f,lower,upper;options=GeneticOptions(),starts=Vector{Float64}[],refine_objective=nothing)
    options isa GeneticOptions||throw(ArgumentError("options must be GeneticOptions"))
    lo,hi=Float64.(lower),Float64.(upper)
    !isempty(lo)&&length(lo)==length(hi)&&all(isfinite,lo)&&all(isfinite,hi)&&all(lo.<hi)||
        throw(ArgumentError("Finite, strictly ordered bounds are required"))
    length(starts)<=options.population||throw(ArgumentError("Initial starts exceed population"))
    all(x->length(x)==length(lo)&&all(isfinite,x)&&all(lo.<=x.<=hi),starts)||
        throw(ArgumentError("Initial starts must be finite and inside bounds"))
    span=hi.-lo;rng=StableRNG(options.seed)
    population=[rand(rng,length(lo)) for _ in 1:options.population]
    for (i,x) in enumerate(starts);population[i]=(Float64.(x).-lo)./span;end
    archive=NamedTuple[];trace=NamedTuple[];evaluations=Ref(0);invalid=Ref(0)
    reasons=Dict{String,Int}();best=Ref(Inf)
    function fitness(z)
        evaluations[]<options.max_evaluations||throw(GeneticEvaluationLimit())
        evaluations[]+=1
        x=clamp.(lo.+z.*span,lo,hi)
        scored=genetic_value(f,x)
        if isfinite(scored.value)
            best[]=min(best[],scored.value)
            push!(archive,(x=copy(x),z=copy(z),objective=scored.value))
        else
            invalid[]+=1;reasons[scored.reason]=get(reasons,scored.reason,0)+1
        end
        scored.value
    end
    iterations=div(options.max_evaluations-options.population,options.population)
    function callback(record)
        push!(trace,(generation=record.iteration,evaluations=evaluations[],
            best_objective=best[],invalid_evaluations=invalid[]))
        evaluations[]+options.population>options.max_evaluations
    end
    method=Evolutionary.GA(populationSize=options.population,crossoverRate=1.,mutationRate=.85,
        epsilon=0,selection=Evolutionary.tournament(2),crossover=Evolutionary.IC(.15),
        mutation=Evolutionary.gaussian(.10),metrics=Evolutionary.ConvergenceMetric[Evolutionary.AbsDiff(0.)])
    opts=Evolutionary.Options(iterations=iterations,successive_f_tol=iterations+1,
        rng=rng,parallelization=:serial,callback=callback,store_trace=false,show_trace=false)
    stopped_by_guard=false
    started=time()
    # Passing the explicit scalar type value avoids the package constructor's
    # otherwise uncounted probe of f(first(population)).
    objective_wrapper=Evolutionary.EvolutionaryObjective(fitness,first(population),0.;eval=:serial)
    try
        Evolutionary.optimize(objective_wrapper,Evolutionary.BoxConstraints(zeros(length(lo)),ones(length(lo))),method,population,opts)
    catch err
        err isa GeneticEvaluationLimit||rethrow()
        stopped_by_guard=true
    end
    isempty(archive)&&error("Genetic search found no finite, physically valid candidates")
    # Keep an archive of successful evaluations, rather than trusting the last
    # population: GA need not retain the best individual in every generation.
    pool=genetic_distinct(archive,4options.refine_candidates)
    refined=NamedTuple[];refinement_evaluations=0;refinement_invalid=0
    for row in pool
        if isnothing(refine_objective)
            scored=(value=row.objective,reason="")
        else
            refinement_evaluations+=1;scored=genetic_value(refine_objective,row.x)
        end
        !isfinite(scored.value)&&(refinement_invalid+=1)
        push!(refined,(x=row.x,z=row.z,objective=scored.value,coarse_objective=row.objective))
    end
    chosen=genetic_distinct(filter(r->isfinite(r.objective),refined),options.refine_candidates)
    isempty(chosen)&&error("Genetic candidates all failed refined numerical evaluation")
    candidates=NamedTuple[]
    for (rank,row) in enumerate(sort(refined;by=r->r.objective))
        accepted=any(s->s.x==row.x,chosen)
        push!(candidates,(rank=rank,coarse_objective=row.coarse_objective,
            refined_objective=row.objective,accepted=accepted,x=copy(row.x)))
    end
    metadata=Dict{String,Any}("algorithm"=>"Evolutionary.GA plus existing local refinement",
        "evolutionary_version"=>string(Base.pkgversion(Evolutionary)),"rng"=>"StableRNGs.StableRNG",
        "seed"=>options.seed,"population"=>options.population,"max_evaluations"=>options.max_evaluations,
        "refine_candidates"=>options.refine_candidates,"evaluations"=>evaluations[],
        "refinement_evaluations"=>refinement_evaluations,"total_search_evaluations"=>evaluations[]+refinement_evaluations,
        "invalid_evaluations"=>invalid[],"invalid_reasons"=>reasons,"invalid_refinement_evaluations"=>refinement_invalid,
        "generations"=>isempty(trace) ? 0 : last(trace).generation,
        "unused_evaluation_budget"=>options.max_evaluations-evaluations[],
        "stop_reason"=>stopped_by_guard ? "hard_evaluation_guard" : "evaluation_budget_complete_generations",
        "elapsed_seconds"=>time()-started,"best_coarse_objective"=>best[],
        "best_refined_objective"=>minimum(r.objective for r in chosen),
        "parallelization"=>"serial","coordinates"=>"existing free coordinates normalized to [0,1]",
        "diversity_max_coordinate_distance"=>.025,"crossover"=>"IC(0.15)",
        "crossover_rate"=>1.,"mutation"=>"gaussian(0.10)","mutation_rate"=>.85,
        "selection"=>"tournament(2)","population_elites"=>0,
        "refinement_budget"=>"separately counted, at most 4 * refine_candidates scalar evaluations")
    (starts=[copy(r.x) for r in chosen],metadata=metadata,trace=trace,candidates=candidates)
end

function genetic_material_loss(study,spec,x,indices;level=0)
    predictions=predict(study,spec,x;level)
    for i in indices
        q=predictions[i]
        all(isfinite,q)&&minimum(q)>=-1e-9&&maximum(q)<=1+1e-9||return Inf
    end
    sum(mean(abs2,predictions[i].-study.runs[i].mass) for i in indices)/(2length(indices))
end

"""Generate training-only global starts for the unchanged local estimator."""
function search_starts(study,spec;indices=collect(eachindex(study.runs)),options=GeneticOptions())
    selected=collect(indices)
    !isempty(selected)&&length(unique(selected))==length(selected)&&
        all(i->i isa Integer&&!(i isa Bool)&&i in eachindex(study.runs),selected)||
        throw(ArgumentError("Training indices must be unique valid run indices"))
    result=bounded_genetic_search(x->genetic_material_loss(study,spec,x,selected),spec.lower,spec.upper;
        options,starts=spec.starts,refine_objective=x->genetic_material_loss(study,spec,x,selected;level=2))
    result.metadata["training_indices"]=selected
    result.metadata["objective"]="One half the mean of per-training-condition mean squared mass errors"
    result.metadata["search_level"]=0;result.metadata["refinement_level"]=2
    result.metadata["parameter_names"]=spec.names
    result
end
