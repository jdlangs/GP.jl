#---------------------------------------------
#  GP Ensemble - multiple GPs blended together
#---------------------------------------------
export GPEnsemble

type GPEnsemble{N}
    models::Vector{GaussianProcess{N}}
    cntrs::Vector{Vector{Float64}}

    w_gen::Float64
    w_qry::Float64
    sig_n2::Float64
    function GPEnsemble(wg,wq,sn2)
        new(GaussianProcess{N}[], Vector{Float64}[], wg, wq, sn2)
    end
end
function Base.show{N}(io::IO, gpm::GPEnsemble{N})
    print(io,string(
        "GPEnsemble{$N}(Nmdls = $(length(gpm.models)), ",
        "w_gen = $(gpm.w_gen)):"
        ))
    for i=1:length(gpm.models)
        @printf(io, "\n\t%3u : ", i)
        _printvec(io, gpm.cntrs[i])
        print(io, " -> ", gpm.models[i])
    end
end

function update!{N}(gpm::GPEnsemble{N}, xn, fn)
    mdl_dists = length(gpm.models) == 0 ? [0.0] :
        [gpm.models[i].K(xn, gpm.cntrs[i]) for i=1:length(gpm.models)]

    max_simil, max_idx = findmax(mdl_dists)
    if max_simil < gpm.w_gen
        #Generate new model
        push!(gpm.cntrs, xn)
        push!(gpm.models, SEGP(SECov{N}(1.0,0.5*ones(N)), gpm.sig_n2, N))
        update!(gpm.models[end],xn,fn)
    else
        #Put into existing model
        lgp = gpm.models[max_idx]
        Xnew = hcat(lgp.Xtrn, xn)
        fnew = [lgp.ftrn; fn]
        popt = parameteropt(lgp, Xnew, fnew)

        gpm.models[max_idx] = SEGP(SECov{N}(popt[1],popt[2:end]),lgp.sig_n2,N)
        train!(gpm.models[max_idx], Xnew, fnew)
        gpm.cntrs[max_idx] = vec(mean(Xnew, 2))
    end
    gpm
end

function train!{N}(gpm::GPEnsemble{N}, X::Matrix, f::Vector)
    for i=1:size(X,2)
        update!(gpm, X[:,i], f[i])
    end
    gpm
end

function query{N}(xq, gpm::GPEnsemble{N})
    @assert length(gpm.models) > 0
    ws = [gpm.models[i].K(xq, gpm.cntrs[i]) for i=1:length(gpm.models)]
    closest_idx = find(d -> d > gpm.w_qry, ws)
    if length(closest_idx) == 0
        closest_idx = [findmax(ws)[2]]
    end

    fmdls = Float64[query(xq, gpm.models[i])[1] for i in closest_idx]

    #Weighted average
    ws_close = ws[closest_idx]
    dot(ws_close, fmdls) / sum(ws_close)
end

function search{N}(ftrgt, tol, gpm::GPEnsemble{N}, startx)
    function optf(x)
        fq = query(x, gpm)
        abs(fq - ftrgt)
    end

    opt_pts = zeros(N, size(startx,2))
    opt_vals = zeros(size(startx,2))
    for i=1:size(startx,2)
        opt_result = Optim.optimize(optf, startx[:,i])
        opt_pts[:,i] = opt_result.minimum
        opt_vals[i]  = opt_result.f_minimum
    end
    opt_pts, opt_vals
end
