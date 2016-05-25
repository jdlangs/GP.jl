#Functions to find an input point that predicts a desired output
import ProgressMeter
using Optim

function search{D}(target::Real, model::GaussianProcess{D})
    moveinfo = Array(Vector{Float64}, length(model))
    ProgressMeter.@showprogress for i=1:length(model)
        moveinfo[i] = calcmove(i, model, target)
    end

    bidx = indmin([m[1] for m in moveinfo])
    if isinf(moveinfo[bidx][1])
        return NaN*ones(D)
    end
    adj_idx = round(Int, moveinfo[bidx][2])
    adj_val = moveinfo[bidx][3]

    npt = adjustparam(getx(model,bidx), adj_idx, adj_val)
    npt
end

function searchgreedy{N}(target, tol, gp::GaussianProcess{N}, startx)
    Cd = 1.0
    count = 0
    function optf(pt, grad)
        count += 1
        length(grad) == 0 || error("Request for grad")
        mpt, sig2pt = query(pt, gp)
        Cprob = 1 - normprob(target-tol, target+tol, mpt, sig2pt)
        Cdist = Cd*abs(mpt - target)
        Cprob + Cdist
    end
    opt_pts = zeros(N, size(startx,2))
    opt_vals = zeros(size(startx,2))
    for i=1:size(startx,2)
        #Global search
        optprob = NLopt.Opt(:GN_CRS2_LM, N)
        #optprob = NLopt.Opt(:LN_COBYLA, N)
        NLopt.lower_bounds!(optprob, -1.0*ones(N))
        NLopt.upper_bounds!(optprob,  1.0*ones(N))
        NLopt.min_objective!(optprob, optf)
        NLopt.ftol_abs!(optprob, 1e-8)
        NLopt.maxeval!(optprob, 500)
        minx = zeros(N)
        minf = 0.0
        ret = 0
        try
            (minf, minx, ret) = NLopt.optimize(optprob, startx[:,i])
        catch ex
            println("startx: ", startx)
            rethrow()
        end

        #Local refinement
        optprob = NLopt.Opt(:LN_COBYLA, N)
        NLopt.lower_bounds!(optprob, -1.0*ones(N))
        NLopt.upper_bounds!(optprob,  1.0*ones(N))
        NLopt.min_objective!(optprob, optf)
        NLopt.ftol_abs!(optprob, 1e-8)
        NLopt.maxeval!(optprob, 500)
        try
            (minf, minx, ret) = NLopt.optimize(optprob, minx)
        catch ex
            println("minx: ", minx)
            rethrow()
        end

        opt_pts[:,i] = minx
        opt_vals[i] = minf
    end
    opt_pts, opt_vals
end

function searchgrad{D}(target, model::GaussianProcess{D}, startx)
    count = 0
    function optf(pt)
        #count += 1
        mpt, sig2pt = query(pt, model)
        #if length(grad) > 0
            #gradient!(grad, pt, model)
            #grad[:] = 2.0*(mpt - target)*grad
        #end
        #println("#$count")
        #println("  x: $pt")
        #println("  g: $grad")
        (mpt - target)^2
    end

    function optg!(pt::Vector, grad)
        mpt, sig2pt = query(pt, model)
        gradient!(grad, pt, model)
        grad[:] = 2.0*(mpt - target)*grad
    end

    opt_pts = zeros(D, size(startx,2))
    opt_vals = zeros(size(startx,2))
    for i=1:size(startx,2)
        #optprob = NLopt.Opt(:LD_MMA, D)
        #NLopt.lower_bounds!(optprob, -1.0*ones(D))
        #NLopt.upper_bounds!(optprob,  1.0*ones(D))
        #NLopt.min_objective!(optprob, optf)
        #NLopt.ftol_abs!(optprob, 1e-8)
        #NLopt.maxeval!(optprob, 500)
        try
            #(minf, minx, ret) = NLopt.optimize(optprob, startx[:,i])
            res = optimize(optf, optg!, startx[:,i];
                           method = LBFGS(),
                           iterations = 100,
                           )
            minx = Optim.minimizer(res)
            minf = Optim.minimum(res)
            opt_pts[:,i] = minx
            opt_vals[i] = minf
        catch ex
            println("startx: ", startx[:,i])
            rethrow()
        end
    end
    opt_pts, opt_vals
end

function searchline{D}(
    target, tol, model::GaussianProcess{D}, func;
    maxDist=0.5, dT=0.0005
    )
    params = [-D:-1; 1:D]
    param_out = zeros(length(params), length(model))
    prerr_out = Inf*ones(length(params), length(model))
    trerr_out = Inf*ones(length(params), length(model))

    println("GPR: Searching over $(length(model)) points")
    ProgressMeter.@showprogress for midx=1:length(model)
        #println("Base point: ", midx)
        xbase = getx(model,midx)
        for (pidx,p) in enumerate(params)
            #print("\tParam ",p," : ")
            dx = zeros(D)
            dx[abs(p)] = sign(p)
            for t=0:dT:maxDist
                xtest = xbase + t*dx
                val,err = query(xtest, model)
                if abs(val - target) < tol
                    terr = abs(val - func(xtest))

                    param_out[pidx, midx] = t
                    prerr_out[pidx, midx] = err
                    trerr_out[pidx, midx] = terr
                    #println(round(Int,t/dT)+1)
                    break
                end
            end
            #Check if no solution was found for the target
            if trerr_out[pidx, midx] < 0.0
                #println(-1)
            end
        end
    end
    param_out, prerr_out, trerr_out
end

function calcmove{D}(idx, model::GaussianProcess{D}, target::Real)
    opt_repeat_param_tol = Main.options["repeat_param_tol"]

    params = [-D:-1; 1:D]
    param_adjs = zeros(2D,3)
    for pidx=1:length(params)
        oneparam!(idx, pidx, model, target, params, param_adjs)
    end
    bidx = indmin(param_adjs[:,1])

    error_val, dx, expct = param_adjs[bidx, :]
    adjinfo = [error_val, params[bidx], dx, expct]

    if isfinite(error_val)
        xval = getx(model,idx)
        nx = adjustparam(xval, params[bidx], dx)

        rpt = isrepeat(nx, model, opt_repeat_param_tol)
        if rpt > 0
            adjinfo[1] = Inf
        end
    end
    adjinfo
end

#Calculate the move for one param at one base point
function oneparam!{D}(
    idx::Integer,
    pidx::Integer,
    model::GaussianProcess{D},
    target::Real,
    params,
    param_adjs
    )
    opt_search_dx    = Main.options["gpr_search_dx"]
    opt_search_maxdx = Main.options["gpr_search_maxdx"]
    opt_search_tol   = Main.options["gpr_search_tol"]

    xval = getx(model,idx)
    fval = getf(model,idx)
    curr_err = fval - target

    param_adjs[pidx,:] = [Inf 0.0 0.0]
    for t=0:opt_search_dx:opt_search_maxdx
        xtest = adjustparam(xval, params[pidx], t)
        val,err = query(xtest, model)
        if abs(val - target) < opt_search_tol
            param_adjs[pidx,:] = [err t val]
            break
        end
    end
end
