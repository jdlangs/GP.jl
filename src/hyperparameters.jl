export parameteropt

#Optimize the hyperparameters of a GP, given a data set
#For now, all parameters are in the covariance function
function parameteropt{N}(gp::GaussianProcess{N}, X, y)
    function func(params)
        #println("Func call: ", params)
        llfunc(params, gp, X, y)
    end
    function gradf!(params, xgrad)
        #println("Grad call: ", params)
        llgrad!(xgrad, params, gp, X, y)
    end
    function nlopt_func(params, grad)
        llfunc_and_grad!(grad, params, gp, X, y)
    end

    init_params = [gp.K.sig_f; gp.K.lambdas]

    #NLOpt solving
    optprob = NLopt.Opt(:LD_MMA, length(init_params))
    NLopt.min_objective!(optprob, nlopt_func)
    NLopt.lower_bounds!(optprob, 0.01)
    NLopt.ftol_abs!(optprob, 1e-1)
    NLopt.maxeval!(optprob,20)

    (optf,optx,ret) = NLopt.optimize(optprob, init_params)
    optx
end
parameteropt{N}(gp::GaussianProcess{N}) = parameteropt(gp, gp.Xtrn, gp.ftrn)

#Simultaneously calculate log-likelihood and its gradient
#In-place update xgrad and return the llfunc value
function llfunc_and_grad!{N}(xgrad, params, gp::GaussianProcess{N}, X, y)
    K = SECov{N}(params[1], params[2:end])
    tgp = SEGP(K, gp.sig_n2, X, y)

    Linv = inv(tgp.L)
    Kinv = Linv'*Linv
    for i=1:length(xgrad)
        dK_i = grad(gp.K, Val{i-1}, X)
        xgrad[i] = -0.5*(y'*Kinv*dK_i*Kinv*y)[1] +
                    0.5*trace(Kinv*dK_i)
    end
    -loglikelihood(tgp)
end

#Calc log-likelihood
function llfunc{N}(params, gp::GaussianProcess{N}, X, y)
    K = SECov{N}(params[1], params[2:end])
    tgp = SEGP(K, gp.sig_n2, X, y)
    -loglikelihood(tgp)
end

#Calc log-likelihood gradient
function llgrad!{N}(xgrad, params, gp::GaussianProcess{N}, X, y)
    #println("grad params: ", params)
    Kt = SECov{N}(params[1], params[2:end])
    tgp = SEGP(Kt, gp.sig_n2, X, y)
    Linv = inv(tgp.L)
    Kinv = Linv'*Linv

    Np = length(params)
    for i=1:Np
        dK_i = grad(gp.K, Val{i-1}, X)
        xgrad[i] = -0.5*(y'*Kinv*dK_i*Kinv*y)[1] +
                    0.5*trace(Kinv*dK_i)
    end
    Ngrad = norm(xgrad)
    for i=1:Np
        xgrad[i] = xgrad[i] / Ngrad
    end
    xgrad
end

#Debugging function
function checkgrad{N}(params, gp::GaussianProcess{N}, X, y)
    Np = length(params)
    dx = 1e-6
    out = zeros(Np,2)

    f0 = llfunc(params, gp, X, y)
    for i=1:Np
        pdiff = copy(params)
        pdiff[i] = params[i] + dx
        fpi = llfunc(pdiff, gp, X, y)
        pdiff[i] = params[i] - dx
        fni = llfunc(pdiff, gp, X, y)
        out[i,:] = [(fpi-f0) (f0-fni)]/dx
    end

    xgrad = zeros(Np)
    llgrad!(xgrad, params, gp, X, y)
    hcat(xgrad, out)
end
