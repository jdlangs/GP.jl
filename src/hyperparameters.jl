export parameteropt

#Optimize the hyperparameters of a GP, given a data set
#For now, all parameters are in the covariance function
function parameteropt(initp::Vector, X, y)
    sig_n2 = 0.001
    function func(params)
        #println("Func call: ", params)
        llfunc(params, sig_n2, X, y)
    end
    function gradf!(params, xgrad)
        #println("Grad call: ", params)
        llgrad!(xgrad, params, sig_n2, X, y)
    end
    function nlopt_func(params, grad)
        llfunc_and_grad!(grad, params, sig_n2, X, y)
    end


    #NLOpt solving
    optprob = NLopt.Opt(:LD_MMA, length(initp))
    NLopt.min_objective!(optprob, nlopt_func)
    NLopt.lower_bounds!(optprob, 0.01)
    NLopt.upper_bounds!(optprob, 2.0)
    NLopt.ftol_abs!(optprob, 1e-3)
    NLopt.maxeval!(optprob, 1000)

    (optf,optx,ret) = NLopt.optimize(optprob, initp)
    optx
end

function parameteropt{N}(gp::GaussianProcess{N}, X, y)
    init_params = [gp.K.sig_f; gp.K.lambdas]
    parameteropt(init_params, X, y)
end
parameteropt{N}(gp::GaussianProcess{N}) = parameteropt(gp, gp.Xtrn, gp.ftrn)


#Simultaneously calculate log-likelihood and its gradient
#In-place update xgrad and return the llfunc value
function llfunc_and_grad!(xgrad, params, sig_n2, X, y)
    K = SECov(params)
    tgp = SEGP(K, sig_n2, X, y)

    Linv = inv(tgp.L)
    Kinv = Linv'*Linv
    for i=1:length(xgrad)
        dK_i = grad(tgp.K, Val{i-1}, X)
        xgrad[i] = -0.5*(y'*Kinv*dK_i*Kinv*y)[1] +
                    0.5*trace(Kinv*dK_i)
    end
    -loglikelihood(tgp)
end

#Calc log-likelihood
function llfunc(params, sig_n2, X, y)
    K = SECov(params)
    tgp = SEGP(K, sig_n2, X, y)
    -loglikelihood(tgp)
end

#Calc log-likelihood gradient
function llgrad!(xgrad, params, sig_n2, X, y)
    #println("grad params: ", params)
    Kt = SECov(params)
    tgp = SEGP(Kt, sig_n2, X, y)
    Linv = inv(tgp.L)
    Kinv = Linv'*Linv

    Np = length(params)
    for i=1:Np
        dK_i = grad(tgp.K, Val{i-1}, X)
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

    f0 = llfunc(params, gp.sig_n2, X, y)
    for i=1:Np
        pdiff = copy(params)
        pdiff[i] = params[i] + dx
        fpi = llfunc(pdiff, gp.sig_n2, X, y)
        pdiff[i] = params[i] - dx
        fni = llfunc(pdiff, gp.sig_n2, X, y)
        out[i,:] = [(fpi-f0) (f0-fni)]/dx
    end

    xgrad = zeros(Np)
    llgrad!(xgrad, params, gp.sig_n2, X, y)
    hcat(xgrad, out)
end
