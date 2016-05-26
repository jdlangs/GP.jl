#Main definitions
export GaussianProcess, SEGP, MSEGP, query, train!, update!

type GaussianProcess{N}
    m::MeanFunc
    K::CovFunc
    sig_n2::Float64
    Xtrn::Matrix{Float64}
    ftrn::Vector{Float64}

    #These variables are computed during training
    L::Matrix{Float64}
    alpha::Vector{Float64}
    #For debug printing
    printlevel::Int

    function GaussianProcess(m, K, sn2)
        new(m, K, sn2, zeros(N,0), zeros(0), zeros(0,0), zeros(0), 0)
    end
end
SEGP(K,sn2,Nd) = GaussianProcess{Nd}(ZeroMean(), K, sn2)
SEGP(K,sn2,X,y) = train!(SEGP(K,sn2,size(X,1)), X, y)

MSEGP(m,K,sn2,Nd) = GaussianProcess{Nd}(m, K, sn2)
MSEGP(m,K,sn2,X,y) = train!(MSEGP(m,K,sn2,size(X,1)), X, y)

Base.show{N}(io::IO, gp::GaussianProcess{N}) =
    print(io, "GP{$N}(Ndata = $(length(gp.ftrn)), K = $(gp.K))")
Base.size{D}(gp::GaussianProcess{D}) = D,length(gp)
Base.length(gp::GaussianProcess) = size(gp.Xtrn, 2)
getx{D}(gp::GaussianProcess{D}, idx::Int) = gp.Xtrn[:,idx]
getf{D}(gp::GaussianProcess{D}, idx::Int) = gp.ftrn[idx]

#Do calculations that depend on training data only
function train!{N}(gp::GaussianProcess{N}, xt, ft)
    gp.Xtrn = copy(xt)
    gp.ftrn = copy(ft)

    fmean = Float64[gp.m(gp.Xtrn[:,i]) for i=1:length(ft)]

    Kt = gp.K(gp.Xtrn, gp.Xtrn)
    Kn = Kt + gp.sig_n2*eye(length(ft)) #n x n
    gp.L = try
        chol(Kn)'
    catch ex
        println("training K info: ", gp.K)
        error("chol failure")
    end
    gp.alpha = gp.L' \ (gp.L \ (gp.ftrn - fmean)) #n x 1
    gp
end

#An incremental update given a single new point. Just as expensize as a full
#batch training.
function update!{N}(gp::GaussianProcess{N}, xn::Vector, fn::Real)
    Xtrn = length(gp) == 0 ?
        reshape(xn,length(xn),1) :
        hcat(gp.Xtrn, xn)
    train!(gp, Xtrn, [gp.ftrn; fn])
end

#Incremental update with multiple new points
function update!{N}(gp::GaussianProcess{N}, Xn::Matrix, fn::Vector)
    Xtrn = hcat(gp.Xtrn, Xn)
    ftrn = vcat(gp.ftrn, fn)
    train!(gp, Xtrn, ftrn)
end

#Evaluate the mean and variance of a single test point
function query{N}(x::Vector, gp::GaussianProcess{N})
    ktrn = vec(gp.K(gp.Xtrn, x)) #n x 1
    mt = dot(ktrn, gp.alpha) + gp.m(x)
    v = gp.L \ ktrn
    vt = gp.K(x,x) - dot(v,v)
    mt, vt
end

#Marginal log-likelihood
function loglikelihood{N}(gp::GaussianProcess{N})
    -0.5*dot(gp.ftrn,gp.alpha) -
        sum(log(diag(gp.L))) -
        length(gp.ftrn)*log(2pi)/2
end

#Condition a set of test points given a trained GP
function condition{N}(x, gp::GaussianProcess{N})
    @assert length(gp.ftrn) != 0 "The process must be trained to condition"
    K_x_xt  = gp.K(x, gp.Xtrn)
    K_xt_x  = K_x_xt'
    K_x_x   = gp.K(x, x)
    Kinv = inv(gp.K(gp.Xtrn, gp.Xtrn))

    mcond = K_x_xt * Kinv * gp.ftrn
    Kcond = K_x_x - K_x_xt * Kinv * K_xt_x
    mcond, Kcond
end

function gradient!{N}(grad::Vector, x::Vector, gp::GaussianProcess{N})
    grad[:] = covgrad(x, gp.K, gp.Xtrn)'*gp.alpha
    grad
end
gradient{N}(x::Vector, gp::GaussianProcess{N}) = gradient!(zeros(N), x, gp)

function checkgrad{N}(x::Vector, gp::GaussianProcess{N}, dx)
    grad = zeros(N)
    for i=1:N
        dv = zeros(N)
        dv[i] = dx
        xt = x + dv
        m1,sig1 = query(x, gp)
        m2,sig2 = query(xt,gp)
        grad[i] = (m2-m1)/dx
    end
    grad
end
