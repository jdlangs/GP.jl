#Covariance functions
export SECov

abstract type CovFunc end

#---------------------------------------
#Squared-exponential covariance function
#    separate length-scales per param
#---------------------------------------

type SECov{N} <: CovFunc
    sig_f::Float64
    lambdas::Vector{Float64}
    function SECov{N}(sf, l::Vector) where N
        @assert isa(N, Integer)
        @assert length(l) == N
        new(sf, l)
    end
end
SECov(p::Vector) = SECov{length(p)-1}(p[1], p[2:end])

function Base.show{N}(io::IO, K::SECov{N})
    print(io,"[sig_f: $(@sprintf("%3.3f",K.sig_f)), lambdas: ")
    _printvec(io, K.lambdas)
    print(io, "]")
end

#Calculate covariance for a pair of points
function (cf::SECov{N}){N}(x1::Vector, x2::Vector)
    @assert length(x1) == N
    @assert length(x2) == N
    cf.sig_f^2*exp(-0.5*dot(x1-x2, diagm(1./cf.lambdas)*(x1-x2)))
end

#Generate covariance matrix from sets of points
function (cf::SECov{N}){N}(X1, X2)
    @assert size(X1,1) == N
    @assert size(X2,1) == N
    D = diagm(1./cf.lambdas)
    K = zeros(size(X1,2),size(X2,2))
    for i=1:size(X1,2)
        for j=1:size(X2,2)
            dx = X1[:,i] - X2[:,j]
            K[i,j] = cf.sig_f^2*exp(-0.5*dot(dx, D*dx))
        end
    end
    K
end
(cf::SECov{N}){N}(X::Matrix) = cf(X,X)

#Return the derivative of a covariance matrix for a data set with respect to a
#single hyperparameter.
#Parameter i == 0: sig_f
function grad{N}(cf::SECov{N}, ::Type{Val{0}}, X)
    D = diagm(1./cf.lambdas)
    K = zeros(size(X,2),size(X,2))
    for i=1:size(X,2)
        for j=1:size(X,2)
            dx = X[:,i] - X[:,j]
            K[i,j] = 2*cf.sig_f*exp(-0.5*dot(dx, D*dx))
        end
    end
    K
end

#Parameter 1 <= i <= n: lambda_i
function grad{Nd,N}(cf::SECov{Nd}, ::Type{Val{N}}, X)
    @assert 1 <= N <= Nd
    D = diagm(1./cf.lambdas)
    K = zeros(size(X,2),size(X,2))
    for i=1:size(X,2)
        for j=1:size(X,2)
            dx = X[:,i] - X[:,j]
            K[i,j] = cf.sig_f^2*exp(-0.5*dot(dx, D*dx))*(dx[N]^2/(2*cf.lambdas[N]^2))
        end
    end
    K
end

"Compute the derivatives of the test point covariance vector w.r.t. the
prediction variables"
function covgrad{Nd}(x::Vector, cf::SECov{Nd}, X)
    @assert length(x) == Nd
    @assert size(X,1) == Nd
    Kd = zeros(size(X,2), Nd)
    D = diagm(1./cf.lambdas)
    for j=1:Nd
        ej = zeros(Nd)
        ej[j] = 1.0
        for i=1:size(X,2)
            dx = X[:,i] - x
            Kd[i,j] = cf.sig_f^2*exp(-0.5*dot(dx,D*dx))*dot(dx,D*ej)
        end
    end
    Kd
end

#The squared-exponential symmetric covariance function
type SECovSym{N} <: CovFunc
    sig_f::Float64
    lambda::Float64
    function SECovSym{N}(sf, l) where N
        @assert isa(N, Integer)
        new(sf, l)
    end
end
SECovSym(N, p::Vector) = SECovSym{N}(p[1], p[2])

function Base.show{N}(io::IO, K::SECovSym{N})
    print(io, @sprintf("[sig_f: %3.3f, lambda: %3.3f]", K.sig_f, K.lambda))
end

#Calculate covariance for a pair of points
function (cf::SECovSym{N}){N}(x1::Vector, x2::Vector)
    @assert length(x1) == N
    @assert length(x2) == N
    cf.sig_f^2*exp(dot(-0.5*(x1-x2),diagm(1./(cf.lambda*ones(N)))*(x1-x2)))
end

#Generate covariance matrix from sets of points
function (cf::SECovSym{N}){N}(X1, X2)
    @assert size(X1,1) == N
    @assert size(X2,1) == N
    D = diagm(1./(cf.lambda*ones(N)))
    K = zeros(size(X1,2),size(X2,2))
    for i=1:size(X1,2)
        for j=1:size(X2,2)
            dx = X1[:,i] - X2[:,j]
            K[i,j] = cf.sig_f^2*exp(-0.5*dot(dx, D*dx))
        end
    end
    K
end
(cf::SECovSym{N}){N}(X::Matrix) = cf(X,X)

#Return the derivative of a covariance matrix for a data set with respect to a
#single hyperparameter.
#Parameter i == 0: sig_f
function grad{N}(cf::SECovSym{N}, ::Type{Val{0}}, X)
    D = diagm(1./(cf.lambda*ones(N)))
    K = zeros(size(X,2),size(X,2))
    for i=1:size(X,2)
        for j=1:size(X,2)
            dx = X[:,i] - X[:,j]
            K[i,j] = 2*cf.sig_f*exp(-0.5*dot(dx, D*dx))
        end
    end
    K
end

#Parameter 1 : lambda
#TODO: recalculate
function grad{N}(cf::SECovSym{N}, ::Type{Val{1}}, X)
    D = diagm(1./(cf.lambda*ones(N)))
    K = zeros(size(X,2),size(X,2))
    for i=1:size(X,2)
        for j=1:size(X,2)
            dx = X[:,i] - X[:,j]
            K[i,j] = cf.sig_f^2*exp(-0.5*dot(dx, D*dx))*(dx[N]^2/(2*cf.lambda^2))
        end
    end
    K
end
