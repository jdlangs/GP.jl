#Estimate the probability of a sample being in the interval [x1,x2] for a
#single-variable normal distribution
function normprob(x1::Real, x2::Real, mu::Real, sigma2::Real)
    sigma = sqrt(sigma2)
    0.5*(erf((x2-mu)/(sqrt(2)*sigma)) - erf((x1-mu)/(sqrt(2)*sigma)))
end

function adjustparam(x::Vector, dx_idx::Integer, dx::Real)
    dxv = zeros(length(x))
    dxv[abs(dx_idx)] = dx*sign(dx_idx)
    x + dxv
end

function isrepeat{D}(pt::Vector, model::GaussianProcess{D}, tol)
    for p = 1:length(model)
        if all(abs(getx(model,p) - pt) .<= tol*ones(D))
            return p
        end
    end
    return -1
end

#A common print helper
function _printvec(io::IO, x)
    print(io,"(")
    for i=1:length(x)
        @printf(io, "%6.3f", x[i])
        if i != length(x)
            print(io, ", ")
        end
    end
    print(io,")")
end
