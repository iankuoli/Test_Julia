function Evaluate_LogLikelihood_Poisson(vec_obsrv::Array{Float64,1}, vec_pred::Array{Float64,1})

    # poisson(k | lambda) = lambda^K * exp(-lambda) ./ gamma(k+1)

    ret = vec_obsrv .* log.(vec_pred) .- vec_pred .- lgamma.(vec_obsrv .+ 1.)
end
