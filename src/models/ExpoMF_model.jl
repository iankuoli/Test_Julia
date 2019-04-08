struct modelExpoMF
    K::Int
    M::Int
    N::Int

    matTheta::Array{Float64,2}
    matBeta::Array{Float64,2}
    vecMu::Array{Float64,1}

    valLambda_y::Float64                # inverse variance of data Y
    valLambda_theta::Float64            # inverse variance of matrix G_matTheta
    valLambda_beta::Float64             # inverse variance of matrix G_matbeta
    val_alpha1::Float64
    val_alpha2::Float64
end
