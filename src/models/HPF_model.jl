struct modelHPF
    K::Int
    M::Int
    N::Int

    prior::Array{Float64,1}

    matTheta::Array{Float64,2}
    matTheta_Shp::Array{Float64,2}
    matTheta_Rte::Array{Float64,2}

    vecEpsilon::Array{Float64,1}
    vecEpsilon_Shp::Array{Float64,1}
    vecEpsilon_Rte::Array{Float64,1}

    matBeta::Array{Float64,2}
    matBeta_Shp::Array{Float64,2}
    matBeta_Rte::Array{Float64,2}

    vecEta::Array{Float64,1}
    vecEta_Shp::Array{Float64,1}
    vecEta_Rte::Array{Float64,1}

end
