struct modelHNBF
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

    matGamma::Array{Float64,2}
    matGamma_Shp::Array{Float64,2}
    matGamma_Rte::Array{Float64,2}

    vecMu::Array{Float64,1}
    vecMu_Shp::Array{Float64,1}
    vecMu_Rte::Array{Float64,1}

    matDelta::Array{Float64,2}
    matDelta_Shp::Array{Float64,2}
    matDelta_Rte::Array{Float64,2}

    vecPi::Array{Float64,1}
    vecPi_Shp::Array{Float64,1}
    vecPi_Rte::Array{Float64,1}

    vec_matR_ui::Array{Float64,1}
    vec_matR_ui_Shp::Array{Float64,1}
    vec_matR_ui_Rte::Array{Float64,1}

    vec_matD_ui::Array{Float64,1}
    vec_matD_ui_Shp::Array{Float64,1}
    vec_matD_ui_Rte::Array{Float64,1}

end
