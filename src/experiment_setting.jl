
dictHNBF_SETTING = Dict("SmallToy" =>      (1, 100, [4], [1,2,3,5],
                                            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                                             200, 1e6, 200, 1e6, 200, 1e4]),
                        "SmallToyML" =>    (2, 400, [4], [1,2,3,5],
                                            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                                             200, 1e6, 200, 1e6, 200, 1e4]),
                        "ML50" =>          (2, 400, [4], [1,2,3,5],
                                            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                                             200, 1e6, 200, 1e6, 200, 1e4]),
                        "MovieLens100K" => (2, 400, [20], [5, 10, 15, 20],
                                            [3., 1., 0.1, 3., 1., 0.1,
                                             1e2, 1e6, 1e2, 1e6, 2., 1e4]),
                        "MovieLens1M" =>   (2, 400, [20], [5, 10, 15, 20],
                                            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                                             1e2, 1e6, 1e2, 1e6, 2, 1e4]),
                        "LastFm2K" =>      (1, 400, [20], [5, 10, 15, 20],
                                            [3, 1, 0.1, 3, 1, 0.1,
                                             1e2, 1e8, 1e2, 1e8, 200, 1e4]),
                        "LastFm1K" =>      (1, 400, [20], [5, 10, 15, 20],
                                            [3, 1, 0.1, 3, 1, 0.1,
                                             1e2, 1e8, 1e2, 1e8, 200, 1e4]),
                        "EchoNest" =>      (1, 600, [20], [5, 10, 15, 20],
                                            [3, 1, 0.1, 3, 1, 0.1,
                                             1e4, 1e8, 1e4, 1e8, 200, 1e4]),
                        "LastFm360K" =>    (1, 600, [20], [5, 10, 15, 20],
                                            [3, 1, 0.1, 3, 1, 0.1,
                                             1e7, 1e11, 1e7, 1e11, 200, 1e4]),
                        "LastFm360K_2K" => (1, 600, [20], [5, 10, 15, 20],
                                            [3, 1, 0.1, 3, 1, 0.1,
                                             1e2, 1e8, 1e2, 1e8, 200, 1e4]),
                        "ML100KPos" =>     (3, 600, [20], [5, 10, 15, 20],
                                            [3, 1, 0.1, 3, 1, 0.1,
                                             1e2, 1e8, 1e2, 1e8, 200, 1e4]))

dictHPF_SETTING = Dict("SmallToy" =>      (1, 200, [4], [1,2,3,5],
                                           [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
                       "SmallToyML" =>    (2, 200, [4], [1,2,3,5],
                                           [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
                       "ML50" =>          (2, 200, [4], [1,2,3,5],
                                           [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
                       "MovieLens100K" => (2, 200, [20], [5, 10, 15, 20],
                                           [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
                       "MovieLens1M" =>   (2, 200, [20], [5, 10, 15, 20],
                                           [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
                       "LastFm2K" =>      (1, 200, [20], [5, 10, 15, 20],
                                           [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
                       "LastFm1K" =>      (1, 200, [20], [5, 10, 15, 20],
                                           [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
                       "EchoNest" =>      (1, 200, [20], [5, 10, 15, 20],
                                           [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
                       "LastFm360K" =>    (1, 200, [20], [5, 10, 15, 20],
                                           [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
                       "LastFm360K_2K" => (1, 200, [20], [5, 10, 15, 20],
                                           [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
                       "ML100KPos" =>     (3, 200, [20], [5, 10, 15, 20],
                                           [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))

dictExpoMF_SETTING = Dict("SmallToy" =>      (1, 100, [4], [1,2,3,5]),
                          "SmallToyML" =>    (2, 100, [4], [1,2,3,5]),
                          "ML50" =>          (2, 100, [4], [1,2,3,5]),
                          "MovieLens100K" => (2, 100, [20], [5, 10, 15, 20]),
                          "MovieLens1M" =>   (2, 100, [20], [5, 10, 15, 20]),
                          "LastFm2K" =>      (1, 100, [20], [5, 10, 15, 20]),
                          "LastFm1K" =>      (1, 100, [20], [5, 10, 15, 20]),
                          "EchoNest" =>      (1, 100, [20], [5, 10, 15, 20]),
                          "LastFm360K" =>    (1, 100, [20], [5, 10, 15, 20]),
                          "LastFm360K_2K" => (1, 100, [20], [5, 10, 15, 20]),
                          "ML100KPos" =>     (3, 100, [20], [5, 10, 15, 20]))

dictBPR_SETTING = Dict("SmallToy" =>      (1, 100, [4], [1,2,3,5], 0.1, 0.01),
                          "SmallToyML" =>    (2, 100, [4], [1,2,3,5], 0.1, 0.01),
                          "ML50" =>          (2, 100, [4], [1,2,3,5], 0.1, 0.01),
                          "MovieLens100K" => (2, 100, [20], [5, 10, 15, 20], 0.1, 0.01),
                          "MovieLens1M" =>   (2, 100, [20], [5, 10, 15, 20], 0.1, 0.01),
                          "LastFm2K" =>      (1, 100, [20], [5, 10, 15, 20], 0.1, 0.01),
                          "LastFm1K" =>      (1, 100, [20], [5, 10, 15, 20], 0.1, 0.01),
                          "EchoNest" =>      (1, 100, [20], [5, 10, 15, 20], 0.1, 0.01),
                          "LastFm360K" =>    (1, 100, [20], [5, 10, 15, 20], 0.1, 0.01),
                          "LastFm360K_2K" => (1, 100, [20], [5, 10, 15, 20], 0.1, 0.01),
                          "ML100KPos" =>     (3, 100, [20], [5, 10, 15, 20], 0.1, 0.01))

ENV_TYPE = Dict(1 =>"OSX", 2=>"Linux1", 3=>"Linux2", 4=>"Windows")

DATA_PATH = Dict("Linux1" => "/home/ian/Dataset/",
                 "Linux2" => "/home/aidesktop2/Dataset/",
                 "OSX" => "/Users/iankuoli/Dataset/",
                 "Windows" => "C:/dataset/",)

DATASET = Dict(1=>"SmallToy", 2=>"SmallToyML", 3=>"ML50",
               4=>"MovieLens100K", 5=>"MovieLens1M",
               6=>"LastFm2K", 7=>"LastFm1K", 8=>"EchoNest", 9=>"LastFm360K_2K", 10=>"LastFm360K",
               11=>"ML100KPos")

TRAIN_PATH = Dict("SmallToy" => "SmallToy_train.csv",
                  "SmallToyML" => "SmallToyML_train.csv",
                  "ML50" => "ML50_train.csv",
                  "MovieLens100K" => "MovieLens100K_train_v2.csv",
                  "MovieLens1M" => "MovieLens1M_train.csv",
                  "LastFm2K" => "LastFm2K_train.csv",
                  "LastFm1K" => "LastFm1K_train.csv",
                  "EchoNest" => "EchoNest_train.csv",
                  "LastFm360K_2K" => "LastFm360K_2K_train.csv",
                  "LastFm360K" => "LastFm360K_train.csv",
                  "ML100KPos" => "ml-100k/movielens-100k-train_original.txt")

TEST_PATH = Dict("SmallToy" => "SmallToy_test.csv",
                 "SmallToyML" => "SmallToyML_test.csv",
                 "ML50" => "ML50_test.csv",
                 "MovieLens100K" => "MovieLens100K_test_v2.csv",
                 "MovieLens1M" => "MovieLens1M_test.csv",
                 "LastFm2K" => "LastFm2K_test.csv",
                 "LastFm1K" => "LastFm1K_test.csv",
                 "EchoNest" => "EchoNest_test.csv",
                 "LastFm360K_2K" => "LastFm360K_2K_test.csv",
                 "LastFm360K" => "LastFm360K_test.csv",
                 "ML100KPos" => "ml-100k/movielens-100k-test_original.txt")

VALID_PATH = Dict("SmallToy" => "SmallToy_valid.csv",
                  "SmallToyML" => "SmallToyML_valid.csv",
                  "ML50" => "ML50_valid.csv",
                  "MovieLens100K" => "MovieLens100K_valid_v2.csv",
                  "MovieLens1M" => "MovieLens1M_valid.csv",
                  "LastFm2K" => "LastFm2K_valid.csv",
                  "LastFm1K" => "LastFm1K_valid.csv",
                  "EchoNest" => "EchoNest_valid.csv",
                  "LastFm360K_2K" => "LastFm360K_2K_valid.csv",
                  "LastFm360K" => "LastFm360K_valid.csv",
                  "ML100KPos" => "ml-100k/movielens-100k-valid_original.txt")

function Env_type(env_id::Int)
    return ENV_TYPE[env_id]
end

function Dataset(dataset_id::Int)
    return DATASET[dataset_id]
end

function Data_path(env_name::String, dataset::String)
    data_path = DATA_PATH[env_name]
    train_path = TRAIN_PATH[dataset]
    valid_path = VALID_PATH[dataset]
    test_path = TEST_PATH[dataset]

    return data_path * train_path, data_path * valid_path, data_path * test_path
end

function ExpSetting_HNBF(dataset::String)
    return dictHNBF_SETTING[dataset]
end

function ExpSetting_HPF(dataset::String)
    return dictHPF_SETTING[dataset]
end

function ExpSetting_ExpoMF(dataset::String)
    return dictExpoMF_SETTING[dataset]
end

function ExpSetting_BPR(dataset::String)
    return dictBPR_SETTING[dataset]
end
