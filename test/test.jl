using Test
using Random
using Distributions
using SparseArrays

include("../src/LogisticRegression.jl")
lr = LogisticRegression
@testset "basic_functionality" begin
    @test lr.sigmoid(3) == 0.9525741268224334
    @test lr.sigmoid(.5) == 0.6224593312018546
    buncha_vals = [2,3,.5,1]
    res = lr.par_sigmoid(buncha_vals)
    @test res ==  [ 0.8807970779778823,
                    0.9525741268224334,
                    0.6224593312018546,
                    0.7310585786300049 ]
end
@testset "logistic regression" begin
    num_instances = 5000
    # generate dense training data
    x1_v1 = rand(Normal(0, 2), num_instances)
    x1_v2 = rand(Normal(0, .3), num_instances)
    x2_v1 = rand(Normal(4, 1), num_instances)
    x2_v2 = rand(Normal(2, .3), num_instances)
    
    x1 = [x1_v1 x1_v2]
    x2 = [x2_v1 x2_v2]
    data = [x1 ; x2]
    labels = [zeros(num_instances) ; ones(num_instances)]

    # generate dense test data
    x3_v1 = rand(Normal(0, 2), num_instances)
    x3_v2 = rand(Normal(0, .3), num_instances)
    x4_v1 = rand(Normal(4, 1), num_instances)
    x4_v2 = rand(Normal(2, .3), num_instances)
    x3 = [x3_v1 x3_v2]
    x4 = [x4_v1 x4_v2]
    test_data = [x3 ; x4]
    test_labels = [zeros(num_instances) ; ones(num_instances)]

    model = lr.LogisticRegressor(add_intercept=true)
    lr.fit!(model, data, labels)
    test_preds = lr.predict(model, test_data)
    acc = sum(test_preds .== test_labels)/length(test_labels)
    @test acc > .95 # this should be very easy to do well
    # since the distributions have exactly the same properties for
    # test and train and there's minimal overlap in classes
end
@testset "sparse logistic regression" begin
    num_instances = 5000
    # generate sparse training data
    x1_v1 = sparse(rand(Binomial(4, .2), num_instances))
    x1_v2 = sparse(rand(Binomial(8, .1),
    num_instances))
    x2_v1 = sparse(rand(Binomial(4, .8), num_instances))
    x2_v2 = sparse(rand(Binomial(8, .7), num_instances))
    
    x1 = [x1_v1 x1_v2]
    x2 = [x2_v1 x2_v2]
    data = [x1 ; x2]
    labels = [zeros(num_instances) ; ones(num_instances)]

    # generate sparse test data
    x3_v1 = sparse(rand(Binomial(4, .2), num_instances))
    x3_v2 = sparse(rand(Binomial(8, .1),
    num_instances))
    x4_v1 = sparse(rand(Binomial(4, .8), num_instances))
    x4_v2 = sparse(rand(Binomial(8, .7), num_instances))
    x3 = [x3_v1 x3_v2]
    x4 = [x4_v1 x4_v2]
    test_data = [x3 ; x4]
    test_labels = [zeros(num_instances) ; ones(num_instances)]

    model = lr.LogisticRegressor(add_intercept=true)
    lr.fit!(model, data, labels)
    test_preds = lr.predict(model, test_data)
    acc = sum(test_preds .== test_labels)/length(test_labels)
    @test acc > .95 # this should be very easy to do well
    # since the distributions have exactly the same properties for
    # test and train and there's minimal overlap in classes
end
