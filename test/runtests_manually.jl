using Pkg
Pkg.activate(joinpath(dirname(@__FILE__),".."))
Pkg.test("MadamOpt"; test_args = ARGS)
