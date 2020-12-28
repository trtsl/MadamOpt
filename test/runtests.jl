#!/usr/bin/env julia

module TestHelpers

"""
Allows filtering tests at the commad line, e.g.:
```
julia -- runtests_manually.jl test_xyz
```
"""
macro testset_filtered(test_label, test)
    if length(ARGS) > 0
        regex = Regex.(ARGS, "i")
        quote
            if any(occursin.($regex, $test_label))
                # could also escape the outer `quote` but his is more precise
                @info("Running test: ", $test_label)
                esc(:(Test.@testset($$test_label, $$test)))
            else
                @info("Skpping test: ", $test_label)
            end
        end
    else
        esc(:(Test.@testset($test_label, $test)))
    end
end

otherFiles(fullPath::String)::Array{String,1} = begin
    dirPath  = dirname(fullPath)

    files = Array{String}(undef,0)

    for f in joinpath.(dirPath,readdir(dirPath))
        if isfile(f) && f!=fullPath && length(f)>2 && f[end-2:end]==".jl"
            push!(files,f)
        end
    end
    
    return files
end

testFiles(files::Array{String}; catch_errors::Bool=false, mod::Module=Main) = begin
    @info("If the tests segfault, try clearing julia's precompile cache at ~/.julia/compiled")
    println("The following files will be tested:")
    println.(" => ".*files)
    runTest(f) = begin
        # put each file in separate module to prevent namespace pollution
        # if Core.eval() is not called in @__MODULE__, then each temporary
        # module will be a submodule of the module including this file;
        Core.eval(mod, :( module $(Symbol("test_"*splitext(basename(f))[1]))
                print("\n\n")
                @info("*** Testing \"$($f)\" in temp module [$(@__MODULE__)]")
                if $catch_errors
                    try
                        include($f)
                    catch err
                        io = IOBuffer()
                        showerror(io,err)
                        err_msg = String(take!(copy(io)))
                        bt = catch_backtrace()
                        @warn("Caught error: $(typeof(err)) $err_msg $(sprint(io->Base.show_backtrace(io, bt)))")
                    end
                else
                    include($f)
                end
            end
        ) )
    end
    runTest.(files)
end

end # module

using MadamOpt, .TestHelpers

files_exclude = joinpath.(dirname(@__FILE__),["runtests_manually.jl"])

files = TestHelpers.otherFiles(@__FILE__)
files = filter(files) do f
    !in(f, files_exclude)
end
TestHelpers.testFiles(files; catch_errors=false)
