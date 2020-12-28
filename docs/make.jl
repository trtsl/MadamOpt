using Pkg
Pkg.activate(joinpath(dirname(@__FILE__),"."))
Pkg.develop(path=joinpath(dirname(@__FILE__),".."))
using Documenter, MadamOpt

makedocs(
    modules = [MadamOpt],
    repo = "https://github.com/trtsl/MadamOpt.git",
    # when running search queries locally with default settings, the browser
    # will open `docs/search` as a directory and submit the query string to the
    # directory itself (which fails); running on a server, the browser does not
    # access the directory and will instead submit the query to
    # `docs/search/index.html` (assuming canonical server configuration); to
    # perform searches locally (e.g. when testing the docs), we can instruct
    # `Documenter` to locate the relevant query form at `docs/search.html` by
    # using the `prettyurls` argument i.e.:
    # `julia -- ./make.jl --local  && firefox --private-window -- ./build/index.html`
    # see:
    # https://en.wikipedia.org/wiki/Webserver_directory_index
    # https://juliadocs.github.io/Documenter.jl/stable/lib/internals/writers/#Documenter.Writers.HTMLWriter.HTML
    format = Documenter.HTML(prettyurls = !in("--local", ARGS)),
    sitename = "MadamOpt.jl",
    pages = [
        "Home" => "index.md",
    ]
)