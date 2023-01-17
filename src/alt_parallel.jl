using ChainRulesCore

struct ParallelPassenger{F, T}
    connection::F
    layers::T

    singleton::Bool
end

function ParallelPassenger(connection, layers; singleton=false)
    ParallelPassenger(connection, layers, singleton)
end

ParallelPassenger(connection) = Passenger(identity)

Flux.@functor ParallelPassenger (layers,)

(pp::ParallelPassenger)(rx::Tuple) = pp(rx...)
(pp::ParallelPassenger)(r, xs...) = pp(r, xs)

function (pp::ParallelPassenger)(r, xs::Tuple)
    if pp.singleton
        pp.connection(map(f -> f(r, xs), pp.layers)...)
    else
        _parallel_check(pp.layers, xs)
        pp.connection(map((f, x) -> f(r, x), pp.layers, xs)...) # TODO Switch everything to unsplatted
    end
end

function (pp::ParallelPassenger)(r, xs::Vector)
    if pp.singleton
        pp.connection(map(f -> f(r, xs), pp.layers)...)
    else
        _parallel_check(pp.layers, xs)
        pp.connection(map((f, x) -> f(r, x), pp.layers, xs)...)
    end
end

function _parallel_check(layers, xs)
    nl = length(layers)
    nx = length(xs) 
    if (nl != nx)
        throw(ArgumentError("Parallel with $nl sub-layers can take one input or $nl inputs, but got $nx inputs"))
    end
  end
ChainRulesCore.@non_differentiable _parallel_check(nl, nx)

Base.getindex(pp::ParallelPassenger, i) = pp.layers[i]
Base.getindex(pp::ParallelPassenger, i::AbstractVector) = ParallelPassenger(pp.connection, pp.layers[i])

function Base.show(io::IO, pp::ParallelPassenger)
    print(io, "ParallelPassenger(", pp.connection, ", ")
    _show_layers(io, pp.layers)
    print(io, ")")
end

_show_layers(io, layers::Tuple) = join(io, layers, ", ")
_show_layers(io, layers::NamedTuple) = join(io, ["$k = $v" for (k, v) in pairs(layers)], ", ")
_show_layers(io, layers::AbstractVector) = (print(io, "["); join(io, layers, ", "); print(io, "]"))
_show_layers(io, layers) = Base.show(io, layers)

"""
Syntactic sugar that behaves identically to Parallel(triv_connect, identity, p::Parallel).
That is, the first element of the tuple is left untouched.
Necessary for the self-interaction and nonlinear layers that make no use of position values (first argument).
"""
struct Passenger{P}
    p::P
end

Passenger(args...) = Passenger(Parallel(args...))

(pgr::Passenger)((r, xs...)::Tuple) = (r, pgr.p(xs...))
(pgr::Passenger)(r, xs) = (r, pgr.p(xs))
(pgr::Passenger)(r, xs...) = (r, pgr.p(xs...))

Flux.@functor Passenger