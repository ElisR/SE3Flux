using ChainRulesCore

struct ParallelPassenger{F, T}
    connection::F
    layers::T
end

# allow ParallelPassenger(op, m1, m2, ...) as a constructor
#ParallelPassenger(connection, layers...) = ParallelPassenger(connection, layers)
#ParallelPassenger(connection, (layer,)) = layer
#function ParallelPassenger(connection, layers::Tuple)
#    if isempty(layers)
#        return Passenger(identity)
#    else
#        return ParallelPassenger(connection, layers)
#    end
#end
ParallelPassenger(connection) = Passenger(identity)

Flux.@functor ParallelPassenger

# TODO Fix this part. (May already be fixed)
# This is currently not working, so will rewrite to be more like original
#=
(pp::ParallelPassenger)((r, y)::Tuple) = pp(r, y)
(pp::ParallelPassenger)(r, xs::Tuple) = pp.connection(map((f, x) -> f(r, x), Tuple(pp.layers), xs)...)
(pp::ParallelPassenger)(r, (x,)) = pp.connection(map(f -> f(r, x), Tuple(pp.layers))...) # This

(pp::ParallelPassenger)(r, xs...) = pp(r, xs)
# New vector implementation. Unexpected behaviour, but what I need in E3ConvLayer
(pp::ParallelPassenger)(r, xs::Vector) = pp.connection(map(f -> f(r, xs...), Tuple(pp.layers))...)
=#

(pp::ParallelPassenger)(rx::Tuple) = pp(rx...)
(pp::ParallelPassenger)(r, xs::Tuple) = pp(r, xs...)

(pp::ParallelPassenger)(r, x) = pp.connection(map(f -> f(r, x), pp.layers)...)
function (pp::ParallelPassenger)(r, xs...)
    _parallel_check(pp.layers, xs)
    pp.connection(map((f, x) -> f(r, x), pp.layers, xs)...)
end
# Just added this / might be wrong
function (pp::ParallelPassenger)(r, xs::Vector)
    _parallel_check(pp.layers, xs)
    pp.connection(map((f, x) -> f(r, x), pp.layers, Tuple(xs))...)
end

# TODO This is not currently used
function _parallel_check(layers, xs)
    nl = length(layers)
    nx = length(xs) 
    if (nl != nx)
      throw(ArgumentError("Parallel with $nl sub-layers can take one input or $nl inputs, but got $nx inputs"))
    end
  end
ChainRulesCore.@non_differentiable _parallel_check(nl, nx)

# OG vector implementation
#=
function (pp::ParallelPassenger)(r, xs::Vector)
    if pp.treat_vec_singleton
        return pp.connection(map(f -> f(r, xs...), Tuple(pp.layers))...)
    else
        return pp.connection(map((f, x) -> f(r, x), pp.layers, xs)...)
    end
end
=#

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