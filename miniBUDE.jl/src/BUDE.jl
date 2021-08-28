using ArgParse
using Parameters
using Printf
using Base: Float64, Float32, Int, Int32
import Base.read

const DefaultInterations = 8
const DefaultNPoses = 65536
const RefNPoses = 65536
const DefaultWGSize = 64
const DefaultPPWI = 4

const Zero = 0.0f0
const Quarter = 0.25f0
const Half = 0.5f0
const One = 1.0f0
const Two = 2.0f0
const Four = 4.0f0
const Cnstnt = 45.0f0

const HbtypeF = 70
const HbtypeE = 69
const Hardness = 38.0f0
const Npnpdist = 5.5f0
const Nppdist = 1.0f0

const Float32Max = typemax(Float32)

struct Vec3f32
  x::Float32
  y::Float32
  z::Float32
end

struct Vec4f32
  x::Float32
  y::Float32
  z::Float32
  w::Float32
end

struct Atom
  x::Float32
  y::Float32
  z::Float32
  type::Int32
end

function read(io::IO, ::Type{Atom})
  Atom(read(io, Float32), read(io, Float32), read(io, Float32), read(io, Int32))
end

struct FFParams
  hbtype::Int32
  radius::Float32
  hphb::Float32
  elsc::Float32
end

function read(io::IO, ::Type{FFParams})
  FFParams(read(io, Int32), read(io, Float32), read(io, Float32), read(io, Float32))
end

@with_kw mutable struct Params
  device::String = "1"
  list::Bool = false
  numposes::UInt = DefaultNPoses
  iterations::UInt = DefaultInterations
  wgsize::UInt = DefaultWGSize
  ppwi::UInt = DefaultPPWI
  deck::String = "../data/bm1"
end

struct Deck
  protein::AbstractArray{Atom}
  ligand::AbstractArray{Atom}
  forcefield::AbstractArray{FFParams}
  poses::AbstractArray{Float32,2}
end

const DeviceWithRepr = Tuple{Any,String,Any}

function parse_options(given::Params)
  s = ArgParseSettings()
  @add_arg_table s begin
    "--list", "-l"
    help = "List available devices"
    action = :store_true
    "--device", "-d"
    help = "Select device at DEVICE or the device name substring if DEVICE is not an integer, NOTE: Julia is 1-indexed"
    arg_type = String
    default = given.device
    "--numposes", "-n"
    help = "Compute energies for N poses"
    arg_type = Int
    default = convert(Int, given.numposes)
    "--iterations", "-i"
    help = "Repeat kernel I times"
    arg_type = Int
    default = convert(Int, given.iterations)
    "--wgsize", "-w"
    help = "Run with work-group size WGSIZE"
    arg_type = Int
    default = convert(Int, given.wgsize)
    "--ppwi", "-p"
    help = "Compute PPWI poses per work-item"
    arg_type = Int
    default = convert(Int, given.ppwi)
    "--deck"
    help = "Use the DECK directory as input deck"
    arg_type = String
    default = given.deck
  end
  args = parse_args(s)
  # surely there's a better way than doing this:
  for (arg, val) in args
    setproperty!(given, Symbol(arg), val)
  end
end

function read_structs(path::String, ::Type{T})::Vector{T} where {T}
  io = open(path, "r")
  size = filesize(io)
  [read(io, T) for _ = 1:size÷sizeof(T)]
end

function print_timings(params::Params, deck::Deck, elapsedSeconds::Float64, wgsize::UInt)

  # Average time per iteration
  averageIterationSeconds = elapsedSeconds / params.iterations

  natlig = length(deck.ligand)
  natpro = length(deck.protein)

  # Compute FLOP/s
  ops_per_wg = (wgsize * 27 + natlig * (2 + wgsize * 18 + natpro * (10 + wgsize * 30)) + wgsize)

  total_ops = ops_per_wg * (params.numposes / wgsize)
  flops = total_ops / averageIterationSeconds
  gflops = flops / 1.0e9

  total_finsts = 25.0 * natpro * natlig * params.numposes
  finsts = total_finsts / averageIterationSeconds
  gfinsts = finsts / 1e9

  interactions = params.numposes * natlig * natpro
  interactions_per_sec = interactions / averageIterationSeconds

  # Print stats
  @printf "- Kernel time:    %.03f ms\n" (elapsedSeconds * 1000.0)
  @printf "- Average time:   %.03f ms\n" (averageIterationSeconds * 1000.0)
  @printf "- Interactions/s: %.03f billion\n" interactions_per_sec / 1e9
  @printf "- GFLOP/s:        %.03f\n" gflops
  @printf "- GFInst/s:       %.03f\n" gfinsts
end

function main()

  params::Params = Params()
  parse_options(params)


  ds = devices()

  if params.list
    for (i, (_, name, type)) in enumerate(ds)
      println("[$i] ($type) $name")
    end
    exit(0)
  end

  deviceIndex = try
    index = parse(Int, params.device)
    if index < 1 || index > length(ds)
      error("Device $(index) out of range (1..$(length(ds))), NOTE: Julia is 1-indexed")
    end
    index
  catch
    index = findfirst(x -> occursin(params.device, "($(x[3])) $(x[2])"), ds)
    if index === nothing
      error("No device match the substring `$(params.device)`, see --list for available devices")
    end
    index
  end


  poses = permutedims( # reshape is column order so we flip it back again
    reshape(
      read_structs("$(params.deck)/poses.in", Float32),  # read poses as one long array
      (params.numposes, 6), # reshape it to 6 slices of numposes sized rows
    ),
  )

  deck = Deck(
    read_structs("$(params.deck)/protein.in", Atom),
    read_structs("$(params.deck)/ligand.in", Atom),
    read_structs("$(params.deck)/forcefield.in", FFParams),
    poses,
  )

  println("Poses     : ", size(deck.poses)[2])
  println("Iterations: ", params.iterations)
  println("Ligands   : ", length(deck.ligand))
  println("Protein   : ", length(deck.protein))
  println("Forcefield: ", length(deck.forcefield))
  println("Deck      : ", params.deck)
  println("WGsize    : ", params.wgsize)
  println("PPWI      : ", params.ppwi)

  set_zero_subnormals(true)
  GC.enable(false)

  (energies, rumtimeSeconds, ppwi) = run(params, deck, ds[deviceIndex])

  GC.enable(true)
  set_zero_subnormals(false)

  print_timings(params, deck, rumtimeSeconds, ppwi)

  output = open("energies.out", "w+")
  println("\nEnergies:")
  for i = 1:size(deck.poses)[2]
    @printf(output, "%7.2f\n", energies[i])
    if (i < 16)
      @printf("%7.2f\n", energies[i])
    end
  end

  ref_energies = open("$(params.deck)/ref_energies.out", "r")
  n_ref_poses = size(deck.poses)[2]
  if n_ref_poses > RefNPoses
    println("Only validating the first $(RefNPoses) poses")
    n_ref_poses = RefNPoses
  end

  maxdiff = 0.0
  for i = 1:n_ref_poses
    line = readline(ref_energies)
    if line == ""
      error("ran out of ref energies lines to verify")
    end
    e = parse(Float32, line)
    if (abs(e) < 1.0 && abs(energies[i]) < 1.0)
      continue
    end
    diff = abs(e - energies[i]) / e
    if (diff > 0.01)
      println("![$(i)] $(e) $(energies[i]) ~ $(diff)")
      maxdiff = diff
    end
  end

  @printf "Largest difference was %.03f%%.\n\n" 100 * maxdiff # Expect numbers to be accurate to 2 decimal places
  close(ref_energies)

end
