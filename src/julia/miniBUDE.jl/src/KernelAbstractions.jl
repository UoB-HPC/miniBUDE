include("BUDE.jl")
using ROCKernels, CUDAKernels, KernelAbstractions, StaticArrays, CUDA, AMDGPU

@enum Backend cuda rocm cpu


function list_rocm_devices()::Vector{DeviceWithRepr}
  try
    # AMDGPU.agents()'s internal iteration order isn't stable
    sorted = sort(AMDGPU.get_agents(:gpu), by = repr)
    map(x -> (x, repr(x), rocm), sorted)
  catch
    # probably unsupported
    []
  end
end

function list_cuda_devices()::Vector{DeviceWithRepr}
  return !CUDA.functional(false) ? [] :
         map(d -> (d, "$(CUDA.name(d)) ($(repr(d)))", cuda), CUDA.devices())
end

function devices()::Vector{DeviceWithRepr}
  cudas = list_cuda_devices()
  rocms = list_rocm_devices()
  cpus = [(undef, "$(Sys.cpu_info()[1].model) ($(Threads.nthreads())T)", cpu)]
  vcat(cpus, cudas, rocms)
end

function run(params::Params, deck::Deck, device::DeviceWithRepr)


  nposes::Int64 = size(deck.poses)[2]
  blocks_size::Int64 = ceil(Int64, nposes / params.ppwi)
  nthreads::Int64 = params.wgsize

  (selected, _, backend) = device
  println("KernelAbstractions Backend: $backend")

  if backend == cpu
    println("Using CPU with max $(Threads.nthreads()) threads")
    protein = deck.protein
    ligand = deck.ligand
    forcefield = deck.forcefield
    poses = deck.poses
    etotals = Array{Float32}(undef, nposes)
    backend_impl = CPU()
  elseif backend == cuda
    CUDA.device!(selected)
    if CUDA.device() != selected
      error("Cannot select CUDA device, expecting $selected, but got $(CUDA.device())")
    end
    if !CUDA.functional(true)
      error("Non-functional CUDA configuration")
    end
    println("Using CUDA device: $(CUDA.name(selected)) ($(repr(selected)))")

    protein = CuArray{Atom}(deck.protein)
    ligand = CuArray{Atom}(deck.ligand)
    forcefield = CuArray{FFParams}(deck.forcefield)
    poses = CuArray{Float32,2}(deck.poses)
    etotals = CuArray{Float32}(undef, nposes)
    backend_impl = CUDADevice()
  elseif backend == rocm
    AMDGPU.DEFAULT_AGENT[] = selected
    if AMDGPU.get_default_agent() != selected
      error("Cannot select HSA device, expecting $selected, but got $(AMDGPU.get_default_agent())")
    end
    println("Using GPU HSA device: $(AMDGPU.get_name(selected)) ($(repr(selected)))")

    protein = ROCArray{Atom}(deck.protein)
    ligand = ROCArray{Atom}(deck.ligand)
    forcefield = ROCArray{FFParams}(deck.forcefield)
    poses = ROCArray{Float32,2}(deck.poses)
    etotals = ROCArray{Float32}(undef, nposes)
    backend_impl = ROCDevice()
  else
    error("unsupported backend $(backend)")
  end

  println("Using kernel parameters: <<<$blocks_size,$nthreads>>> 1:$nposes")


  kernel! = fasten_main(backend_impl, nthreads, blocks_size)

  wait(
    kernel!(
      nthreads,
      nposes,
      protein,
      ligand,
      forcefield,
      poses,
      etotals,
      Val(convert(Int, params.ppwi)),
      ndrange = blocks_size,
    ),
  )

  elapsed = @elapsed for i = 1:params.iterations
    wait(
      kernel!(
        nthreads,
        nposes,
        protein,
        ligand,
        forcefield,
        poses,
        etotals,
        Val(convert(Int, params.ppwi)),
        ndrange = blocks_size,
      ),
    )
  end

  (Array{Float32}(etotals), elapsed, params.ppwi)
end

@fastmath @kernel function fasten_main(
  wgsize::Int64,
  nposes::Int64,
  @Const(protein),
  @Const(ligand),
  @Const(forcefield),
  @Const(poses),
  etotals,
  ::Val{PPWI},
) where {PPWI}

  gid = @index(Group)
  lid = @index(Local)

  #  Get index of first TD
  ix = (gid - 1) * wgsize * PPWI + lid

  #  Have extra threads do the last member intead of return.
  #  A return would disable use of barriers, so not using return is better
  ix = ix <= nposes ? ix : nposes - PPWI

  etot = MArray{Tuple{PPWI},Float32}(undef)
  transform = MArray{Tuple{PPWI,3},Vec4f32}(undef)

  lsz = wgsize
  for i = 1:PPWI
    index = (ix) + (i - 1) * lsz
    @inbounds sx::Float32 = sin(poses[1, index])
    @inbounds cx::Float32 = cos(poses[1, index])
    @inbounds sy::Float32 = sin(poses[2, index])
    @inbounds cy::Float32 = cos(poses[2, index])
    @inbounds sz::Float32 = sin(poses[3, index])
    @inbounds cz::Float32 = cos(poses[3, index])
    @inbounds transform[i, 1] = #
      Vec4f32(cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz, poses[4, index])
    @inbounds transform[i, 2] = #
      Vec4f32(cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz, poses[5, index])
    @inbounds transform[i, 3] = #
      Vec4f32(-sy, sx * cy, cx * cy, poses[6, index])
    etot[i] = 0
  end

  @inbounds for l_atom::Atom in ligand

    l_params::FFParams = forcefield[l_atom.type+1]
    lhphb_ltz::Bool = l_params.hphb < Zero
    lhphb_gtz::Bool = l_params.hphb > Zero

    lpos = MArray{Tuple{PPWI},Vec3f32}(undef)

    @simd for i = 1:PPWI
      # Transform ligand atom
      @inbounds lpos[i] = Vec3f32(
        transform[i, 1].w +
        l_atom.x * transform[i, 1].x +
        l_atom.y * transform[i, 1].y +
        l_atom.z * transform[i, 1].z,
        transform[i, 2].w +
        l_atom.x * transform[i, 2].x +
        l_atom.y * transform[i, 2].y +
        l_atom.z * transform[i, 2].z,
        transform[i, 3].w +
        l_atom.x * transform[i, 3].x +
        l_atom.y * transform[i, 3].y +
        l_atom.z * transform[i, 3].z,
      )
    end

    @inbounds for p_atom::Atom in protein
      @inbounds p_params::FFParams = forcefield[p_atom.type+1]

      radij::Float32 = p_params.radius + l_params.radius
      r_radij::Float32 = One / radij

      elcdst::Float32 = (p_params.hbtype == HbtypeF && l_params.hbtype == HbtypeF) ? Four : Two
      elcdst1::Float32 = (p_params.hbtype == HbtypeF && l_params.hbtype == HbtypeF) ? Quarter : Half
      type_E::Bool = ((p_params.hbtype == HbtypeE || l_params.hbtype == HbtypeE))

      phphb_ltz::Bool = p_params.hphb < Zero
      phphb_gtz::Bool = p_params.hphb > Zero
      phphb_nz::Bool = p_params.hphb != Zero
      p_hphb::Float32 = p_params.hphb * (phphb_ltz && lhphb_gtz ? -One : One)
      l_hphb::Float32 = l_params.hphb * (phphb_gtz && lhphb_ltz ? -One : One)
      distdslv::Float32 =
        (phphb_ltz ? (lhphb_ltz ? Npnpdist : Nppdist) : (lhphb_ltz ? Nppdist : -Float32Max))
      r_distdslv::Float32 = One / distdslv

      chrg_init::Float32 = l_params.elsc * p_params.elsc
      dslv_init::Float32 = p_hphb + l_hphb


      @simd for i = 1:PPWI
        @inbounds x::Float32 = lpos[i].x - p_atom.x
        @inbounds y::Float32 = lpos[i].y - p_atom.y
        @inbounds z::Float32 = lpos[i].z - p_atom.z

        distij::Float32 = sqrt(x ^ 2 + y ^ 2 + z ^ 2)

        # Calculate the sum of the sphere radii
        distbb::Float32 = distij - radij
        zone1::Bool = (distbb < Zero)

        # Calculate steric energy
        @inbounds etot[i] += (One - (distij * r_radij)) * (zone1 ? Two * Hardness : Zero)

        # Calculate formal and dipole charge interactions
        chrg_e::Float32 =
          chrg_init * ((zone1 ? One : (One - distbb * elcdst1)) * (distbb < elcdst ? One : Zero))
        neg_chrg_e::Float32 = -abs(chrg_e)
        chrg_e = type_E ? neg_chrg_e : chrg_e
        @inbounds etot[i] += chrg_e * Cnstnt

        # Calculate the two cases for Nonpolar-Polar repulsive interactions
        coeff::Float32 = (One - (distbb * r_distdslv))
        dslv_e::Float32 = dslv_init * ((distbb < distdslv && phphb_nz) ? One : Zero)
        dslv_e *= (zone1 ? One : coeff)
        @inbounds etot[i] += dslv_e
      end
    end
  end

  td_base = (gid - 1) * wgsize * PPWI + lid
  if td_base <= nposes
    @simd for i = 1:PPWI
      @inbounds etotals[td_base+(i-1)*wgsize] = etot[i] * Half
    end
  end

  nothing
end

main()
