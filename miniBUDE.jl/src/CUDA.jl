include("BUDE.jl")
using CUDA, StaticArrays


function devices()::Vector{DeviceWithRepr}
  return !CUDA.functional(false) ? [] :
         map(d -> (d, "$(CUDA.name(d)) ($(repr(d)))", "CUDA.jl"), CUDA.devices())
end

function run(params::Params, deck::Deck, device::DeviceWithRepr)
  # show_reason is set to true here so it dumps CUDA info 
  # for us regardless of whether it's functional
  if !CUDA.functional(true)
    error("Non-functional CUDA configuration")
  end

  # so CUDA's device is 0 indexed, so -1 from Julia
  CUDA.device!(device[1])
  CUDA.math_mode!(CUDA.FAST_MATH)

  println("Using CUDA device: $(CUDA.name(device[1])) ($(repr(device[1])))")

  protein = CuArray{Atom}(deck.protein)
  ligand = CuArray{Atom}(deck.ligand)
  forcefield = CuArray{FFParams}(deck.forcefield)
  poses = CuArray{Float32,2}(deck.poses)

  nprotein::Int = length(deck.protein)
  nligand::Int = length(deck.ligand)
  nforcefield::Int = length(deck.forcefield)
  nposes::Int = size(deck.poses)[2]

  etotals = CuArray{Float32}(undef, nposes)
  nblocks = ceil(UInt, ceil(nposes / params.ppwi) / params.wgsize)
  nthreads = params.wgsize

  println("Using kernel parameters: <<<$(nblocks),$(nthreads)>>> 1:$nposes")

  shared = false
  shmem_bytes = shared ? sizeof(FFParams) * nforcefield : 0

  # warmup
  @cuda blocks = nblocks threads = nthreads shmem = shmem_bytes fasten_main(
    nprotein,
    nligand,
    nforcefield,
    nposes,
    protein,
    ligand,
    forcefield,
    poses,
    etotals,
    Val(shared),
    Val(convert(Int, params.ppwi)),
  )
  CUDA.synchronize()
  elapsed = @elapsed begin
    for _ = 1:params.iterations
      @cuda blocks = nblocks threads = nthreads shmem = shmem_bytes fasten_main(
        nprotein,
        nligand,
        nforcefield,
        nposes,
        protein,
        ligand,
        forcefield,
        poses,
        etotals,
        Val(shared),
        Val(convert(Int, params.ppwi)),
      )
    end
    CUDA.synchronize()
  end

  (Array{Float32}(etotals), elapsed, params.ppwi)
end

# const NUM_TD_PER_THREAD = 8

@fastmath function fasten_main(
  nprotein::Int,
  nligand::Int,
  nforcefield::Int,
  nposes::Int,
  protein_::CuDeviceVector{Atom},
  ligand_::CuDeviceVector{Atom},
  forcefield_::CuDeviceVector{FFParams},
  poses_::CuDeviceMatrix{Float32},
  etotals::CuDeviceVector{Float32},
  ::Val{Shared},
  ::Val{PPWI},
) where {Shared,PPWI}


  protein = CUDA.Const(protein_)
  ligand = CUDA.Const(ligand_)
  global_forcefield = CUDA.Const(forcefield_)
  poses = CUDA.Const(poses_)

  #  Get index of first TD
  ix = (blockIdx().x - 1) * blockDim().x * PPWI + threadIdx().x

  #  Have extra threads do the last member intead of return.
  #  A return would disable use of barriers, so not using return is better
  ix = ix <= nposes ? ix : nposes - PPWI

  etot = MArray{Tuple{PPWI},Float32}(undef)
  transform = MArray{Tuple{PPWI,3},Vec4f32}(undef)


  if Shared
    forcefield = @cuDynamicSharedMem(FFParams, (nforcefield))
    if ix < nforcefield
      @inbounds forcefield[ix] = global_forcefield[ix]
    end
  else
    forcefield = global_forcefield
  end



  lsz = blockDim().x
  @simd for i = 1:PPWI
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
    @inbounds etot[i] = 0
  end

  if Shared
    CUDA.sync_threads()
  end

  for il = 1:nligand

    @inbounds l_atom::Atom = ligand[il]
    @inbounds l_params::FFParams = forcefield[l_atom.type+1]
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

    for ip = 1:nprotein
      @inbounds p_atom = protein[ip]
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

  td_base = (blockIdx().x - 1) * blockDim().x * PPWI + threadIdx().x
  if td_base <= nposes
    @simd for i = 1:PPWI
      @inbounds etotals[td_base+(i-1)*blockDim().x] = etot[i] * Half
    end
  end

  nothing
end

main()
