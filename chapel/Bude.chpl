// Use Chapel argument parser
module Bude {
  use IO;
  use Time;
  use AutoMath;

  config param WGSIZE = 4,
    DEFAULT_ITERS = 8,
    DEFAULT_NPOSES = 65536,
    REF_NPOSES = 65536,
    DATA_DIR = "../data/bm1",
    FILE_LIGAND = "/ligand.in",
    FILE_PROTEIN = "/protein.in",
    FILE_FORCEFIELD = "/forcefield.in",
    FILE_POSES = "/poses.in",
    FILE_REF_ENERGIES = "/ref_energies.out",
    ATOM_SIZE = 16,
    FFPARAMS_SIZE = 16;

  // Energy evaluation parameters
  const CNSTNT: real(32) = 45.0;
  const HBTYPE_F: real(32) = 70.0;
  const HBTYPE_E: real(32) = 69.0;
  const HARDNESS: real(32) = 38.0;
  const NPNPDIST: real(32) = 5.5;
  const NPPDIST: real(32) = 1.0;

  const WORK_GROUP = 0..<WGSIZE;

  record atom {
    var x, y, z: real(32);
    var aType: int(32);
  }

  record ffParams {
    var hbtype: int(32);
    var radius: real(32);
    var hphb: real(32);
    var elsc: real(32);
  }

  record context {
    var iterations: int = DEFAULT_ITERS;
    var nposes: int = DEFAULT_NPOSES;
    var deckDir: string = DATA_DIR;

    // Domains for arrays
    var natlig: int;
    var ligandDom: domain(1);
    var natpro: int;
    var proteinDom: domain(1);
    var ntypes: int;
    var forcefieldDom: domain(1);
    var posesDom: domain(2);
    
    var ligand: [ligandDom] atom;
    var protein: [proteinDom] atom;
    var forcefield: [forcefieldDom] ffParams;
    var poses: [posesDom] real(32);

    proc init() { }

    proc init(args: [] string) {
      /* Load command-line arguments */
      const argc = args.size;
      var arg: string;

      var t_deckDir = DATA_DIR;

      var i = 1;
      while i < argc {
        arg = args[i];
        if arg == "--iterations" || arg == "-i" {
          if i + 1 >= argc || parseInt(this.iterations, args[i+1]) < 0 {
            writeln("Invalid number of iterations");
            exit(1);
          }
          i += 1;
        } else if arg == "--numposes" || arg == "-n" {
          if i + 1 >= argc || parseInt(this.nposes, args[i+1]) < 0 {
            writeln("Invalid number of poses");
            exit(1);
          }
          i += 1;
        } else if arg == "--help" || arg == "-h" {
          writeln("");
          writeln("Usage: ./bude [OPTIONS]");
          writeln("Options:");
          writeln("  -h  --help               Print this message");
          writeln("  -i  --iterations I       Repeat kernel I times (default: ", DEFAULT_ITERS, ")");
          writeln("  -n  --numposes   N       Compute energies for N poses (default: ", DEFAULT_NPOSES, ")");
          writeln("      --deck       DECK    Use the DECK directory as input deck (default: ", DATA_DIR, ")");
          writeln("");
          exit(0);
        } else if arg == "--deck" {
          if (i + 1 >= argc) {
            writeln("Invalid deck");
            exit(1);
          }
          t_deckDir = args[i + 1];
          i += 1;
        } else {
          writeln("Unrecognized argument '", arg, "' (try '--help')\n");
          exit(1);
        }
        i += 1;
      }

      this.deckDir = t_deckDir;
      var length: int;
      var aFile: file;

      /* init ligand array */
      aFile = openFile(this.deckDir, FILE_LIGAND, iomode.r, length);
      this.natlig = length / ATOM_SIZE;
      this.ligandDom = {0..(this.natlig-1)};

      /* init protein array */
      aFile = openFile(this.deckDir, FILE_PROTEIN, iomode.r, length);
      this.natpro = length / ATOM_SIZE;
      this.proteinDom = {0..(this.natpro-1)};

      /* init forcefield array */
      aFile = openFile(this.deckDir, FILE_FORCEFIELD, iomode.r, length); 
      this.ntypes = length / FFPARAMS_SIZE;
      this.forcefieldDom = {0..(this.ntypes-1)};

      /* init poses array */
      this.posesDom = { 0..5, (0..this.nposes-1) };
    }

    proc load() {
      var length: int;
      var aFile: file;
      
      /* load ligand */
      aFile = openFile(this.deckDir, FILE_LIGAND, iomode.r, length);
      loadData(aFile, this.ligand, ATOM_SIZE);

      /* load protein */
      aFile = openFile(this.deckDir, FILE_PROTEIN, iomode.r, length);
      loadData(aFile, this.protein, ATOM_SIZE);

      /* load forcefields */
      aFile = openFile(this.deckDir, FILE_FORCEFIELD, iomode.r, length);
      loadData(aFile, this.forcefield, FFPARAMS_SIZE);

      /* load poses */
      aFile = openFile(this.deckDir, FILE_POSES, iomode.r, length);
      var available = length / (6 * 4);
      var cur_poses = 0, fetch, address = 0;
      while (cur_poses < this.nposes) {
        fetch = this.nposes - cur_poses;
        if (fetch > available) {
          fetch = available;
        }
        address = 0; // rewind
        for i in 0..<6 {
          address = i * available * 4;
          for j in 0..(fetch-1) {
            loadDataPiece(aFile, this.poses(i, cur_poses+j), address, 4);
            address += 4;
          }
        }
        cur_poses += fetch;
      }

      this.nposes = cur_poses;
    }

    /* Load data from file to record array */
    proc loadData(aFile: file, ref A: [] ?t, size: int) {
      const n = A.size;
      var readChannel = try! aFile.reader(kind=iokind.native, region=0..n*size);
      try! readChannel.read(A);
      try! readChannel.close();
    }

    /* Load data piece */
    proc loadDataPiece(aFile: file, ref A: ?t, base: int, offset: int) {
      var r = try! aFile.reader(kind=iokind.native, region=base..base+offset);
      try! r.read(A);
      try! r.close();
    }

    /* Convert a string to integer */
    proc parseInt(ref x: int, s: string): int {
      try {
        x = s: int;
      } catch {
        return -1;
      }
      return x;
    }
  }

  proc openFile(parent: string, child: string, mode: iomode, ref length: int): file {
    const name = parent + child;
    var aFile: file;

    try {
      aFile = open(name, mode);
      length = aFile.size;
    } catch {
      try {
        stderr.writeln("Failed to open '", name, "'");
        exit(0);
      } catch {
        exit(0);
      }
    }

    return aFile;
  }

  var params: context;

  proc main(args: [] string) {
    params = new context(args);
    params.load();

    // Show meta-information
    writeln("");
    writeln("Poses     : ", params.nposes);
    writeln("Iterations: ", params.iterations);
    writeln("Ligands   : ", params.natlig);
    writeln("Proteins  : ", params.natpro);
    writeln("Deck      : ", params.deckDir);

    
    var energiesChapel: [0..<params.nposes] real(32);
    // Compute
    compute(energiesChapel);

    // Validate energies
    var length: int;
    const ref_energies = openFile(params.deckDir, FILE_REF_ENERGIES, iomode.r, length);
    var e: real(32);
    var diff: real(32);
    var maxdiff: real(32) = -100.0;
    var n_ref_poses = params.nposes;
    if (params.nposes > REF_NPOSES) {
      writeln("Only validating the first ", REF_NPOSES, " poses");
      n_ref_poses = REF_NPOSES;
    }

    var reader = try! ref_energies.reader();
    for i in 0..<n_ref_poses {
      try! reader.read(e);
      if (abs(e) < 1.0 && abs(energiesChapel(i)) < 1.0) {
        continue;
      }
      diff = abs(e - energiesChapel(i)) / e;
      if (diff > maxdiff) {
        maxdiff = diff;
      }
    }

    writef("\nLargest difference was %{.###}%%.\n\n", 100 * maxdiff);
  }


  proc compute(results: [] real(32)) {
    writeln("\nRunning Chapel");

    var buffer: [0..<params.nposes] real(32);    
    // Copy data
    var poses = params.poses;
    var protein = params.protein;
    var ligand = params.ligand;
    var forcefield = params.forcefield;

    // Warm-up
    forall group in 0..<params.nposes/WGSIZE {
      fasten_main(params.natlig, params.natpro, protein, ligand,
                  poses, buffer, forcefield, group);
    }

    // Core part of computing
    const start = timestamp();
    for itr in 0..<params.iterations {
      forall group in 0..<params.nposes / WGSIZE {
        fasten_main(params.natlig, params.natpro, protein, ligand, poses, buffer, forcefield, group);
      }
    }
    const end = timestamp();

    // Copy to result
    results = buffer;

    printTimings(start, end);
  }

  proc timestamp(): real(64) {
    return getCurrentTime(unit=TimeUnits.milliseconds);
  }

  proc printTimings(start: real(64), end: real(64)) {
    const ms = (end - start) / params.iterations;
    const runtime = ms * 1e-3;

    const ops_per_wg = WGSIZE * 27 + params.natlig * (2 + WGSIZE * 18 + params.natpro * (10 + WGSIZE * 30)) + WGSIZE;
    const total_ops = ops_per_wg * (params.nposes / WGSIZE);
    const flops = total_ops / runtime;
    const gflops = flops / 1e9;

    const total_finsts = 25.0 * params.natpro * params.natlig * params.nposes;
    const finsts = total_finsts / runtime;
    const gfinsts = finsts / 1e9;

    const interactions = 1.0 * params.nposes * params.natlig * params.natpro;
    const interactions_per_sec = interactions / runtime;

    // Print stats
    writef("- Total time:     %7.3dr ms\n", end - start);
    writef("- Average time:   %7.3dr ms\n", ms);
    writef("- Interactions/s: %7.3dr billion\n", (interactions_per_sec / 1e9));
    writef("- GFLOP/s:        %7.3dr\n", gflops);
    writef("- GFInst/s:       %7.3dr\n", gfinsts);
  }

  proc fasten_main(
    natlig: int,
    natpro: int,
    protein: [] atom,
    ligand: [] atom,
    transforms: [] real(32),
    results: [] real(32),
    forcefield: [] ffParams,
    group: int) {

    var transform: [0..<3, 0..<4, 0..<WGSIZE] real(32) = noinit;
    var etot: [0..<WGSIZE] real(32) = noinit;

    // Compute transformation matrix
    foreach i in WORK_GROUP {
      const ix = group*WGSIZE + i;
      const sx = sin(transforms(0, ix));
      const cx = cos(transforms(0, ix));
      const sy = sin(transforms(1, ix));
      const cy = cos(transforms(1, ix));
      const sz = sin(transforms(2, ix));
      const cz = cos(transforms(2, ix));
      transform(0, 0, i) = cy*cz;
      transform(0, 1, i) = sx*sy*cz - cx*sz;
      transform(0, 2, i) = cx*sy*cz + sx*sz;
      transform(0, 3, i) = transforms(3, ix);
      transform(1, 0, i) = cy*sz;
      transform(1, 1, i) = sx*sy*sz + cx*cz;      
      transform(1, 2, i) = cx*sy*sz - sx*cz;
      transform(1, 3, i) = transforms(4, ix);
      transform(2, 0, i) = -sy;
      transform(2, 1, i) = sx*cy;
      transform(2, 2, i) = cx*cy;
      transform(2, 3, i) = transforms(5, ix);

      etot[i] = 0.0;
    }
    
    foreach il in 0..<natlig {
      const l_atom = ligand[il];
      const l_params = forcefield[l_atom.aType];
      const lhphb_ltz = l_params.hphb < 0.0;
      const lhphb_gtz = l_params.hphb > 0.0;

      // Transform ligand atom
      var lpos_x: [0..WGSIZE] real(32) = noinit;
      var lpos_y: [0..WGSIZE] real(32) = noinit;
      var lpos_z: [0..WGSIZE] real(32) = noinit;

      foreach l in WORK_GROUP {
        lpos_x[l] = transform(0, 3, l)
          + l_atom.x * transform(0, 0, l)
          + l_atom.y * transform(0, 1, l)
          + l_atom.z * transform(0, 2, l);

        lpos_y[l] = transform(1, 3, l)
          + l_atom.x * transform(1, 0, l)
          + l_atom.y * transform(1, 1, l)
          + l_atom.z * transform(1, 2, l);

        lpos_z[l] = transform(2, 3, l)
          + l_atom.x * transform(2, 0, l)
          + l_atom.y * transform(2, 1, l)
          + l_atom.z * transform(2, 2, l);
      }

      foreach ip in 0..<natpro {
        const p_atom = protein(ip);
        const p_params = forcefield(p_atom.aType);

        const radij = p_params.radius + l_params.radius;
        const r_radij = 1.0 / radij;

        const elcdst = if
          p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F
          then 4.0: real(32)
          else 2.0: real(32);

        const elcdst1 = if
          p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F
          then 0.25: real(32)
          else 0.5: real(32);

        const type_E = p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E;
        const phphb_ltz = p_params.hphb <  0;
        const phphb_gtz = p_params.hphb >  0;
        const phphb_nz  = p_params.hphb != 0;

        const p_hphb = p_params.hphb 
          * if phphb_ltz && lhphb_gtz then -1.0: real(32) else 1.0: real(32);

        const l_hphb = l_params.hphb 
          * if phphb_gtz && lhphb_ltz then -1.0: real(32) else 1.0: real(32);

        const distdslv =
          if phphb_ltz
          then (
            if lhphb_ltz
            then NPNPDIST
            else NPPDIST
          ) else (
            if lhphb_ltz
            then NPPDIST
            else -max(real(32))
          );

        const r_distdslv = 1.0 / distdslv;
        const chrg_init = l_params.elsc * p_params.elsc;
        const dslv_init = p_hphb + l_hphb; 

        foreach l in WORK_GROUP {
          // Calculate distance between atoms
          const x = lpos_x(l) - p_atom.x;
          const y = lpos_y(l) - p_atom.y;
          const z = lpos_z(l) - p_atom.z;
          const distij = sqrt(x * x + y * y + z* z); 

          // Calculate the sum of the sphere radii
          const distbb = distij - radij;
          const zone1 = distbb < 0.0: real(32);

          // Calculate steric energy
          etot[l] += (1.0 - distij * r_radij)
            * if zone1 then 2.0: real(32) * HARDNESS else 0.0: real(32);

          // Calculate formal and dipole charge interactions
          var chrg_e =
            chrg_init * (
              if zone1 
              then 1.0: real(32)
                else 1.0: real(32) - distbb * elcdst1
            ) * (
              if distbb < elcdst 
              then 1.0: real(32)
              else 0.0: real(32)
            );
          
          var neg_chrg_e = -abs(chrg_e);
          chrg_e = if type_E then neg_chrg_e else chrg_e;
          etot[l] += chrg_e * CNSTNT;

          const coeff = 1.0 - distbb * r_distdslv;
          var dslv_e = dslv_init 
            * if distbb < distdslv && phphb_nz then 1.0: real(32) else 0.0: real(32);

          dslv_e *= if zone1 then 1.0: real(32) else coeff;
          etot[l] += dslv_e;
        }
      }
    }

    results[group * WGSIZE..<(group + 1) * WGSIZE] = 0.5 : real(32) * etot;
  }
}