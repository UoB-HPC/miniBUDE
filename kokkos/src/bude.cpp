#include <cmath>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>

#include "bude.h"

typedef std::chrono::high_resolution_clock::time_point TimePoint;

struct Params {

	size_t natlig;
	size_t natpro;
	size_t ntypes;
	size_t nposes;

	std::vector<Atom> protein;
	std::vector<Atom> ligand;
	std::vector<FFParams> forcefield;
	std::array<std::vector<float>, 6> poses;

	size_t iterations;

	std::string deckDir;

	friend std::ostream &operator<<(std::ostream &os, const Params &params) {
		os <<
				"natlig:      " << params.natlig << "\n" <<
				"natpro:      " << params.natpro << "\n" <<
				"ntypes:      " << params.ntypes << "\n" <<
				"nposes:      " << params.nposes << "\n" <<
				"iterations:  " << params.iterations << "\n" <<
				"wgSize:      " << WG_SIZE << "\n";
		return os;
	}
};

void fasten_main(
		size_t ntypes, size_t nposes,
		size_t natlig, size_t natpro,
		const Kokkos::View<const Atom *> &protein_molecule,
		const Kokkos::View<const Atom *> &ligand_molecule,
		const Kokkos::View<const float *> &transforms_0,
		const Kokkos::View<const float *> &transforms_1,
		const Kokkos::View<const float *> &transforms_2,
		const Kokkos::View<const float *> &transforms_3,
		const Kokkos::View<const float *> &transforms_4,
		const Kokkos::View<const float *> &transforms_5,
		const Kokkos::View<const FFParams *> &forcefield,
		const Kokkos::View<float *> &etotals
);

void printTimings(const Params &params, const TimePoint &start, const TimePoint &end, double poses_per_wi) {

	auto elapsedNs = static_cast<double>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
	double ms = ((elapsedNs) / params.iterations) * 1e-6;

	// Compute FLOP/s
	double runtime = ms * 1e-3;
	double ops_per_wi = 27 * poses_per_wi
			+ params.natlig * (3 + 18 * poses_per_wi + params.natpro * (11 + 30 * poses_per_wi))
			+ poses_per_wi;
	double total_ops = ops_per_wi * (params.nposes / poses_per_wi);
	double flops = total_ops / runtime;
	double gflops = flops / 1e9;

	double total_finsts = 25.0 * params.natpro * params.natlig * params.nposes;
	double finsts = total_finsts / runtime;
	double gfinsts = finsts / 1e9;

	double interactions =
			(double) params.nposes
					* (double) params.natlig
					* (double) params.natpro;
	double interactions_per_sec = interactions / runtime;

	// Print stats
	std::cout.precision(3);
	std::cout << std::fixed;
	std::cout << "- Total time:     " << (elapsedNs * 1e-6) << " ms\n";
	std::cout << "- Average time:   " << ms << " ms\n";
	std::cout << "- Interactions/s: " << (interactions_per_sec / 1e9) << " billion\n";
	std::cout << "- GFLOP/s:        " << gflops << "\n";
	std::cout << "- GFInst/s:       " << gfinsts << "\n";
}

template<typename T>
std::vector<T> readNStruct(const std::string &path) {
	std::fstream s(path, std::ios::binary | std::ios::in);
	if (!s.good()) {
		throw std::invalid_argument("Bad file: " + path);
	}
	s.ignore(std::numeric_limits<std::streamsize>::max());
	auto len = s.gcount();
	s.clear();
	s.seekg(0, std::ios::beg);
	std::vector<T> xs(len / sizeof(T));
	s.read(reinterpret_cast<char *>(xs.data()), len);
	s.close();
	return xs;
}


Params loadParameters(const std::vector<std::string> &args) {

	Params params = {};

	// Defaults
	params.iterations = DEFAULT_ITERS;
	params.nposes = DEFAULT_NPOSES;
	params.deckDir = DATA_DIR;

	const auto readParam = [&args](size_t &current,
			const std::string &arg,
			const std::initializer_list<std::string> &matches,
			const std::function<void(std::string)> &handle) {
		if (matches.size() == 0) return false;
		if (std::find(matches.begin(), matches.end(), arg) != matches.end()) {
			if (current + 1 < args.size()) {
				current++;
				handle(args[current]);
			} else {
				std::cerr << "[";
				for (const auto &m : matches) std::cerr << m;
				std::cerr << "] specified but no value was given" << std::endl;
				std::exit(EXIT_FAILURE);
			}
			return true;
		}
		return false;
	};

	const auto bindInt = [](const std::string &param, size_t &dest, const std::string &name) {
		try {
			auto parsed = std::stol(param);
			if (parsed < 0) {
				std::cerr << "positive integer required for <" << name << ">: `" << parsed << "`" << std::endl;
				std::exit(EXIT_FAILURE);
			}
			dest = parsed;
		} catch (...) {
			std::cerr << "malformed value, integer required for <" << name << ">: `" << param << "`" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	};


	for (size_t i = 0; i < args.size(); ++i) {
		using namespace std::placeholders;
		const auto arg = args[i];
		if (readParam(i, arg, {"--iterations", "-i"}, std::bind(bindInt, _1, std::ref(params.iterations), "iterations"))) continue;
		if (readParam(i, arg, {"--numposes", "-n"}, std::bind(bindInt, _1, std::ref(params.nposes), "numposes"))) continue;
		if (readParam(i, arg, {"--deck"}, [&](const std::string &param) { params.deckDir = param; })) continue;

		if (arg == "--help" || arg == "-h") {
			std::cout << "\n";
			std::cout << "Usage: ./bude [OPTIONS]\n\n"
					<< "Options:\n"
					<< "  -h  --help               Print this message\n"
					<< "  -i  --iterations I       Repeat kernel I times (default: " << DEFAULT_ITERS << ")\n"
					<< "  -n  --numposes   N       Compute energies for N poses (default: " << DEFAULT_NPOSES << ")\n"
					<< "      --deck       DECK    Use the DECK directory as input deck (default: " << DATA_DIR << ")"
					<< std::endl;
			std::exit(EXIT_SUCCESS);
		}

		std::cout << "Unrecognized argument '" << arg << "' (try '--help')" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	params.ligand = readNStruct<Atom>(params.deckDir + FILE_LIGAND);
	params.natlig = params.ligand.size();

	params.protein = readNStruct<Atom>(params.deckDir + FILE_PROTEIN);
	params.natpro = params.protein.size();

	params.forcefield = readNStruct<FFParams>(params.deckDir + FILE_FORCEFIELD);
	params.ntypes = params.forcefield.size();

	auto poses = readNStruct<float>(params.deckDir + FILE_POSES);
	if (poses.size() / 6 != params.nposes) {
		throw std::invalid_argument("Bad poses: " + std::to_string(poses.size()));
	}

	for (size_t i = 0; i < 6; ++i) {
		params.poses[i].resize(params.nposes);
		std::copy(
				std::next(poses.cbegin(), i * params.nposes),
				std::next(poses.cbegin(), i * params.nposes + params.nposes),
				params.poses[i].begin());

	}

	return params;
}

std::vector<float> runKernel(Params params) {

	std::vector<float> energies(params.nposes);



	Kokkos::View<const Atom *> protein(params.protein.data(), params.protein.size());
	Kokkos::View<const Atom *> ligand(params.ligand.data(), params.ligand.size());
	Kokkos::View<const float *> transforms_0(params.poses[0].data(), params.poses[0].size());
	Kokkos::View<const float *> transforms_1(params.poses[1].data(), params.poses[1].size());
	Kokkos::View<const float *> transforms_2(params.poses[2].data(), params.poses[2].size());
	Kokkos::View<const float *> transforms_3(params.poses[3].data(), params.poses[3].size());
	Kokkos::View<const float *> transforms_4(params.poses[4].data(), params.poses[4].size());
	Kokkos::View<const float *> transforms_5(params.poses[5].data(), params.poses[5].size());
	Kokkos::View<const FFParams *> forcefield(params.forcefield.data(), params.forcefield.size());
	Kokkos::View<float *> results(energies.data(), energies.size());

	const auto runKernel = [&](){
		fasten_main(
				params.ntypes, params.nposes,
				params.natlig, params.natpro,
				protein, ligand,
				transforms_0, transforms_1, transforms_2,
				transforms_3, transforms_4, transforms_5,
				forcefield, results);
	};

	// warm up
	runKernel();
	Kokkos::fence();

	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < params.iterations; ++i) {
		runKernel();
	}
	Kokkos::fence();
	auto end = std::chrono::high_resolution_clock::now();

	printTimings(params, start, end, 1);
	return energies;
}

int main(int argc, char *argv[]) {


	auto args = std::vector<std::string>(argv + 1, argv + argc);
	auto params = loadParameters(args);

//	std::cout << "Parameters:\n" << params << std::endl;
	std::cout << "Poses     : " << params.nposes << std::endl;
	std::cout << "Iterations: " << params.iterations << std::endl;
	std::cout << "Ligands   : " << params.natlig << std::endl;
	std::cout << "Proteins  : " << params.natpro << std::endl;
	std::cout << "Deck      : " << params.deckDir << std::endl;
	std::cout << "WG_SIZE   : " << WG_SIZE << std::endl;

	Kokkos::initialize(argc, argv);

	if (Kokkos::hwloc::available()) {
		std::cout << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
				<< "] x CORE[" << Kokkos::hwloc::get_available_cores_per_numa()
				<< "] x HT[" << Kokkos::hwloc::get_available_threads_per_core() << "] )"
				<< std::endl;
	}
#if defined(KOKKOS_ENABLE_CUDA)
	Kokkos::Cuda::print_configuration(std::cout);
#endif

	auto energies = runKernel(params);

	Kokkos::finalize();

	//XXX Keep the output format consistent with the C impl. so no fancy streams here
	FILE *output = fopen("energies.out", "w+");
	// Print some energies
	printf("\nEnergies\n");
	for (size_t i = 0; i < params.nposes; i++) {
		fprintf(output, "%7.2f\n", energies[i]);
		if (i < 16)
			printf("%7.2f\n", energies[i]);
	}

	// Validate energies
	std::ifstream refEnergies(params.deckDir + FILE_REF_ENERGIES);
	size_t nRefPoses = params.nposes;
	if (params.nposes > REF_NPOSES) {
		std::cout << "Only validating the first " << REF_NPOSES << " poses.\n";
		nRefPoses = REF_NPOSES;
	}

	std::string line;
	float maxdiff = 0.0f;
	for (size_t i = 0; i < nRefPoses; i++) {
		if (!std::getline(refEnergies, line)) {
			throw std::logic_error("ran out of ref energies lines to verify");
		}
		float e = std::stof(line);
		if (std::fabs(e) < 1.f && std::fabs(energies[i]) < 1.f) continue;


		float diff = std::fabs(e - energies[i]) / e;
//		std::cout <<  "" << i << " = "<< diff << " " << "\n";

		if (diff > maxdiff) maxdiff = diff;
	}
	std::cout << "Largest difference was " <<
			std::setprecision(3) << (100 * maxdiff)
			<< "%.\n\n"; // Expect numbers to be accurate to 2 decimal places
	refEnergies.close();

	fclose(output);
}
