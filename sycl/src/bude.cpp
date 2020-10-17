#include <cmath>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <functional>
#include <fstream>

#include "bude.h"

typedef std::chrono::high_resolution_clock::time_point TimePoint;


struct Params {

	int natlig;
	int natpro;
	int ntypes;
	int nposes;

	std::vector<Atom> protein;
	std::vector<Atom> ligand;
	std::vector<FFParams> forcefield;

	std::array<std::vector<float>, 6> poses;

	int iterations;

	friend std::ostream &operator<<(std::ostream &os, const Params &params) {
		os <<
		   "natlig:     " << params.natlig << "\n" <<
		   "natpro:     " << params.natpro << "\n" <<
		   "ntypes:     " << params.ntypes << "\n" <<
		   "nposes:     " << params.nposes << "\n" <<
		   "iterations: " << params.iterations;
		return os;
	}

};


void fasten_main(int natlig,
                 int natpro,
                 const std::vector<Atom> &protein,
                 const std::vector<Atom> &ligand,
                 const std::vector<float> &transforms_0,
                 const std::vector<float> &transforms_1,
                 const std::vector<float> &transforms_2,
                 const std::vector<float> &transforms_3,
                 const std::vector<float> &transforms_4,
                 const std::vector<float> &transforms_5,
                 std::vector<float> &results,
                 const std::vector<FFParams> &forcefield,
                 int group);

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

	const auto readParam = [&args](size_t &current,
	                               const std::string &emptyMessage,
	                               const std::function<void(std::string)> &map) {
		if (current + 1 < args.size()) {
			current++;
			return map(args[current]);
		} else {
			std::cerr << emptyMessage << std::endl;
			std::exit(EXIT_FAILURE);
		}
	};

	for (size_t i = 0; i < args.size(); ++i) {
		const auto arg = args[i];
		if (arg == "--iterations" || arg == "-i") {
			readParam(i, "--iterations specified but no value was given", [&](const std::string &param) {
				auto iter = std::stoul(param);
				if (iter < 0) {
					std::cerr << "Invalid number of iterations: `" << param << "`" << std::endl;
					std::exit(EXIT_FAILURE);
				}
				params.iterations = iter;
			});
		} else if (arg == "--numposes" || arg == "-n") {
			readParam(i, "--numposes specified but no value was given ", [&](const std::string &param) {
				auto nps = std::stoul(param);
				if (nps < 0) {
					std::cerr << "Invalid number of poses: `" << param << "`" << std::endl;
					std::exit(EXIT_FAILURE);
				}
				params.nposes = nps;
			});
		} else if (arg == "--help" || arg == "-h") {
			std::cout << "\n";
			std::cout << "Usage: ./bude [OPTIONS]\n\n"
			          << "Options:\n"
			          << "  -h  --help               Print this message\n"
			          << "  -i  --iterations I       Repeat kernel I times (default: " << DEFAULT_ITERS << ")\n"
			          << "  -n  --numposes   N       Compute energies for N poses (default: " << DEFAULT_NPOSES << ")"
			          << std::endl;
			std::exit(EXIT_SUCCESS);
		} else {
			std::cout << "Unrecognized argument '" << arg << "' (try '--help')" << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}

	params.ligand = readNStruct<Atom>(FILE_LIGAND);
	params.natlig = params.ligand.size();


	params.protein = readNStruct<Atom>(FILE_PROTEIN);
	params.natpro = params.protein.size();

	params.forcefield = readNStruct<FFParams>(FILE_FORCEFIELD);
	params.ntypes = params.forcefield.size();


	auto poses = readNStruct<float>(FILE_POSES);
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

	printf("\nRunning C/OpenMP\n");

	std::vector<float> energies(params.nposes);

	auto start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel
	for (int itr = 0; itr < params.iterations; itr++) {

		#pragma omp for
		for (int group = 0; group < (params.nposes / WGSIZE / PPWI); group++) {
			fasten_main(params.natlig, params.natpro,
			            params.protein, params.ligand,
			            params.poses[0], params.poses[1], params.poses[2],
			            params.poses[3], params.poses[4], params.poses[5],
			            energies, params.forcefield, group);
		}
	}

	auto end = std::chrono::high_resolution_clock::now();

	printTimings(params, start, end, PPWI);
	return energies;
}

int main(int argc, char *argv[]) {

	auto args = std::vector<std::string>(argv + 1, argv + argc);
	auto params = loadParameters(args);

	std::cout << "Workgroup Size: " << WGSIZE << std::endl;
	std::cout << "Parameters:\n" << params << std::endl;


	auto energies = runKernel(params);

	//XXX Keep the output format consistent with the C impl. so no fancy streams here
	FILE *output = fopen("energies.dat", "w+");
	// Print some energies
	printf("\nEnergies\n");
	for (int i = 0; i < params.nposes; i++) {
		fprintf(output, "%7.2f\n", energies[i]);
		if (i < 16)
			printf("%7.2f\n", energies[i]);
	}

	fclose(output);
}
