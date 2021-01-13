#include <cmath>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include <CL/sycl.hpp>

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

	// XXX bring this back once all SYCL implementations implement 2020 spec
//	size_t posesPerWI;
	size_t wgSize;
	std::string deckDir;

	clsycl::device device;

	friend std::ostream &operator<<(std::ostream &os, const Params &params) {
		os <<
		   "natlig:      " << params.natlig << "\n" <<
		   "natpro:      " << params.natpro << "\n" <<
		   "ntypes:      " << params.ntypes << "\n" <<
		   "nposes:      " << params.nposes << "\n" <<
		   "iterations:  " << params.iterations << "\n" <<
		   "posesPerWI:  " << NUM_TD_PER_THREAD << "\n" <<
		   "wgSize:      " << params.wgSize << "\n" <<
		   "SYCL device: " << params.device.get_info<clsycl::info::device::name>();
		return os;
	}
};

void fasten_main(
		clsycl::handler &h,
//		size_t posesPerWI,
		size_t wgSize,
		size_t ntypes, size_t nposes,
		size_t natlig, size_t natpro,
		clsycl::accessor<Atom, 1, R, Global> protein_molecule,
		clsycl::accessor<Atom, 1, R, Global> ligand_molecule,
		clsycl::accessor<float, 1, R, Global> transforms_0,
		clsycl::accessor<float, 1, R, Global> transforms_1,
		clsycl::accessor<float, 1, R, Global> transforms_2,
		clsycl::accessor<float, 1, R, Global> transforms_3,
		clsycl::accessor<float, 1, R, Global> transforms_4,
		clsycl::accessor<float, 1, R, Global> transforms_5,
		clsycl::accessor<FFParams, 1, R, Global> forcefield,
		clsycl::accessor<float, 1, RW, Global> etotals);


double elapsedMillis( const TimePoint &start, const TimePoint &end){
	auto elapsedNs = static_cast<double>(
			std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
	return elapsedNs * 1e-6;
}

void printTimings(const Params &params, double millis) {

	// Average time per iteration
	double ms = (millis / params.iterations);
	double runtime = ms * 1e-3;

	// Compute FLOP/s
	double ops_per_wg = NUM_TD_PER_THREAD * 27 +
			params.natlig * (2 +
					NUM_TD_PER_THREAD * 18 +
					params.natpro * (10 + NUM_TD_PER_THREAD * 30)
			) + NUM_TD_PER_THREAD;
	double total_ops = ops_per_wg * ((double) params.nposes / NUM_TD_PER_THREAD);
	double flops = total_ops / runtime;
	double gflops = flops / 1e9;

	double total_finsts = 25.0 * params.natpro * params.natlig * params.nposes;
	double finsts = total_finsts / runtime;
	double gfinsts = finsts / 1e9;

	double interactions = (double) params.nposes * (double) params.natlig * (double) params.natpro;
	double interactions_per_sec = interactions / runtime;

	// Print stats
	std::cout.precision(3);
	std::cout << std::fixed;
	std::cout << "- Kernel time:    " << (millis) << " ms\n";
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

std::string deviceName(clsycl::info::device_type type) {
	//@formatter:off
	switch (type){
		case clsycl::info::device_type::cpu: return "cpu";
		case clsycl::info::device_type::gpu: return "gpu";
		case clsycl::info::device_type::accelerator: return "accelerator";
		case clsycl::info::device_type::custom: return "custom";
		case clsycl::info::device_type::automatic: return "automatic";
		case clsycl::info::device_type::host: return "host";
		case clsycl::info::device_type::all: return "all";
		default: return "(unknown: " + std::to_string(static_cast<unsigned int >(type))+ ")";
	}
	//@formatter:on
}

void printSimple(const clsycl::device &device, size_t index) {
	std::cout << std::setw(3) << index << ". "
	          << device.get_info<clsycl::info::device::name>()
	          << "(" << deviceName(device.get_info<clsycl::info::device::device_type>()) << ")"
	          << std::endl;
}

Params loadParameters(const std::vector<std::string> &args) {

	Params params = {};

	// Defaults
	params.iterations = DEFAULT_ITERS;
	params.nposes = DEFAULT_NPOSES;
	params.wgSize = DEFAULT_WGSIZE;
	params.deckDir = DATA_DIR;
//	params.posesPerWI = DEFAULT_PPWI;

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

	const auto &devices = clsycl::device::get_devices();
	if (devices.empty()) {
		std::cerr << "No SYCL devices available!" << std::endl;
		std::exit(EXIT_FAILURE);
	} else {
		std::cout << "Available SYCL devices:" << std::endl;

		for (size_t j = 0; j < devices.size(); ++j) printSimple(devices[j], j);
	}

	params.device = devices[0];

	for (size_t i = 0; i < args.size(); ++i) {
		using namespace std::placeholders;
		const auto arg = args[i];
		if (readParam(i, arg, {"--iterations", "-i"}, std::bind(bindInt, _1, std::ref(params.iterations), "iterations"))) continue;
		if (readParam(i, arg, {"--numposes", "-n"}, std::bind(bindInt, _1, std::ref(params.nposes), "numposes"))) continue;
//		if (readParam(i, arg, {"--posesperwi", "-p"}, std::bind(bindInt, _1, std::ref(params.posesPerWI), "posesperwi"))) continue;
		if (readParam(i, arg, {"--wgsize", "-w"}, std::bind(bindInt, _1, std::ref(params.wgSize), "wgsize"))) continue;
		if (readParam(i, arg, {"--device", "-d"}, [&](const std::string &param) {

			auto devices = clsycl::device::get_devices();

			try { params.device = devices.at(std::stoul(param)); }
			catch (const std::exception &e) {
				std::cout << "Unable to parse/select device index `" << param << "`:" << e.what() << std::endl;
				std::cout << "Attempting to match device with substring  `" << param << "`" << std::endl;

				auto matching = std::find_if(devices.begin(), devices.end(), [param](const clsycl::device &device) {
					return device.get_info<clsycl::info::device::name>().find(param) != std::string::npos;
				});
				if (matching != devices.end()) {
					params.device = *matching;
					std::cout << "Using first device matching substring `" << param << "`" << std::endl;
				}
				else if(devices.size() == 1)
					std::cerr << "No matching device but there's only one device, will be using that anyway"<< std::endl;
				else {
					std::cerr << "No matching devices" << std::endl;
					std::exit(EXIT_FAILURE);
				}
			}
		})) { continue; }
		if (readParam(i, arg, {"--deck"}, [&](const std::string &param) { params.deckDir = param; })) continue;


		if (arg == "--list" || arg == "-l") {
			for (size_t j = 0; j < devices.size(); ++j) printSimple(devices[j], j);
			std::exit(EXIT_SUCCESS);
		}

		if (arg == "--help" || arg == "-h") {
			std::cout << "\n";
			std::cout << "Usage: ./bude [OPTIONS]\n\n"
			          << "Options:\n"
			          << "  -h  --help               Print this message\n"
			          << "  -i  --iterations I       Repeat kernel I times (default: " << DEFAULT_ITERS << ")\n"
			          << "  -n  --numposes   N       Compute energies for N poses (default: " << DEFAULT_NPOSES << ")\n"
//			          << "  -p  --poserperwi PPWI    Compute PPWI poses per work-item (default: " << DEFAULT_PPWI << ")\n"
					  << "  -w  --wgsize     WGSIZE  Run with work-group size WGSIZE using nd_range, set to 0 for plain range (default: " << DEFAULT_WGSIZE << ")\n"
			          << "  -d  --device     INDEX   Select device at INDEX from output of --list, "
			                                         "performs a substring match of device names if INDEX is not an integer (default: first device of the list)\n"
			          << "      --deck       DECK    Use the DECK directory as input deck (default: " << DATA_DIR << ")"
			          << "  -l  --list               List available devices"
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

template<typename T>
static clsycl::buffer<T> mkBuffer(clsycl::queue &queue, const std::vector< T> &xs ) {
	clsycl::buffer<T> buffer(xs.size());
	queue.submit([&](clsycl::handler &h) {
		h.copy(xs.data(), buffer.template get_access<RW>(h));
	});
	return buffer;
}

std::vector<float> runKernel(Params params) {

	std::vector<float> energies(params.nposes);
	auto contextStart = std::chrono::high_resolution_clock::now();
	clsycl::queue queue(params.device);
	auto contextEnd = std::chrono::high_resolution_clock::now();

	auto xferAllocStart = std::chrono::high_resolution_clock::now();

	auto protein = mkBuffer(queue, params.protein);
	auto ligand = mkBuffer(queue, params.ligand);
	auto transforms_0 = mkBuffer(queue, params.poses[0]);
	auto transforms_1 = mkBuffer(queue, params.poses[1]);
	auto transforms_2 = mkBuffer(queue, params.poses[2]);
	auto transforms_3 = mkBuffer(queue, params.poses[3]);
	auto transforms_4 = mkBuffer(queue, params.poses[4]);
	auto transforms_5 = mkBuffer(queue, params.poses[5]);
	auto forcefield = mkBuffer(queue, params.forcefield);

	clsycl::buffer<float> results(energies.size());
	queue.submit([&](clsycl::handler &h) {
		h.fill(results.get_access<W>(h), 0.f);
	});

	queue.wait();
	auto xferAllocEnd = std::chrono::high_resolution_clock::now();

	const auto runKernel = [&]() {
		queue.submit([&](clsycl::handler &h) {
			fasten_main(h,
					params.wgSize,
					params.ntypes, params.nposes,
					params.natlig, params.natpro,
					protein.get_access<R>(h),
					ligand.get_access<R>(h),
					transforms_0.get_access<R>(h),
					transforms_1.get_access<R>(h),
					transforms_2.get_access<R>(h),
					transforms_3.get_access<R>(h),
					transforms_4.get_access<R>(h),
					transforms_5.get_access<R>(h),
					forcefield.get_access<R>(h),
					results.get_access<RW>(h)
			);
		});
	};

	auto warmupStart = std::chrono::high_resolution_clock::now();
	runKernel(); //warm up
	queue.wait();
	auto warmupEnd = std::chrono::high_resolution_clock::now();

	auto kernelStart = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < params.iterations; ++i) {
		runKernel();
	}
	queue.wait();
	auto kernelEnd = std::chrono::high_resolution_clock::now();

	clsycl::buffer<float> buffer(energies.data(), energies.size());
	queue.submit([&](clsycl::handler &h) { h.copy(results.get_access<RW>(h), buffer.get_access<RW>(h)); });
	queue.wait();

	std::cout
			<< "Context time:    " << elapsedMillis(contextStart, contextEnd) << " ms\n"
			<< "Xfer+Alloc time: " << elapsedMillis(xferAllocStart, xferAllocEnd) << " ms\n"
			<< "Warmup time:     " << elapsedMillis(warmupStart, warmupEnd) << " ms\n" << std::endl;

	printTimings(params, elapsedMillis(kernelStart, kernelEnd));
	return energies;
}

int main(int argc, char *argv[]) {

	auto args = std::vector<std::string>(argv + 1, argv + argc);
	auto params = loadParameters(args);

//	std::cout << "Parameters:\n" << params << std::endl;
	std::cout << "Device    : " << params.device.get_info<clsycl::info::device::name>() << std::endl;
	std::cout << "\tType    : " << deviceName(params.device.get_info<clsycl::info::device::device_type>()) << std::endl;
	std::cout << "\tProfile : " << params.device.get_info<clsycl::info::device::profile>() << std::endl;
	std::cout << "\tVersion : " << params.device.get_info<clsycl::info::device::version>() << std::endl;
	std::cout << "\tVendor  : " << params.device.get_info<clsycl::info::device::vendor>() << std::endl;
	std::cout << "\tDriver  : " << params.device.get_info<clsycl::info::device::driver_version>() << std::endl;
	std::cout << "Poses     : " << params.nposes << std::endl;
	std::cout << "Iterations: " << params.iterations << std::endl;
	std::cout << "Ligands   : " << params.natlig << std::endl;
	std::cout << "Proteins  : " << params.natpro << std::endl;
	std::cout << "Deck      : " << params.deckDir << std::endl;
	std::cout << "WG        : " << params.wgSize << std::boolalpha << " (use nd_range:" << (params.wgSize > 0) << ")" << std::endl;
	auto energies = runKernel(params);

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
		if (diff > maxdiff) maxdiff = diff;
	}
	std::cout << "Largest difference was " <<
	          std::setprecision(3) << (100 * maxdiff)
	          << "%.\n\n"; // Expect numbers to be accurate to 2 decimal places
	refEnergies.close();

	fclose(output);
}
