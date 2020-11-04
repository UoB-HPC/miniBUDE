#include "catch2/catch.hpp"
#include "input-utils.h"

#include <filesystem>
#include <variant>
#include <iostream>

namespace fs = std::filesystem;

template<class... Ts>
struct overloaded : Ts ... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

TEST_CASE("parser") {

	const fs::path testPath = "./samples";

	std::vector<fs::path> testFiles;
	for (const auto &entry : fs::directory_iterator(testPath))
		testFiles.push_back(entry);


	auto chooseExt = [testFiles](const std::string &ext) {
		std::vector<fs::path> xs;
		std::copy_if(testFiles.begin(), testFiles.end(), std::back_inserter(xs),
		             [ext](const auto &a) { return a.extension().string() == ext; });
		return xs;
	};
	std::vector<fs::path> bhffs = chooseExt(".bhff");
	std::vector<fs::path> mol2s = chooseExt(".mol2");


	SECTION("can parse bhff") {
		for (const fs::path &bhff : bhffs) {
			SECTION(bhff.string()) {
				std::cout << bhff.string() << std::endl;
				REQUIRE(!bude::readForceField(bhff, false).empty());
			}
		}
	} //

	SECTION("can parse mol2") {


		std::vector<std::pair<fs::path, bude::BudeForceField>> ffs;
		std::transform(bhffs.begin(), bhffs.end(), std::back_inserter(ffs),
		               [](const auto &f) { return std::make_pair(f, bude::readForceField(f, false)); });

		for (const fs::path &mol2 : mol2s) {
			typedef std::variant<std::string, size_t> Result;

			// parsing mol2 requires some valid forcefield input, so we test all available
			SECTION(mol2.string()) {
				std::cout << mol2.string() << std::endl;
				std::vector<std::pair<fs::path, Result>> results;

				// the idea here is that if none of the forcefield file is compatible with the mol2, we got a test failure
				std::transform(ffs.begin(), ffs.end(), std::back_inserter(results), [&](const auto &ff) {
					try {
						return std::make_pair(ff.first, Result{bude::readMol2(mol2, ff.second, false).first.size()});
					} catch (const std::exception &e) {
						return std::make_pair(ff.first, Result{std::string(e.what())});
					}
				});

				// we need at least 1 non-empty result to pass
				auto someFFPassed = std::any_of(results.begin(), results.end(), [&](const auto &r) {
					return std::visit(overloaded{
							[](size_t arg) { return arg != 0; },
							[](std::string &) { return false; },
							[](auto) { return false; }
					}, r.second);
				});

				// if we do fail, list out why for each forcefield-mol2 combination
				if (!someFFPassed) {
					for (auto &&[bhff, r] : results) {
						std::visit(overloaded{
								[&](size_t arg) { UNSCOPED_INFO(bhff << " : " << arg); },
								[&](std::string &e) { UNSCOPED_INFO(bhff << " : " << e); },
						}, r);
					}
				}
				REQUIRE(someFFPassed);
			}

		}

	} //

}


TEST_CASE("poses is valid") {

	std::vector<float> oneRadian{180.f / M_PI};
	std::vector<float> one{1.f};

	bude::Pose<std::vector<float>> expected = {
			one, one, one,
			one, one, one,
	};

	auto actual = bude::generatePoses(1, 42, {
			oneRadian, oneRadian, oneRadian,
			one, one, one,
	});

	REQUIRE(expected.fields() == actual.fields());

}
