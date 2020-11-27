#include "bude.h"
#include <limits>
#include <cmath>

#define ZERO    0.0f
#define QUARTER 0.25f
#define HALF    0.5f
#define ONE     1.0f
#define TWO     2.0f
#define FOUR    4.0f
#define CNSTNT 45.0f

// Energy evaluation parameters
#define HBTYPE_F 70
#define HBTYPE_E 69
#define HARDNESS 38.0f
#define NPNPDIST 5.5f
#define NPPDIST  1.0f

constexpr const auto FloatMax = std::numeric_limits<float>::max();

void fasten_main(
		size_t group,
		size_t ntypes, size_t nposes,
		size_t natlig, size_t natpro,
		Kokkos::View<Atom *> protein_molecule,
		Kokkos::View<Atom *> ligand_molecule,
		Kokkos::View<float *> transforms_0,
		Kokkos::View<float *> transforms_1,
		Kokkos::View<float *> transforms_2,
		Kokkos::View<float *> transforms_3,
		Kokkos::View<float *> transforms_4,
		Kokkos::View<float *> transforms_5,
		Kokkos::View<FFParams *> forcefield,
		Kokkos::View<float *> etotals
) {

	float etot[WG_SIZE];
	float transform[3][4][WG_SIZE];


	// Compute transformation matrix to private memory
	for (size_t i = 0; i < WG_SIZE; i++) {

		size_t ix = group * WG_SIZE + i;

		const float sx = sinf(transforms_0[ix]);
		const float cx = cosf(transforms_0[ix]);
		const float sy = sinf(transforms_1[ix]);
		const float cy = cosf(transforms_1[ix]);
		const float sz = sinf(transforms_2[ix]);
		const float cz = cosf(transforms_2[ix]);

		transform[0][0][i] = cy * cz;
		transform[0][1][i] = sx * sy * cz - cx * sz;
		transform[0][2][i] = cx * sy * cz + sx * sz;
		transform[0][3][i] = transforms_3[ix];
		transform[1][0][i] = cy * sz;
		transform[1][1][i] = sx * sy * sz + cx * cz;
		transform[1][2][i] = cx * sy * sz - sx * cz;
		transform[1][3][i] = transforms_4[ix];
		transform[2][0][i] = -sy;
		transform[2][1][i] = sx * cy;
		transform[2][2][i] = cx * cy;
		transform[2][3][i] = transforms_5[ix];

		etot[i] = ZERO;
	}

	// Loop over ligand atoms
	size_t il = 0;
	do {
		// Load ligand atom data
		const Atom l_atom = ligand_molecule[il];
		const FFParams l_params = forcefield[l_atom.type];
		const bool lhphb_ltz = l_params.hphb < ZERO;
		const bool lhphb_gtz = l_params.hphb > ZERO;

		float lpos_x[WG_SIZE], lpos_y[WG_SIZE], lpos_z[WG_SIZE];
		for (size_t i = 0; i < WG_SIZE; i++) {
			// Transform ligand atom
			lpos_x[i] = transform[0][3][i]
					+ l_atom.x * transform[0][0][i]
					+ l_atom.y * transform[0][1][i]
					+ l_atom.z * transform[0][2][i];
			lpos_y[i] = transform[1][3][i]
					+ l_atom.x * transform[1][0][i]
					+ l_atom.y * transform[1][1][i]
					+ l_atom.z * transform[1][2][i];
			lpos_z[i] = transform[2][3][i]
					+ l_atom.x * transform[2][0][i]
					+ l_atom.y * transform[2][1][i]
					+ l_atom.z * transform[2][2][i];
		}

		// Loop over protein atoms
		size_t ip = 0;
		do {
			// Load protein atom data
			const Atom p_atom = protein_molecule[ip];
			const FFParams p_params = forcefield[p_atom.type];

			const float radij = p_params.radius + l_params.radius;
			const float r_radij = 1.f / (radij);

			const float elcdst = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? FOUR : TWO;
			const float elcdst1 = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) ? QUARTER : HALF;
			const bool type_E = ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

			const bool phphb_ltz = p_params.hphb < ZERO;
			const bool phphb_gtz = p_params.hphb > ZERO;
			const bool phphb_nz = p_params.hphb != ZERO;
			const float p_hphb = p_params.hphb * (phphb_ltz && lhphb_gtz ? -ONE : ONE);
			const float l_hphb = l_params.hphb * (phphb_gtz && lhphb_ltz ? -ONE : ONE);
			const float distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FloatMax));
			const float r_distdslv = 1.f / (distdslv);

			const float chrg_init = l_params.elsc * p_params.elsc;
			const float dslv_init = p_hphb + l_hphb;

			for (size_t i = 0; i < WG_SIZE; i++) {
				// Calculate distance between atoms
				const float x = lpos_x[i] - p_atom.x;
				const float y = lpos_y[i] - p_atom.y;
				const float z = lpos_z[i] - p_atom.z;

				const float distij = sqrtf(x * x + y * y + z * z);

				// Calculate the sum of the sphere radii
				const float distbb = distij - radij;
				const bool zone1 = (distbb < ZERO);

				// Calculate steric energy
				etot[i] += (ONE - (distij * r_radij)) * (zone1 ? 2 * HARDNESS : ZERO);

				// Calculate formal and dipole charge interactions
				float chrg_e = chrg_init * ((zone1 ? 1 : (ONE - distbb * elcdst1)) * (distbb < elcdst ? 1 : ZERO));
				const float neg_chrg_e = -fabsf(chrg_e);
				chrg_e = type_E ? neg_chrg_e : chrg_e;
				etot[i] += chrg_e * CNSTNT;

				// Calculate the two cases for Nonpolar-Polar repulsive interactions
				const float coeff = (ONE - (distbb * r_distdslv));
				float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? 1 : ZERO);
				dslv_e *= (zone1 ? 1 : coeff);
				etot[i] += dslv_e;
			}
		} while (++ip < natpro); // loop over protein atoms
	} while (++il < natlig); // loop over ligand atoms

	// Write results
	for (size_t l = 0; l < WG_SIZE; l++) {
		// Write result
		etotals[group * WG_SIZE + l] = etot[l] * 0.5f;
	}


}
