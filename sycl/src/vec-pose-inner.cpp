#include "bude.h"

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
#define NPNPDIST  5.5f
#define NPPDIST   1.0f


void fasten_main(
		clsycl::handler &h,
		size_t posesPerWI, size_t wgSize,
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
		clsycl::accessor<float, 1, RW, Global> etotals) {


	constexpr const auto FloatMax = std::numeric_limits<float>::max();


	size_t global = std::ceil((nposes) / static_cast<double> (posesPerWI));
	global = wgSize * std::ceil(static_cast<double> (global) / wgSize);

	clsycl::accessor<FFParams, 1, RW, Local> local_forcefield(clsycl::range<1>(ntypes), h);

	clsycl::accessor<float, 2, RW, Local> etot(clsycl::range<2>(wgSize, posesPerWI), h);
	clsycl::accessor<clsycl::float3, 2, RW, Local> lpos(clsycl::range<2>(wgSize, posesPerWI), h);
	clsycl::accessor<clsycl::float4, 3, RW, Local> transform(clsycl::range<3>(wgSize, posesPerWI, 3), h);

	h.parallel_for<class bude_kernel>(clsycl::nd_range<1>(global, wgSize), [=](clsycl::nd_item<1> item) {

		const size_t lid = item.get_local_id(0);
		const size_t gid = item.get_group(0);
		const size_t lrange = item.get_local_range(0);


		size_t ix = gid * lrange * posesPerWI + lid;
		ix = ix < nposes ? ix : nposes - posesPerWI;

		// TODO async_work_group_copy takes only gentypes, so no FFParams,
		//  casting *_ptr<ElementType> parameter requires first converting to void and then to gentype
		//  although probably free, there must be a better way of doing this
		clsycl::device_event event = item.async_work_group_copy<float>(
				clsycl::local_ptr<float>(clsycl::local_ptr<void>(local_forcefield.get_pointer())),
				clsycl::global_ptr<float>(clsycl::global_ptr<void>(forcefield.get_pointer())),
				ntypes * sizeof(FFParams) / sizeof(float));

		// Compute transformation matrix to private memory
		const size_t lsz = lrange;
		for (size_t i = 0; i < posesPerWI; i++) {
			size_t index = ix + i * lsz;

			const float sx = clsycl::sin(transforms_0[index]);
			const float cx = clsycl::cos(transforms_0[index]);
			const float sy = clsycl::sin(transforms_1[index]);
			const float cy = clsycl::cos(transforms_1[index]);
			const float sz = clsycl::sin(transforms_2[index]);
			const float cz = clsycl::cos(transforms_2[index]);

			transform[clsycl::id<3>(lid, i, 0)].x() = cy * cz;
			transform[clsycl::id<3>(lid, i, 0)].y() = sx * sy * cz - cx * sz;
			transform[clsycl::id<3>(lid, i, 0)].z() = cx * sy * cz + sx * sz;
			transform[clsycl::id<3>(lid, i, 0)].w() = transforms_3[index];
			transform[clsycl::id<3>(lid, i, 1)].x() = cy * sz;
			transform[clsycl::id<3>(lid, i, 1)].y() = sx * sy * sz + cx * cz;
			transform[clsycl::id<3>(lid, i, 1)].z() = cx * sy * sz - sx * cz;
			transform[clsycl::id<3>(lid, i, 1)].w() = transforms_4[index];
			transform[clsycl::id<3>(lid, i, 2)].x() = -sy;
			transform[clsycl::id<3>(lid, i, 2)].y() = sx * cy;
			transform[clsycl::id<3>(lid, i, 2)].z() = cx * cy;
			transform[clsycl::id<3>(lid, i, 2)].w() = transforms_5[index];

			etot[clsycl::id<2>(lid, i)] = ZERO;
		}

		item.wait_for(event);

		// Loop over ligand atoms
		size_t il = 0;
		do {
			// Load ligand atom data
			const Atom l_atom = ligand_molecule[il];
			const FFParams l_params = local_forcefield[l_atom.type];
			const bool lhphb_ltz = l_params.hphb < ZERO;
			const bool lhphb_gtz = l_params.hphb > ZERO;

			const clsycl::float4 linitpos(l_atom.x, l_atom.y, l_atom.z, ONE);
			for (size_t i = 0; i < posesPerWI; i++) {
				const clsycl::id<2> id(lid, i);
				const clsycl::id<3> i0(lid, i, 0);
				const clsycl::id<3> i1(lid, i, 1);
				const clsycl::id<3> i2(lid, i, 2);
				// Transform ligand atom
				lpos[id].x() = transform[i0].w() +
				               linitpos.x() * transform[i0].x() +
				               linitpos.y() * transform[i0].y() +
				               linitpos.z() * transform[i0].z();
				lpos[id].y() = transform[i1].w() +
				               linitpos.x() * transform[i1].x() +
				               linitpos.y() * transform[i1].y() +
				               linitpos.z() * transform[i1].z();
				lpos[id].z() = transform[i2].w() +
				               linitpos.x() * transform[i2].x() +
				               linitpos.y() * transform[i2].y() +
				               linitpos.z() * transform[i2].z();
			}

			// Loop over protein atoms
			size_t ip = 0;
			do {
				// Load protein atom data
				const Atom p_atom = protein_molecule[ip];
				const FFParams p_params = local_forcefield[p_atom.type];

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

				for (size_t i = 0; i < posesPerWI; i++) {
					const clsycl::id<2> id(lid, i);
					// Calculate distance between atoms

					const float x = lpos[id].x() - p_atom.x;
					const float y = lpos[id].y() - p_atom.y;
					const float z = lpos[id].z() - p_atom.z;
					const float distij = clsycl::native::sqrt(x * x + y * y + z * z);

					//TODO replace with:
					// const float distij = clsycl::distance(lpos[id], clsycl::float3(p_atom.x, p_atom.y, p_atom.z));

					// Calculate the sum of the sphere radii
					const float distbb = distij - radij;
					const bool zone1 = (distbb < ZERO);

					// Calculate steric energy
					etot[id] += (ONE - (distij * r_radij)) * (zone1 ? 2 * HARDNESS : ZERO);

					// Calculate formal and dipole charge interactions
					float chrg_e = chrg_init * ((zone1 ? 1 : (ONE - distbb * elcdst1)) * (distbb < elcdst ? 1 : ZERO));
					const float neg_chrg_e = -clsycl::fabs(chrg_e);
					chrg_e = type_E ? neg_chrg_e : chrg_e;
					etot[id] += chrg_e * CNSTNT;

					// Calculate the two cases for Nonpolar-Polar repulsive interactions
					const float coeff = (ONE - (distbb * r_distdslv));
					float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? 1 : ZERO);
					dslv_e *= (zone1 ? 1 : coeff);
					etot[id] += dslv_e;
				}
			} while (++ip < natpro); // loop over protein atoms
		} while (++il < natlig); // loop over ligand atoms

		// Write results
		const size_t td_base = gid * lrange * posesPerWI + lid;

		if (td_base < nposes) {
			for (size_t i = 0; i < posesPerWI; i++) {
				etotals[td_base + i * lrange] = etot[clsycl::id<2>(lid, i)] * HALF;
			}
		}

	});

}
