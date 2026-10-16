# Reagents (rule-based modifier)

The rule-based modifier works only at the reagent level. It is a flat list of reagents; reaction families play no part. Each reagent carries its effect on the four target properties. At run time IGNITE compares the seed's predicted properties with the targets, ranks reagents whose effects close the gap, and picks the reaction site at random (seeded) among that reagent's valid sites.

Effects were taken from the Direction column of [reaction_families.md](reaction_families.md); the reaction SMARTS live in `backend/app/services/reaction_primitives.py` under the same reagent id. The code list is `backend/app/services/reagents.py`; a unit test keeps this table, that list, and the primitive deltas in sync.

62 reagents.

| Reagent | Edit | density | det_velocity | det_pressure | hf_solid |
| --- | --- | --- | --- | --- | --- |
| `nitrate_arom_CH` | Aromatic C–H → C–NO2 | raise | raise | raise | neutral |
| `nitrate_alkyl_CH` | Aliphatic C–H → C–NO2 | raise | raise | raise | neutral |
| `nitramine_NH` | Amine N–H → nitramine N–NO2 | raise | raise | raise | raise |
| `methyl_to_nitromethyl` | Ar–CH3 → Ar–CH2–NO2 | neutral | raise | raise | neutral |
| `halogen_Br_to_nitro` | Ar–Br → Ar–NO2 | lower | raise | raise | raise |
| `nitrate_ester_OH` | Alcohol O–H → nitrate ester O–NO2 | raise | raise | neutral | neutral |
| `gem_dinitro` | Aliphatic CH2 → gem-dinitro C(NO2)2 | raise | raise | raise | raise |
| `trinitromethyl` | Methyl → trinitromethyl C(NO2)3 | raise | raise | neutral | raise |
| `nitroso_arom` | Aromatic C–H → C–NO | neutral | lower | raise | raise |
| `azide_arom` | Aromatic C–H → C–N3 | neutral | neutral | lower | raise |
| `azide_methyl` | Ar–CH3 → Ar–CH2–N3 | neutral | raise | lower | raise |
| `amino_to_azide` | Ar–NH2 → Ar–N3 | lower | neutral | neutral | raise |
| `halogen_I_to_azide` | Ar–I → Ar–N3 | lower | raise | raise | raise |
| `tetrazole_CH` | Aromatic C–H → C–tetrazole | raise | neutral | raise | raise |
| `triazole_CH` | Aromatic C–H → C–(1,2,4-triazole) | neutral | neutral | raise | raise |
| `furazan_CH` | Aromatic C–H → C–furazan (1,2,5-oxadiazole) | raise | raise | raise | raise |
| `furoxan_CH` | Aromatic C–H → C–furoxan | raise | raise | neutral | raise |
| `tetrazine_CH` | Aromatic C–H → C–(1,2,4,5-tetrazine) | neutral | raise | raise | raise |
| `oxadiazole_CH` | Aromatic C–H → C–(1,3,4-oxadiazole) | raise | neutral | neutral | raise |
| `pyrazole_CH` | Aromatic C–H → C–pyrazole | neutral | neutral | neutral | raise |
| `imidazole_CH` | Aromatic C–H → C–imidazole | neutral | neutral | raise | neutral |
| `triazine_CH` | Aromatic C–H → C–(1,3,5-triazine) | neutral | raise | neutral | raise |
| `nitrile_to_tetrazole` | Nitrile → 5-substituted tetrazole | raise | raise | raise | neutral |
| `n_oxide` | Aromatic N → N-oxide | neutral | raise | raise | raise |
| `hydroxyl_CH` | Aromatic C–H → C–OH | raise | lower | raise | lower |
| `methoxy_CH` | Aromatic C–H → C–OCH3 | raise | neutral | lower | neutral |
| `carboxyl_CH` | Aromatic C–H → C–COOH | neutral | neutral | raise | lower |
| `methylester_CH` | Aromatic C–H → C–COOCH3 | raise | neutral | lower | lower |
| `amino_to_nitro` | Ar–NH2 → Ar–NO2 | neutral | raise | raise | lower |
| `hydroperoxy_CH` | Aromatic C–H → C–OOH | raise | neutral | neutral | raise |
| `epoxidize_CC` | Aliphatic C=C → epoxide | raise | neutral | raise | neutral |
| `formyl_CH` | Aromatic C–H → C–CHO | neutral | lower | neutral | lower |
| `cyano_CH` | Aromatic C–H → C–CN | raise | neutral | raise | raise |
| `hydrazino_CH` | Aromatic C–H → C–NHNH2 | raise | neutral | lower | raise |
| `amino_CH` | Aromatic C–H → C–NH2 | neutral | lower | lower | raise |
| `guanidino_CH` | Aromatic C–H → C–guanidine | neutral | raise | neutral | raise |
| `amide_CH` | Aromatic C–H → C–CONH2 | neutral | lower | raise | lower |
| `isocyanate_CH` | Aromatic C–H → C–NCO | neutral | raise | lower | neutral |
| `fluoro_arom_CH` | Aromatic C–H → C–F | raise | raise | raise | lower |
| `trifluoromethyl_CH` | Aromatic C–H → C–CF3 | raise | raise | neutral | lower |
| `chloro_arom_CH` | Aromatic C–H → C–Cl | raise | raise | lower | lower |
| `bromo_arom_CH` | Aromatic C–H → C–Br | raise | raise | neutral | neutral |
| `iodo_arom_CH` | Aromatic C–H → C–I | raise | neutral | neutral | neutral |
| `difluoramino_CH` | Aromatic C–H → C–NF2 | raise | raise | raise | neutral |
| `fluoro_alkyl` | Aliphatic C–H → C–F | raise | neutral | raise | lower |
| `dehydrogenate_CC` | C(sp3)–C(sp3) → C=C | neutral | raise | lower | raise |
| `alkyne_CC` | Acyclic C–C → C≡C | neutral | raise | neutral | raise |
| `add_phenyl_CH` | Aromatic C–H → C–Ph | neutral | neutral | raise | neutral |
| `benzyl_CH` | Aromatic C–H → C–CH2Ph | neutral | lower | raise | neutral |
| `remove_nitro_arom` | Remove aromatic nitro | lower | lower | lower | neutral |
| `remove_bromo_arom` | Remove aromatic Br | lower | neutral | neutral | neutral |
| `remove_azide_arom` | Remove aromatic azide | neutral | neutral | neutral | lower |
| `nitro_to_methyl_arom` | Ar–NO2 → Ar–CH3 | lower | lower | lower | lower |
| `nitramine_to_amine` | Nitramine N–NO2 → amine N–H | neutral | neutral | lower | lower |
| `methyl_to_H` | Ar–CH3 → Ar–H | lower | neutral | lower | raise |
| `hydrogenate_CC` | Aliphatic C=C → C–C | neutral | lower | raise | lower |
| `nitro_to_amino` | Ar–NO2 → Ar–NH2 | neutral | lower | lower | raise |
| `azide_to_amino` | Ar–N3 → Ar–NH2 | lower | neutral | raise | lower |
| `methyl_CH` | Aromatic C–H → C–CH3 | lower | lower | neutral | neutral |
| `ethyl_CH` | Aromatic C–H → C–CH2CH3 | lower | neutral | lower | neutral |
| `hydroxymethyl_CH` | Aromatic C–H → C–CH2OH | neutral | lower | neutral | lower |
| `aromatize_diene` | Cyclohexadiene → benzene | neutral | raise | neutral | lower |
