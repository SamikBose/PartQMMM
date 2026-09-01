# Validation

The repository-integrated analysis suite is validated against the supplied hCAII/AZM topology/PDB and synthetic coordination geometries.

Validated invariants:

- triclinic PBC/minimum-image geometry is shared with production partitioning;
- H-bond-selected waters match the partitioner;
- direct Zn-coordinated waters are selected when `Zn-O <= 2.6 A`;
- a water satisfying both criteria is included only once;
- the complete OPC topology residue, including its virtual charge site, is removed from MM;
- only real O/H/H atoms enter the QM XYZ;
- adaptive neutral waters do not change the configured formal QM charge;
- QM + MM embedding charge reproduces the original force-field system charge;
- shift and RCD preserve identical QM geometry for the same adaptive-water selection;
- RCD virtual sites remain embedding-only;
- analysis uses the same root partitioner and independently checks the union of H-bond and Zn-coordination selections.

The supplied first PDB has its nearest Zn-water O around 5.15 A, so it does not trigger the 2.6 A Zn-coordination rule. The tests additionally create a synthetic 2.10 A Zn--water geometry to verify the new pathway.

## Emax identity / site-field update

The updated analysis additionally records the exact site carrying link-inclusive
and real-QM-only Emax values. Synthetic link atoms are explicit (`Emax_is_link_atom`)
and do not receive a topology atom id.

`site_specific_fields.csv` selects Zn, histidine ND1/NE2, glutamate OE1/OE2,
and adaptive-QM-water oxygen sites from the already-computed per-QM field table.
The plotting module was smoke-tested on synthetic two-run tables and generated
all Emax-identity and site-specific time-series figures without errors.
