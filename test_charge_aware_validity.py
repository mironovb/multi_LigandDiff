from rdkit import Chem


def _nitrate():
    rw = Chem.RWMol()
    n = Chem.Atom("N"); n.SetFormalCharge(1); i_n = rw.AddAtom(n)
    for fc in (0, -1, -1):
        o = Chem.Atom("O"); o.SetFormalCharge(fc); i_o = rw.AddAtom(o)
        rw.AddBond(i_n, i_o, Chem.BondType.DOUBLE if fc == 0 else Chem.BondType.SINGLE)
    return rw.GetMol()


def test_nitrate_sanitizes_with_charges():
    Chem.SanitizeMol(_nitrate())            # must NOT raise


def test_nitrate_fails_without_charges():
    rw = Chem.RWMol(_nitrate())
    for a in rw.GetAtoms():
        a.SetFormalCharge(0)
    try:
        Chem.SanitizeMol(rw.GetMol()); raised = False
    except Exception:
        raised = True
    assert raised, "stripping charges should make nitrate N hit the valence error"


def test_gate_accepts_nitrate():
    from src.molecule_builder import BasicLigandMetrics
    # _nitrate() built WITH charges (as make_mol_openbabel now yields after prompt 01)
    assert len(BasicLigandMetrics().compute_validity([_nitrate()])) == 1


def _metal_overvalent_amine():
    # N single-bonded to 3 C and 1 Fe. Plain RDKit sanitize sees N with explicit
    # valence 4 (over-valent) and raises; reset_dative_bonds (in the gate) converts
    # the Fe-N single bond to DATIVE, dropping N's counted valence to 3 -> legal.
    rw = Chem.RWMol()
    i_fe = rw.AddAtom(Chem.Atom("Fe"))
    i_n = rw.AddAtom(Chem.Atom("N"))
    rw.AddBond(i_fe, i_n, Chem.BondType.SINGLE)
    for _ in range(3):
        i_c = rw.AddAtom(Chem.Atom("C"))
        rw.AddBond(i_n, i_c, Chem.BondType.SINGLE)
    return rw.GetMol()


def test_plain_sanitize_rejects_metal_overvalent_amine():
    try:
        Chem.SanitizeMol(_metal_overvalent_amine()); raised = False
    except Exception:
        raised = True
    assert raised, "a 4-single-bond N (incl. metal) must over-valence under plain sanitize"


def test_gate_accepts_metal_donor_via_dative():
    # This is the test that actually exercises the prompt-02 gate logic
    # (reset_dative_bonds + ADJUSTHS sanitize), unlike the metal-free nitrate above
    # which already passes plain SanitizeMol.
    from src.molecule_builder import BasicLigandMetrics
    mol = _metal_overvalent_amine()
    out = BasicLigandMetrics().compute_validity([mol])
    assert len(out) == 1                              # gate accepts the metal-donor
    assert out[0] is mol                              # returns the ORIGINAL, not the dative copy
    # the original Fe-N bond is untouched (still SINGLE), so downstream GetMolFrags
    # in compute_connectivity sees the original connectivity -- the append-original-i contract
    assert mol.GetBondBetweenAtoms(0, 1).GetBondType() == Chem.BondType.SINGLE
