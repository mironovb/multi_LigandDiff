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
