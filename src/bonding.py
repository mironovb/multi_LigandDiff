"""Single source of truth for "is there a bond at distance d?" (Finding 6).

The pipeline used to decide *what is bonded* three incompatible ways, so the
model learned "what a ligand is" under one rule and was graded under others:

  1. training-data prep  -- a hard 1.9 A organic / 3.0 A donor distance cutoff
     (``prepare_training_data.ORGANIC_BOND_CUTOFF`` / ``DONOR_CUTOFF``);
  2. generation/validity -- molSimplify covalent-radii bonds, via
     ``molecule_builder.sanitycheck`` -> ``ligand_breakdown`` -> molSimplify
     ``mol3D.getBondedAtomsSmart`` -> ``getBondCutoff``;
  3. molecule build      -- OpenBabel connectivity + the pm-scaled ``const.BONDS_*``
     tables (``molecule_builder.get_bond_order``) for bond *order*.

This module is the shared arbiter for #1 and #2: the *presence* of a bond. It
mirrors the rule the model is actually graded against (#2, the validity gate),
so prep can call the very same predicate the validator uses.

The canonical rule (faithful re-implementation of molSimplify
``mol3D.getBondCutoff``, the default ``BondedOct=False`` path that
``ligand_breakdown`` walks)::

    cutoff(i, j) = 1.15 * (r_cov_i + r_cov_j)          # base
    if one atom is C and the other is not H:  cutoff = min(2.75, cutoff)
    if one atom is H and the other is a transition metal:
                                              cutoff = 1.10 * (r_cov_i + r_cov_j)
    bonded  <=>  d < cutoff

The covalent radii are copied verbatim from molSimplify's
``globalvars.amass()`` (the ``rad`` column molSimplify feeds into the rule), so
``are_bonded`` agrees with the validator element-for-element. Because the same
1.15x rule governs the metal in the default path, the Ln<->donor cutoff falls
straight out of the same expression (e.g. La-O 2.78, La-N 2.81, La-S 3.12 A) --
no separate hand-tuned donor constant is needed.

Relationship to ``const.BONDS_*`` / ``MARGINS_EDM`` (molecule build, #3)
-----------------------------------------------------------------------
``molecule_builder.get_bond_order`` answers a *different* question -- the bond
*order* (single/double/triple) of an already-perceived contact -- using
element-pair *equilibrium* lengths in picometres plus ``MARGINS_EDM=[10,5,2]``::

    single-bond present  <=>  100*d < BONDS_1[i][j] + 10   (i.e. d < (BONDS_1+10)/100 A)

Those single-bond thresholds are tighter than this module's covalent cutoff
(e.g. C-O 1.53 vs 1.72 A, C-C 1.64 vs 1.77 A), because ``get_bond_order`` only
*grades the order of* bonds that OpenBabel already drew -- it is not the
is-there-a-bond arbiter. We deliberately leave molecule-build's order table
alone (order != presence); the contract this module enforces is narrower and
the one Finding 6 is about: prep and the validity gate must agree on **whether
a bond exists at distance d**. They now do, because both go through here.

``src/projection.py`` carries a third, more permissive table
(``BOND_PERCEPTION_CUTOFFS``, ~1.3x covalent radii) describing OpenBabel's
*build-time* connectivity -- the cutoff the diffusion exclusion-shell must
clear so two donors are not re-fused. It intentionally sits *above* this
module's gate (a contact this module would call bonded must also be cleared by
projection), so the three perceptions are ordered, not contradictory:

    get_bond_order single-bond  <  bonding gate (here)  <  projection shell
        (~BONDS_1+margin)            (1.15x covrad)          (~1.3x covrad)

If molSimplify is ever re-pinned, re-extract ``COVALENT_RADII`` and the 1.15x /
2.75 / 1.10 constants from ``mol3D.getBondCutoff`` so this module keeps tracking
the arbiter. ``cross_check_against_molsimplify`` (below) re-derives the gate
from a live molSimplify and asserts agreement -- run it after any bump.
"""

# ---------------------------------------------------------------------------
# Covalent radii (Angstrom).
#
# Copied verbatim from molSimplify ``globalvars.amass()`` -- the ``rad`` (third)
# column of each element record, which molSimplify's ``atom3D.rad`` reads and
# ``mol3D.getBondCutoff`` sums. Keeping these identical is what makes
# ``are_bonded`` agree with the validity gate. Unknown elements fall back to
# molSimplify's own default (``atom3D`` uses 0.75 when a symbol is absent).
# ---------------------------------------------------------------------------
COVALENT_RADII = {
    # non-metals (model's heavy-atom vocabulary + H and a few CIF stragglers)
    'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71,
    'P': 1.06, 'S': 1.02, 'Cl': 0.99, 'Br': 1.14, 'I': 1.4,
    'B': 0.85, 'Si': 1.16, 'As': 1.21,
    # transition metals (const.metals2idx coverage)
    'Ti': 1.36, 'V': 1.22, 'Cr': 1.27, 'Mn': 1.39, 'Fe': 1.25,
    'Co': 1.26, 'Ni': 1.21, 'Cu': 1.38, 'Zn': 1.31, 'Zr': 1.54,
    'Mo': 1.38, 'Ru': 1.25, 'Rh': 1.25, 'Pd': 1.2, 'Cd': 1.48,
    'W': 1.46, 'Re': 1.59, 'Os': 1.28, 'Ir': 1.37, 'Pt': 1.23,
    # lanthanides (the Ln-adapted dataset)
    'La': 1.69, 'Ce': 1.63, 'Pr': 1.76, 'Nd': 1.74, 'Pm': 1.73,
    'Sm': 1.72, 'Eu': 1.68, 'Gd': 1.69, 'Tb': 1.68, 'Dy': 1.67,
    'Ho': 1.66, 'Er': 1.65, 'Tm': 1.64, 'Yb': 1.7, 'Lu': 1.62,
}

# molSimplify's fallback radius for symbols missing from the table
# (``atom3D.__init__``: ``self.rad = 0.75``).
DEFAULT_RADIUS = 0.75

# Lanthanide elements (matches prepare_training_data.LN_ELEMENTS, incl. Pm).
LN_ELEMENTS = frozenset({
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
})

# d-block transition metals. Used ONLY for the metal-H branch of the cutoff, to
# mirror molSimplify's ``atom.ismetal()`` whose default is
# ``transition_metals_only=True`` -- so the 1.10x tweak fires for Ti-H but NOT
# for the lanthanides (La-H stays 1.15x, exactly as molSimplify computes it).
# Verified a superset of ``globalvars.metalslist(transition_metals_only=True)``
# for every symbol in this repo's vocabulary.
_TRANSITION_METALS = frozenset({
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
})

# Cutoff constants, lifted from molSimplify ``mol3D.getBondCutoff``.
BOND_SCALE = 1.15          # base multiplier on the summed covalent radii
METAL_H_BOND_SCALE = 1.10  # tighter multiplier for (transition-metal)-H bonds
C_BOND_CAP = 2.75          # ceiling on C-X (X != H) cutoffs (molSimplify, 2021-07-22)


def covalent_radius(element):
    """Covalent radius (Angstrom) for *element*, molSimplify's value.

    Falls back to ``DEFAULT_RADIUS`` for symbols outside the table, mirroring
    molSimplify's own behaviour for unrecognised atoms.
    """
    return COVALENT_RADII.get(element, DEFAULT_RADIUS)


def bond_cutoff(element_i, element_j):
    """Maximum bonded distance (Angstrom) for the *(element_i, element_j)* pair.

    Faithful re-implementation of molSimplify ``mol3D.getBondCutoff`` -- the
    cutoff the validity gate (``ligand_breakdown``/``sanitycheck``) applies on
    its default ``BondedOct=False`` path. Two atoms are bonded iff their
    separation is below this value; see :func:`are_bonded`.
    """
    r_i = covalent_radius(element_i)
    r_j = covalent_radius(element_j)
    cutoff = BOND_SCALE * (r_i + r_j)

    # C-X (X != H): cap so over-long carbon contacts are not called bonds.
    if element_i == 'C' and element_j != 'H':
        cutoff = min(C_BOND_CAP, cutoff)
    if element_j == 'C' and element_i != 'H':
        cutoff = min(C_BOND_CAP, cutoff)

    # Transition-metal-H: tighter cutoff (lanthanides excluded, matching
    # molSimplify's ``ismetal`` default of transition_metals_only=True).
    if element_j == 'H' and element_i in _TRANSITION_METALS:
        cutoff = METAL_H_BOND_SCALE * (r_i + r_j)
    if element_i == 'H' and element_j in _TRANSITION_METALS:
        cutoff = METAL_H_BOND_SCALE * (r_i + r_j)

    return cutoff


def are_bonded(element_i, element_j, distance):
    """True iff atoms of *element_i*/*element_j* at *distance* A are bonded.

    The single is-there-a-bond predicate shared by training-data prep and the
    validity gate. Accepts plain floats or numpy scalars (e.g. from
    ``np.linalg.norm``); always returns a Python ``bool``.
    """
    return bool(distance < bond_cutoff(element_i, element_j))


# Representative lanthanide radius (mean over LN_ELEMENTS) for the element-
# agnostic ``donor_cutoff()`` convenience value. Per-element donor decisions in
# prep go through ``are_bonded`` with the actual Ln symbol, so this average is
# only used for reporting/diagnostics, never as the binding decision.
_REPRESENTATIVE_LN_RADIUS = (
    sum(COVALENT_RADII[el] for el in LN_ELEMENTS) / len(LN_ELEMENTS)
)


def donor_cutoff(donor_element='O', metal_element=None):
    """Ln<->donor bonded-distance cutoff (Angstrom).

    With *metal_element* given, returns the exact shared cutoff for that
    metal/donor pair (identical to ``bond_cutoff`` and thus to :func:`are_bonded`).
    Called bare, returns the representative value for a mean lanthanide bonding
    a *donor_element* (default O) -- the figure printed in the prep startup
    banner. Replaces the old flat ``DONOR_CUTOFF = 3.0``.
    """
    if metal_element is not None:
        return bond_cutoff(metal_element, donor_element)
    # Heavy donor (never C/H), so neither the C-cap nor the metal-H tweak
    # applies; the base 1.15x rule is exact here.
    return BOND_SCALE * (_REPRESENTATIVE_LN_RADIUS + covalent_radius(donor_element))


# ---------------------------------------------------------------------------
# Traceability: the cutoffs this module supersedes in prepare_training_data.py.
# ---------------------------------------------------------------------------
PREP_LEGACY_ORGANIC_CUTOFF = 1.9  # old flat organic-bond cutoff (A)
PREP_LEGACY_DONOR_CUTOFF = 3.0    # old flat Ln-donor cutoff (A)


def cutoff_change_banner():
    """One-shot startup diff describing prep's move onto the shared rule.

    Printed by ``prepare_training_data.main`` so a re-prep is traceable: the
    effective cutoffs shift from the old flat constants to the covalent-radii
    rule the validity gate uses, which is *expected and intended* (Finding 6).
    """
    return (
        "[bonding] Finding 6: prep now shares src.bonding's covalent-radii rule "
        "(the molSimplify validity-gate arbiter), replacing module-local flat "
        "cutoffs. Re-prep recommended for full consistency (Prompt 18 retrain).\n"
        f"  organic bond: old {PREP_LEGACY_ORGANIC_CUTOFF:.2f} A flat "
        f"-> shared 1.15x(r_i+r_j), e.g. C-C {bond_cutoff('C', 'C'):.2f}, "
        f"C-O {bond_cutoff('C', 'O'):.2f}, O-O {bond_cutoff('O', 'O'):.2f} A\n"
        f"  Ln-donor:     old {PREP_LEGACY_DONOR_CUTOFF:.2f} A flat "
        f"-> shared donor_cutoff() = {donor_cutoff():.2f} A "
        f"(per-element at runtime, e.g. La-O {donor_cutoff(metal_element='La'):.2f} A)"
    )


def cross_check_against_molsimplify(pairs=None, tol=1e-6):
    """Assert this module reproduces molSimplify ``getBondCutoff`` exactly.

    Optional belt-and-suspenders run only where molSimplify is importable (the
    SLURM/validation env, not necessarily a dev box). Returns the list of
    ``(element_i, element_j, our_cutoff, molsimplify_cutoff)`` checked; raises
    ``AssertionError`` on any disagreement so a radii/constant drift is caught.
    """
    from molSimplify.Classes.atom3D import atom3D
    from molSimplify.Classes.mol3D import mol3D

    if pairs is None:
        pairs = [
            ('C', 'C'), ('C', 'O'), ('C', 'N'), ('N', 'O'), ('O', 'O'),
            ('C', 'H'), ('La', 'O'), ('Nd', 'O'), ('La', 'N'), ('La', 'S'),
            ('La', 'Cl'), ('C', 'Nd'), ('La', 'H'), ('Ti', 'H'), ('Fe', 'H'),
        ]
    probe = mol3D()
    checked = []
    for el_i, el_j in pairs:
        ours = bond_cutoff(el_i, el_j)
        theirs = probe.getBondCutoff(atom3D(el_i, [0, 0, 0]),
                                     atom3D(el_j, [0, 0, 0]))
        assert abs(ours - theirs) < tol, (
            f"{el_i}-{el_j}: bonding={ours:.6f} != molSimplify={theirs:.6f}")
        checked.append((el_i, el_j, ours, theirs))
    return checked


if __name__ == '__main__':
    print(cutoff_change_banner())
    print(f"\ndonor_cutoff() = {donor_cutoff():.3f} A")
    for a, b, d in [('C', 'O', 1.43), ('C', 'O', 3.0), ('O', 'O', 1.30)]:
        print(f"  are_bonded({a!r}, {b!r}, {d}) = {are_bonded(a, b, d)}"
              f"   (cutoff {bond_cutoff(a, b):.3f} A)")
