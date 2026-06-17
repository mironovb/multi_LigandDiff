"""Built-in ligand-skeleton templates for ``generate.py --ligand_templates`` (Prompt 10).

A *template* is a tiny, fixed heavy-atom (non-H) skeleton for a whole ligand slot:
the element identity of every atom, the slot's denticity (how many of those atoms
touch the metal), and -- optionally -- a rough internal geometry and connectivity.
Seeding a template into a generated ligand slot lets the diffusion sampler *place
and refine* a known fragment (e.g. a nitrate) instead of having to *invent* its
composition, which is the failure mode the bare-Eu run hits on the three nitrates.

This is the cheapest bridge to the Strategy's "produce a valence-valid graph, then
realise geometry": Prompt 09 (``--donor_spec``) fixes which atoms touch the metal;
this goes one step further and seeds the *whole* small skeleton.

IMPORTANT -- a template is a *seeding aid*, NOT a hard constraint (see the
``generate.py`` docstring). The atom-count budget takes effect immediately (a
nitrate slot becomes exactly 4 atoms); the element identities / geometry are seeded
into the input representation, but the current sampler re-noises the generated
region from N(0, I), so they bias the *representation*, not (yet) the reverse
process. Hard, valence-correct enforcement is Prompt 16.

Atom-ordering convention (important): the FIRST ``denticity`` atoms of ``elements``
are the donor atoms (the coord_site==1 rows that face the metal). ``generate.py``
flags the first ``denticity`` rows of each slot as donors, so list donor atoms
first and keep ``coords``/``bonds`` consistent with that ordering.

Template schema (per tag):
    elements     (required) list of element symbols, donor atoms first.
    denticity    (required) int >= 1, number of donor atoms; must be <= len(elements).
    coords       (optional) list of [x, y, z] rough internal geometry, one per atom.
    bonds        (optional) list of [i, j, order] connectivity (0-based atom indices;
                            order 1=single, 2=double, ...). Documentation / Prompt 16.
    smiles       (optional) SMILES string of the fragment (metadata).
    charge       (optional) formal charge of the fragment (metadata).
    description  (optional) human-readable note.

This module is intentionally dependency-free (stdlib ``json`` only) so it imports
cleanly even when the heavy ``generate.py`` stack (torch / rdkit / openbabel /
wandb) is unavailable. Element-symbol validity is checked by the caller
(``generate.py``) against ``const.ATOM2IDX``, not here.
"""

import json


# Built-in templates. Bond order: 1 single, 2 double.
TEMPLATES = {
    'nitrate': {
        'description': 'Bidentate nitrate NO3- (kappa2-O,O): N(+) with one N=O and two N-O(-).',
        'smiles': '[O-][N+](=O)[O-]',
        'elements': ['O', 'O', 'N', 'O'],   # donors first: 2 coordinating O, then N, then terminal O
        'denticity': 2,
        'charge': -1,
        # rough planar geometry: N at origin, N-O ~1.25 A, O-N-O ~120 deg
        'coords': [
            [1.250, 0.000, 0.000],     # O (donor)
            [-0.625, 1.083, 0.000],    # O (donor)
            [0.000, 0.000, 0.000],     # N
            [-0.625, -1.083, 0.000],   # O (terminal, N=O)
        ],
        'bonds': [[2, 0, 1], [2, 1, 1], [2, 3, 2]],  # N-O, N-O, N=O
    },
    'water': {
        'description': 'Aqua ligand; heavy-atom skeleton is a single O donor (H added later).',
        'smiles': 'O',
        'elements': ['O'],
        'denticity': 1,
        'charge': 0,
        'coords': [[0.000, 0.000, 0.000]],
        'bonds': [],
    },
    'carboxylate': {
        'description': 'Bidentate carboxylate -COO- core (kappa2-O,O), formate-like CO2 head.',
        'smiles': '[O-]C=O',
        'elements': ['O', 'O', 'C'],   # donors first: 2 coordinating O, then the carboxyl C
        'denticity': 2,
        'charge': -1,
        # rough geometry: C at origin, C-O ~1.26 A, O-C-O ~124 deg
        'coords': [
            [0.591, 1.107, 0.000],     # O (donor)
            [0.591, -1.107, 0.000],    # O (donor)
            [0.000, 0.000, 0.000],     # C
        ],
        'bonds': [[2, 0, 1], [2, 1, 2]],  # C-O, C=O
    },
}


def validate_template(tag, tpl):
    """Structurally validate one template dict; raise ValueError on a malformed entry.

    Checks the required ``elements``/``denticity`` fields and the internal consistency
    of the optional ``coords``/``bonds`` (lengths, index ranges). Element-symbol
    validity (against ``const.ATOM2IDX``) is intentionally left to the caller so this
    module stays dependency-free. Returns True on success.
    """
    if not isinstance(tpl, dict):
        raise ValueError(f"template '{tag}' must be a dict, got {type(tpl).__name__}")
    elements = tpl.get('elements')
    if not isinstance(elements, list) or not elements:
        raise ValueError(f"template '{tag}' needs a non-empty 'elements' list")
    if not all(isinstance(e, str) and e for e in elements):
        raise ValueError(f"template '{tag}' 'elements' must be element-symbol strings")
    n = len(elements)
    dent = tpl.get('denticity')
    if not isinstance(dent, int) or isinstance(dent, bool) or dent < 1:
        raise ValueError(f"template '{tag}' needs an integer 'denticity' >= 1")
    if dent > n:
        raise ValueError(
            f"template '{tag}' denticity {dent} exceeds its {n} atom(s); the first "
            f"'denticity' atoms are the donors, so denticity must be <= atom count")
    coords = tpl.get('coords')
    if coords is not None:
        if not isinstance(coords, list) or len(coords) != n:
            raise ValueError(
                f"template '{tag}' has {len(coords) if isinstance(coords, list) else '?'} "
                f"coord rows for {n} atom(s); supply one [x,y,z] per atom (or omit 'coords')")
        for row in coords:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ValueError(f"template '{tag}' 'coords' rows must be [x, y, z] triples")
    bonds = tpl.get('bonds')
    if bonds is not None:
        if not isinstance(bonds, list):
            raise ValueError(f"template '{tag}' 'bonds' must be a list of [i, j, order]")
        for b in bonds:
            if not isinstance(b, (list, tuple)) or len(b) < 2:
                raise ValueError(f"template '{tag}' bond {b} must be at least [i, j]")
            if not (0 <= int(b[0]) < n) or not (0 <= int(b[1]) < n):
                raise ValueError(f"template '{tag}' bond {b} indexes outside 0..{n - 1}")
    return True


def build_library(extra=None):
    """Return the template library: built-in :data:`TEMPLATES` merged with ``extra``.

    ``extra`` is an optional ``{tag: template}`` map (e.g. loaded from a user JSON via
    ``--ligand_templates``); its entries override built-ins of the same tag. Every entry
    -- built-in and extra -- is structurally validated. Returns a fresh dict; the
    module-level :data:`TEMPLATES` is never mutated.
    """
    lib = dict(TEMPLATES)
    if extra is not None:
        if not isinstance(extra, dict):
            raise ValueError("template library extension must be a {tag: template} map")
        for tag, tpl in extra.items():
            validate_template(tag, tpl)
            lib[tag] = tpl
    for tag, tpl in lib.items():
        validate_template(tag, tpl)
    return lib


def load_library_json(path):
    """Load a ``--ligand_templates`` JSON file and return ``(extra_map, assign_list)``.

    The JSON must be an object. Recognised keys:
        "templates" (alias "library")  -- optional {tag: template} map extending the
                                          built-in library.
        "assign"                       -- required list of template tags, one per
                                          generated ligand slot (build order).
    Raises ValueError on a malformed file. The actual merge/validation against
    ``const.ATOM2IDX`` is done by the caller via :func:`build_library`.
    """
    with open(path) as fh:
        blob = json.load(fh)
    if not isinstance(blob, dict):
        raise ValueError(f"--ligand_templates JSON '{path}' must be a JSON object")
    extra = blob.get('templates', blob.get('library'))
    assign = blob.get('assign')
    if not isinstance(assign, list) or not assign:
        raise ValueError(
            f"--ligand_templates JSON '{path}' needs a non-empty 'assign' list of tags, "
            f"one per generated ligand slot, e.g. \"assign\": [\"nitrate\", \"nitrate\"]")
    return extra, [str(t).strip() for t in assign]
