"""Working with pdb in Python."""

import os
import sys
import urllib

import Bio
from Bio.PDB.Polypeptide import protein_letters_3to1

protein_letters_3to1["UNK"] = "X"  # Unknown amino acid
protein_letters_3to1["SEC"] = "X"  # Selenocysteine
protein_letters_3to1["PYL"] = "X"  # Pyrrolysine

TIMEOUT = 10  # wait only 10 seconds


def get_sequence_from_pdb_file(pdb_filename, nonstandard_warnings=False):
    """
    Retrieves the protein sequence from a PDB file.

    Args:
        pdb_filename (str): The path to the PDB file.
        nonstandard_warnings (bool, optional): If True, warning messages for nonstandard residues.
            Defaults to False.

    Returns:
        str: The protein sequence extracted from the PDB file.
    """
    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_filename, pdb_filename)
    assert len(structure) == 1

    seq = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":  # This checks if it's a standard residue
                    seq.append(protein_letters_3to1[residue.get_resname()])
                else:
                    if nonstandard_warnings:
                        print("nonstandard", residue.get_id())

    return "".join(seq)


def get_res_list_from_pdb_file(pdb_filename, nonstandard_warnings=False):
    """
    Extracts a list of residue numbers from a PDB file.

    Args:
        pdb_filename (str): The path to the PDB file.
        nonstandard_warnings (bool, optional): If True, warning messages for nonstandard residues.
            Defaults to False.

    Returns:
        list: A list of residue identifiers.

    Raises:
        AssertionError: If the PDB file contains more than one structure.
    """
    pdb_parser = Bio.PDB.PDBParser()
    structure = pdb_parser.get_structure(pdb_filename, pdb_filename)
    assert len(structure) == 1

    res_list = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":  # This checks if it's a standard residue
                    # from here: https://bioinformatics.stackexchange.com/a/15961
                    # .get_id() return a tuple with (hetero flag, sequence identifier, insertion code)
                    res_list.append(residue.get_id()[1])
                else:
                    if nonstandard_warnings:
                        print("nonstandard", residue.get_id())

    return _find_consecutive_periods(res_list)


def _find_consecutive_periods(lst):
    """
    Finds consecutive periods in a list of numbers. Useful for getting boundaries of residues
    in a PDB file.

    Example:

    >>> _find_consecutive_periods([1, 2, 3, 5, 6, 7, 10, 11, 12])
    [(1, 3), (5, 7), (10, 12)]

    Args:
        lst (list): A list of numbers.

    Returns:
        list: A list of tuples representing consecutive periods.
        Each tuple contains the start and end of a consecutive period.
    """
    periods = []
    start = None
    end = None

    for i in range(len(lst)):
        if start is None:
            start = lst[i]
            end = lst[i]
        elif lst[i] == end + 1:
            end = lst[i]
        else:
            periods.append((start, end))
            start = lst[i]
            end = lst[i]

    if start is not None:
        periods.append((start, end))

    return periods


def download_pdb(pdbcode, datadir, downloadurl="https://files.rcsb.org/download/"):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :param downloadurl: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
    :return: the full path to the downloaded PDB file or None if something went wrong
    """

    pdbfn = pdbcode + ".pdb"
    url = downloadurl + pdbfn
    outfnm = os.path.join(datadir, pdbfn)

    if os.path.exists(outfnm):
        print(
            f"File {outfnm} already exists for pdb code: {pdbcode}, skipping download"
        )
        return outfnm

    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm
    except Exception as err:
        print(str(err), file=sys.stderr)
        return None
