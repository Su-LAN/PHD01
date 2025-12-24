"""
Utilities to extract the hypertension subgraph from iBKH and align it to
MIMIC-derived variables.

Usage (inside data.ipynb):
    from htn_ibkh import extract_htn_ibkh_subgraph, align_ibkh_to_mimic
    nodes, edges = extract_htn_ibkh_subgraph()
    alignment = align_ibkh_to_mimic(nodes, cohort_columns=list(cohort.columns))
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


IBKH_BASE_DEFAULT = Path("Dataset/iBKH")

# Drug name patterns to match ACEI/ARB for alignment to MIMIC exposures
ACEI_NAMES = [
    "PRIL",  # catches most ACEIs
    "CAPTOPRIL",
    "ENALAPRIL",
    "LISINOPRIL",
    "BENAZEPRIL",
    "FOSINOPRIL",
    "MOEXIPRIL",
    "PERINDOPRIL",
    "QUINAPRIL",
    "RAMIPRIL",
    "TRANDOLAPRIL",
]
ARB_NAMES = [
    "SARTAN",  # catches most ARBs
    "LOSARTAN",
    "VALSARTAN",
    "IRBESARTAN",
    "OLMESARTAN",
    "CANDESARTAN",
    "TELMISARTAN",
    "EPROSARTAN",
    "AZILSARTAN",
]


def _load_vocabs(base: Path) -> Dict[str, pd.DataFrame]:
    """Load iBKH vocabulary tables needed for hypertension subgraph."""
    entity = base / "entity"
    return {
        "disease": pd.read_csv(entity / "disease_vocab.csv"),
        "drug": pd.read_csv(entity / "drug_vocab.csv"),
        "symptom": pd.read_csv(entity / "symptom_vocab.csv"),
        "gene": pd.read_csv(entity / "gene_vocab.csv"),
        "pathway": pd.read_csv(entity / "pathway_vocab.csv"),
    }


def _lookup_name(table: pd.DataFrame, id_col: str, name_col: str, node_id: str) -> str:
    row = table.loc[table[id_col] == node_id]
    if row.empty:
        return node_id
    val = row.iloc[0][name_col]
    return str(val) if pd.notna(val) else node_id


def find_hypertension_ids(vocabs: Dict[str, pd.DataFrame], include_variants: bool = True) -> List[str]:
    """
    Find DOID identifiers for hypertension (and optional variants).
    """
    disease = vocabs["disease"]
    patterns = ["hypertension"]
    if not include_variants:
        mask = disease["name"].str.fullmatch("hypertension", case=False, na=False)
    else:
        mask = disease["name"].str.contains("|".join(patterns), case=False, na=False)
    ids = disease.loc[mask, "primary"].unique().tolist()
    return ids


def _add_node(nodes: Dict[str, Dict[str, str]], node_id: str, name: str, node_type: str):
    if node_id not in nodes:
        nodes[node_id] = {"id": node_id, "name": name, "type": node_type}


def _drug_edges(base: Path, disease_ids: Iterable[str], vocabs: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], Dict[str, Dict[str, str]]]:
    rel = pd.read_csv(base / "relation" / "D_Di_res.csv")
    disease_ids = set(disease_ids)
    rel = rel[rel["Disease"].isin(disease_ids)]

    nodes: Dict[str, Dict[str, str]] = {}
    edges: List[Dict] = []
    for _, row in rel.iterrows():
        drug_id = row["Drug"]
        dis_id = row["Disease"]
        drug_name = _lookup_name(vocabs["drug"], "primary", "name", drug_id)
        dis_name = _lookup_name(vocabs["disease"], "primary", "name", dis_id)

        relation = "treats" if row.get("Treats", 0) == 1 else "associate"

        _add_node(nodes, drug_id, drug_name, "drug")
        _add_node(nodes, dis_id, dis_name, "disease")
        edges.append(
            {
                "src": drug_id,
                "dst": dis_id,
                "relation": relation,
                "src_type": "drug",
                "dst_type": "disease",
                "src_name": drug_name,
                "dst_name": dis_name,
                "source": row.get("Source", ""),
            }
        )
    return edges, nodes


def _symptom_edges(base: Path, disease_ids: Iterable[str], vocabs: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], Dict[str, Dict[str, str]]]:
    rel = pd.read_csv(base / "relation" / "Di_Sy_res.csv")
    disease_ids = set(disease_ids)
    rel = rel[rel["Disease"].isin(disease_ids)]

    nodes: Dict[str, Dict[str, str]] = {}
    edges: List[Dict] = []
    for _, row in rel.iterrows():
        sym_id = row["Symptom"]
        dis_id = row["Disease"]
        sym_name = _lookup_name(vocabs["symptom"], "primary", "name", sym_id)
        dis_name = _lookup_name(vocabs["disease"], "primary", "name", dis_id)

        _add_node(nodes, sym_id, sym_name, "symptom")
        _add_node(nodes, dis_id, dis_name, "disease")
        edges.append(
            {
                "src": dis_id,
                "dst": sym_id,
                "relation": "has_symptom",
                "src_type": "disease",
                "dst_type": "symptom",
                "src_name": dis_name,
                "dst_name": sym_name,
                "source": row.get("Source", ""),
            }
        )
    return edges, nodes


def _gene_edges(base: Path, disease_ids: Iterable[str], vocabs: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], Dict[str, Dict[str, str]]]:
    rel = pd.read_csv(base / "relation" / "Di_G_res.csv")
    disease_ids = set(disease_ids)
    rel = rel[rel["Disease"].isin(disease_ids)]

    nodes: Dict[str, Dict[str, str]] = {}
    edges: List[Dict] = []
    for _, row in rel.iterrows():
        gene_id = row["Gene"]
        dis_id = row["Disease"]
        gene_name = _lookup_name(vocabs["gene"], "primary", "symbol", gene_id)
        dis_name = _lookup_name(vocabs["disease"], "primary", "name", dis_id)

        _add_node(nodes, gene_id, gene_name, "gene")
        _add_node(nodes, dis_id, dis_name, "disease")
        edges.append(
            {
                "src": dis_id,
                "dst": gene_id,
                "relation": "associated_gene",
                "src_type": "disease",
                "dst_type": "gene",
                "src_name": dis_name,
                "dst_name": gene_name,
                "source": row.get("Source", ""),
            }
        )
    return edges, nodes


def _pathway_edges(base: Path, disease_ids: Iterable[str], vocabs: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], Dict[str, Dict[str, str]]]:
    rel = pd.read_csv(base / "relation" / "Di_Pwy_res.csv")
    disease_ids = set(disease_ids)
    rel = rel[rel["Disease"].isin(disease_ids)]

    nodes: Dict[str, Dict[str, str]] = {}
    edges: List[Dict] = []
    for _, row in rel.iterrows():
        pwy_id = row["Pathway"]
        dis_id = row["Disease"]
        pwy_name = _lookup_name(vocabs["pathway"], "primary", "name", pwy_id)
        dis_name = _lookup_name(vocabs["disease"], "primary", "name", dis_id)

        _add_node(nodes, pwy_id, pwy_name, "pathway")
        _add_node(nodes, dis_id, dis_name, "disease")
        edges.append(
            {
                "src": dis_id,
                "dst": pwy_id,
                "relation": "associated_pathway",
                "src_type": "disease",
                "dst_type": "pathway",
                "src_name": dis_name,
                "dst_name": pwy_name,
                "source": row.get("Source", ""),
            }
        )
    return edges, nodes


def extract_htn_ibkh_subgraph(
    base: Path | str = IBKH_BASE_DEFAULT,
    include_variants: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the hypertension iBKH subgraph (nodes, edges) as pandas DataFrames.
    """
    base_path = Path(base)
    vocabs = _load_vocabs(base_path)
    htn_ids = find_hypertension_ids(vocabs, include_variants=include_variants)

    nodes: Dict[str, Dict[str, str]] = {}
    edges_collect: List[Dict] = []

    # Add seed hypertension nodes
    for hid in htn_ids:
        name = _lookup_name(vocabs["disease"], "primary", "name", hid)
        _add_node(nodes, hid, name, "disease")

    # Collect edges from several relation tables
    for edge_fn in (_drug_edges, _symptom_edges, _gene_edges, _pathway_edges):
        e, n = edge_fn(base_path, htn_ids, vocabs)
        edges_collect.extend(e)
        for nid, data in n.items():
            nodes.setdefault(nid, data)

    nodes_df = pd.DataFrame(nodes.values())
    edges_df = pd.DataFrame(edges_collect)
    return nodes_df, edges_df


def align_ibkh_to_mimic(
    nodes_df: pd.DataFrame,
    cohort_columns: Iterable[str] | None = None,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Heuristic alignment of iBKH hypertension nodes to MIMIC cohort columns.
    Returns a dict with lists for exposures, risks/mediators, outcomes.
    """
    cohort_columns = list(cohort_columns) if cohort_columns is not None else []
    exposures: List[Dict[str, str]] = []
    risks: List[Dict[str, str]] = []
    outcomes: List[Dict[str, str]] = []

    # Exposures: ACEI / ARB drugs
    for _, row in nodes_df[nodes_df["type"] == "drug"].iterrows():
        name = str(row["name"]).upper()
        if any(tag in name for tag in ACEI_NAMES):
            exposures.append(
                {"id": row["id"], "name": row["name"], "maps_to": "acei_exposed"}
            )
        if any(tag in name for tag in ARB_NAMES):
            exposures.append(
                {"id": row["id"], "name": row["name"], "maps_to": "arb_exposed"}
            )

    # Risks / mediators: map common comorbidities to cohort flags
    risk_map = [
        ("diabetes", "diabetes"),
        ("kidney", "ckd"),
        ("chronic kidney", "ckd"),
        ("cardiovascular disease", "cad"),
        ("coronary", "cad"),
        ("heart failure", "heart_failure"),
        ("obesity", "bmi_omr"),
        ("body mass index", "bmi_omr"),
        ("glucose", "glucose_baseline"),
        ("creatinine", "creatinine_baseline"),
    ]
    for _, row in nodes_df.iterrows():
        name = str(row["name"]).lower()
        for key, target in risk_map:
            if key in name:
                risks.append({"id": row["id"], "name": row["name"], "maps_to": target})
                break

    # Outcomes: mortality
    for _, row in nodes_df.iterrows():
        name = str(row["name"]).lower()
        if "mortality" in name or "death" in name:
            outcomes.append(
                {"id": row["id"], "name": row["name"], "maps_to": "hospital_mortality"}
            )

    # Optionally filter to only columns present in cohort
    if cohort_columns:
        exposures = [r for r in exposures if r["maps_to"] in cohort_columns]
        risks = [r for r in risks if r["maps_to"] in cohort_columns]
        outcomes = [r for r in outcomes if r["maps_to"] in cohort_columns]

    return {
        "exposures": exposures,
        "risks": risks,
        "outcomes": outcomes,
    }

