import pandas as pd
import networkx as nx
from sqlalchemy import create_engine
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent

DB_CONFIG = {
    "host": "localhost",
    "database": "pharmgraph",
    "user": "maywelkin",
    "password": "",
    "port": 5432,
}

HGNC_FILE = "hgnc_complete_set.tsv"
OUTPUT_JSON = "gene_drug.json"


def get_engine():
    conn_string = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(conn_string)


def load_gene_family_map(hgnc_path: Path) -> pd.DataFrame:
    """
    Return a dataframe with:
    symbol | gene_family

    In the HGNC file, the family-like field is `gene_group`.
    """
    hgnc_df = pd.read_csv(hgnc_path, sep="\t", dtype=str)

    if "symbol" not in hgnc_df.columns:
        raise ValueError("HGNC file is missing required column: 'symbol'")

    family_col = None
    for col in ["gene_group", "gene_family", "gene_family_name"]:
        if col in hgnc_df.columns:
            family_col = col
            break

    if family_col is None:
        raise ValueError(
            "Could not find a gene family column in HGNC file. "
            "Expected one of: gene_group, gene_family, gene_family_name"
        )

    family_df = hgnc_df[["symbol", family_col]].copy()
    family_df = family_df.rename(columns={family_col: "gene_family"})

    family_df["gene_family"] = (
        family_df["gene_family"]
        .fillna("")
        .astype(str)
        .str.replace("|", "; ", regex=False)
        .str.strip()
    )

    family_df = family_df.drop_duplicates(subset=["symbol"])
    return family_df


def main():
    engine = get_engine()

    query = """
    SELECT
        g.gene_symbol AS gene,
        d.drug_name AS drug,
        gdi.source_database,
        gdi.source_version,
        gdi.interaction_type,
        gdi.interaction_score
    FROM gene_drug_interactions gdi
    JOIN genes g ON g.gene_id = gdi.gene_id
    JOIN drugs d ON d.drug_id = gdi.drug_id
    """

    df = pd.read_sql(query, engine)

    hgnc_path = BASE_DIR / HGNC_FILE
    if not hgnc_path.exists():
        raise FileNotFoundError(f"Cannot find HGNC file: {hgnc_path}")

    gene_family_df = load_gene_family_map(hgnc_path)

    # Group edges
    grouped = (
        df.groupby(["gene", "drug"], dropna=False)
        .agg(
            max_score=("interaction_score", "max"),
            mean_score=("interaction_score", "mean"),
            edge_count=("interaction_type", "size"),
            interaction_types=("interaction_type", lambda x: sorted({str(v) for v in x.dropna()})),
            source_databases=("source_database", lambda x: sorted({str(v) for v in x.dropna()})),
        )
        .reset_index()
    )

    # Gene nodes with family
    genes = pd.DataFrame({
        "id": grouped["gene"].unique(),
        "label": grouped["gene"].unique(),
        "node_type": "gene",
    }).merge(
        gene_family_df,
        how="left",
        left_on="id",
        right_on="symbol"
    ).drop(columns=["symbol"])

    genes["gene_family"] = genes["gene_family"].fillna("")

    # Drug nodes
    drugs = pd.DataFrame({
        "id": grouped["drug"].unique(),
        "label": grouped["drug"].unique(),
        "node_type": "drug",
    })

    nodes = pd.concat([genes, drugs], ignore_index=True).drop_duplicates(subset=["id"])

    # Degree calculation
    G = nx.Graph()
    for _, row in grouped.iterrows():
        G.add_edge(row["gene"], row["drug"])

    node_rows = []
    for _, row in nodes.iterrows():
        node_id = row["id"]
        degree = G.degree(node_id) if node_id in G else 0
        size = 2 + min(degree * 0.15, 20)

        color = "#ff7f0e" if row["node_type"] == "gene" else "#4da6ff"

        node_data = {
            "id": node_id,
            "label": row["label"],
            "type": row["node_type"],
            "degree": degree,
            "size": size,
            "color": color,
        }

        if row["node_type"] == "gene":
            node_data["gene_family"] = row.get("gene_family", "") if pd.notna(row.get("gene_family", "")) else ""

        node_rows.append(node_data)

    edge_rows = []
    for i, row in grouped.iterrows():
        edge_rows.append({
            "id": f"e{i}",
            "source": row["gene"],
            "target": row["drug"],
            "weight": None if pd.isna(row["max_score"]) else float(row["max_score"]),
            "mean_score": None if pd.isna(row["mean_score"]) else float(row["mean_score"]),
            "edge_count": int(row["edge_count"]),
            "interaction_types": row["interaction_types"],
            "source_databases": row["source_databases"],
        })

    out = {
        "nodes": node_rows,
        "edges": edge_rows,
    }

    with open(BASE_DIR / OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    gene_nodes = sum(1 for n in node_rows if n["type"] == "gene")
    gene_nodes_with_family = sum(1 for n in node_rows if n["type"] == "gene" and n.get("gene_family", "").strip())

    print(f"Saved {OUTPUT_JSON}")
    print("Nodes:", len(node_rows))
    print("Edges:", len(edge_rows))
    print(f"Gene nodes with family: {gene_nodes_with_family}/{gene_nodes}")


if __name__ == "__main__":
    main()