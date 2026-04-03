import pandas as pd
import networkx as nx
from sqlalchemy import create_engine
from pathlib import Path

DB_CONFIG = {
    "host": "localhost",
    "database": "pharmgraph",
    "user": "maywelkin",
    "password": "",
    "port": 5432,
}

HGNC_FILE = "hgnc_complete_set.tsv"
OUTPUT_CSV = "gene_drug_edges.csv"


def get_engine():
    conn_string = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(conn_string)


def load_gene_family_map(hgnc_path: Path) -> pd.DataFrame:
    """
    Load HGNC data and return a 2-column dataframe:
    symbol | gene_family

    In the HGNC file you uploaded, the family-like field is `gene_group`,
    so we rename it to `gene_family` for easier downstream use.
    """
    hgnc_df = pd.read_csv(hgnc_path, sep="\t", dtype=str)

    if "symbol" not in hgnc_df.columns:
        raise ValueError("HGNC file is missing required column: 'symbol'")

    # Use gene_group as gene family
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

    # Make multi-family values easier to read in CSV/filter UI
    family_df["gene_family"] = (
        family_df["gene_family"]
        .fillna("")
        .astype(str)
        .str.replace("|", "; ", regex=False)
        .str.strip()
    )

    # Remove duplicate symbols if any
    family_df = family_df.drop_duplicates(subset=["symbol"])

    return family_df


def main():
    base_dir = Path(__file__).resolve().parent
    hgnc_path = base_dir / HGNC_FILE
    output_path = base_dir / OUTPUT_CSV

    if not hgnc_path.exists():
        raise FileNotFoundError(f"Cannot find HGNC file: {hgnc_path}")

    engine = get_engine()

    # Get gene-drug edge list from database
    query = """
    SELECT
        g.gene_symbol AS source,
        'gene' AS source_type,
        d.drug_name AS target,
        'drug' AS target_type,
        gdi.source_database,
        gdi.source_version,
        gdi.interaction_type,
        gdi.interaction_score
    FROM gene_drug_interactions gdi
    JOIN genes g ON g.gene_id = gdi.gene_id
    JOIN drugs d ON d.drug_id = gdi.drug_id
    """

    edges_df = pd.read_sql(query, engine)

    # Add gene family column from HGNC
    gene_family_df = load_gene_family_map(hgnc_path)
    edges_df = edges_df.merge(
        gene_family_df,
        how="left",
        left_on="source",
        right_on="symbol"
    )
    edges_df = edges_df.drop(columns=["symbol"])
    edges_df["gene_family"] = edges_df["gene_family"].fillna("")

    print("Number of edges:", len(edges_df))
    print(edges_df.head())

    # Create graph bipartite: gene <-> drug
    G = nx.Graph()

    for _, row in edges_df.iterrows():
        G.add_node(
            row["source"],
            node_type=row["source_type"],
            gene_family=row["gene_family"]
        )
        G.add_node(row["target"], node_type=row["target_type"])

        G.add_edge(
            row["source"],
            row["target"],
            source_database=row["source_database"],
            source_version=row["source_version"],
            interaction_type=row["interaction_type"],
            interaction_score=row["interaction_score"],
        )

    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges in graph:", G.number_of_edges())

    # Export to CSV
    edges_df.to_csv(output_path, index=False)
    print(f"Saved edge list to {output_path}")

    # Check mapping coverage
    matched = (edges_df["gene_family"].str.strip() != "").sum()
    total_unique_genes = edges_df["source"].nunique()
    matched_unique_genes = edges_df.loc[
        edges_df["gene_family"].str.strip() != "", "source"
    ].nunique()
    print(f"Rows with gene family: {matched}/{len(edges_df)}")
    print(f"Unique genes with gene family: {matched_unique_genes}/{total_unique_genes}")

    # Top 10 gene connected to most drug
    gene_degrees = []
    for node, degree in G.degree():
        if G.nodes[node].get("node_type") == "gene":
            gene_degrees.append((node, degree))

    gene_degrees = sorted(gene_degrees, key=lambda x: x[1], reverse=True)
    print("\nTop 10 most connected genes:")
    for gene, deg in gene_degrees[:10]:
        family = G.nodes[gene].get("gene_family", "")
        print(gene, deg, f"| family: {family}")


if __name__ == "__main__":
    main()