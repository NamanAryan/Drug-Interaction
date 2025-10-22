import torch
from rdkit import Chem
from torch_geometric.data import Data
import pandas as pd
import requests

# Load fallback CSV
drug_df = pd.read_csv("drugs.csv")
fallback_smiles_dict = dict(zip(drug_df['name'].str.lower(), drug_df['smiles']))
# TODO: validate CSV format (missing columns or empty rows)

def drug_name_to_smiles(drug_name):
    name = drug_name.strip().lower()

    # Check fallback CSV
    if name in fallback_smiles_dict:
        return fallback_smiles_dict[name]

    # Try PubChem as backup
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except:
        # FIXME: add logging for PubChem API failures
        return None
    return None

def get_available_drug_names():
    return list(fallback_smiles_dict.keys())
    # TODO: maybe sort alphabetically before returning for UI consistency

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # FIXME: handle invalid SMILES strings gracefully
        return Data(x=torch.empty((0,1)), edge_index=torch.empty((2,0), dtype=torch.long))

    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    # NOTE: currently only using atomic numbers, consider adding features like formal charge or hybridization
    x = torch.tensor([atom.GetAtomicNum() for atom in atoms], dtype=torch.float).unsqueeze(1)
    edge_index = []
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def get_clinical_insights(drug1, drug2):
    try:
        df = pd.read_csv("interactions.csv")
        # TODO: validate CSV columns ('drug1','drug2','risk_level','interaction_type','mechanism','significance','recommendations')
        for i in range(len(df)):
            d1 = df.loc[i, "drug1"].strip().lower()
            d2 = df.loc[i, "drug2"].strip().lower()
            if (drug1 == d1 and drug2 == d2) or (drug1 == d2 and drug2 == d1):
                # NOTE: splitting recommendations by newline may fail if CSV uses different separator
                return {
                    "risk": df.loc[i, "risk_level"],
                    "type": df.loc[i, "interaction_type"],
                    "mechanism": df.loc[i, "mechanism"],
                    "significance": df.loc[i, "significance"],
                    "recommendations": df.loc[i, "recommendations"].split("\n")
                }
    except Exception as e:
        print(f"[Clinical Lookup Error] {e}")
        # TODO: add logging to file for debugging
    return None
