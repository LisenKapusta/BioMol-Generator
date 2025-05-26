import pandas as pd
import os
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MCS
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()

class MoleculeEvaluator:
    def __init__(self, input_file_path, output_file_path):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.df = None
        self.results = {}

    def load_data(self):
        if not os.path.exists(self.input_file_path):
            raise FileNotFoundError(f"File not found: {self.input_file_path}")
        self.df = pd.read_csv(self.input_file_path)
        self.df = self.df.dropna(subset='Generated_SMILES')

    def validate_molecules(self):
        def is_valid_smiles(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                return mol is not None
            except:
                return False

        self.df["Is_Valid"] = self.df["Generated_SMILES"].progress_apply(is_valid_smiles)

    def calculate_properties(self):
        def calc_props(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"LogP": None, "Molecular Weight": None}
            return {
                "LogP": Descriptors.MolLogP(mol),
                "Molecular Weight": Descriptors.MolWt(mol)
            }

        self.df["Properties"] = self.df["Generated_SMILES"].progress_apply(calc_props)
        self.df["LogP"] = self.df["Properties"].progress_apply(lambda x: x["LogP"])
        self.df["Molecular_Weight"] = self.df["Properties"].progress_apply(lambda x: x["Molecular Weight"])

    def calculate_tanimoto_similarity(self):
        valid_labels_dict = self.df.groupby("Target")["Drug"].progress_apply(set).to_dict()

        def tanimoto(row):
            pred_smiles = row["Generated_SMILES"]
            pred_mol = Chem.MolFromSmiles(pred_smiles)
            if pred_mol is None:
                return 0.0

            max_sim = 0.0
            true_smiles_list = valid_labels_dict.get(row["Target"], set())

            for true_smiles in true_smiles_list:
                true_mol = Chem.MolFromSmiles(true_smiles)
                if true_mol is None:
                    continue
                fp_true = AllChem.GetMorganFingerprint(true_mol, 2)
                fp_pred = AllChem.GetMorganFingerprint(pred_mol, 2)
                sim = DataStructs.TanimotoSimilarity(fp_true, fp_pred)
                max_sim = max(max_sim, sim)

            return max_sim

        self.df["Tanimoto_Similarity"] = self.df.progress_apply(tanimoto, axis=1)
        self.df["Is_Acceptable_Tanimoto"] = self.df["Tanimoto_Similarity"] >= 0.8

    def calculate_ecfp_similarity(self, radius=2, col_name="ECFP4_Similarity"):
        """Вычисляет сходство на основе MorganFingerprint (ECFP4 или ECFP6)"""
        valid_labels_dict = self.df.groupby("Target")["Drug"].progress_apply(set).to_dict()

        def ecfp_similarity(row):
            pred_smiles = row["Generated_SMILES"]
            pred_mol = Chem.MolFromSmiles(pred_smiles)
            if pred_mol is None:
                return 0.0

            max_sim = 0.0
            true_smiles_list = valid_labels_dict.get(row["Target"], set())

            for true_smiles in true_smiles_list:
                true_mol = Chem.MolFromSmiles(true_smiles)
                if true_mol is None:
                    continue
                fp_true = AllChem.GetMorganFingerprint(true_mol, radius)
                fp_pred = AllChem.GetMorganFingerprint(pred_mol, radius)
                sim = DataStructs.TanimotoSimilarity(fp_true, fp_pred)
                max_sim = max(max_sim, sim)

            return max_sim

        self.df[col_name] = self.df.progress_apply(ecfp_similarity, axis=1)

    def calculate_mcs_similarity(self):
        """Вычисляет схожесть через Max Common Substructure"""
        valid_labels_dict = self.df.groupby("Target")["Drug"].progress_apply(set).to_dict()

        def mcs_score(row):
            pred_smiles = row["Generated_SMILES"]
            pred_mol = Chem.MolFromSmiles(pred_smiles)
            if pred_mol is None:
                return 0.0

            max_score = 0.0
            true_smiles_list = valid_labels_dict.get(row["Target"], set())

            for true_smiles in true_smiles_list:
                true_mol = Chem.MolFromSmiles(true_smiles)
                if true_mol is None:
                    continue
                try:
                    result = MCS.FindMCS([pred_mol, true_mol], timeout=5)
                    if result.numAtoms == 0:
                        score = 0.0
                    else:
                        score = result.numAtoms / max(pred_mol.GetNumAtoms(), true_mol.GetNumAtoms())
                    max_score = max(max_score, score)
                except:
                    continue

            return max_score

        self.df["MCS_Ratio"] = self.df.progress_apply(mcs_score, axis=1)
        self.df["Is_Acceptable_MCS"] = self.df["MCS_Ratio"] >= 0.7

    def generate_summary(self):
        valid_molecules = self.df["Is_Valid"].sum()
        total_molecules = len(self.df)
        valid_ratio = valid_molecules / total_molecules * 100

        acceptable_by_tanimoto = self.df["Is_Acceptable_Tanimoto"].sum()
        ratio_tanimoto = acceptable_by_tanimoto / total_molecules * 100

        acceptable_by_mcs = self.df["Is_Acceptable_MCS"].sum()
        ratio_mcs = acceptable_by_mcs / total_molecules * 100

        avg_tanimoto = self.df["Tanimoto_Similarity"].mean()
        avg_mcs = self.df["MCS_Ratio"].mean()
        avg_logp = self.df["LogP"].mean()
        avg_mw = self.df["Molecular_Weight"].mean()

        self.results = {
            "Valid_Molecules": valid_molecules,
            "Total_Molecules": total_molecules,
            "Valid_Ratio": round(valid_ratio, 2),
            "Acceptable_Tanimoto": acceptable_by_tanimoto,
            "Ratio_Tanimoto": round(ratio_tanimoto, 2),
            "Acceptable_MCS": acceptable_by_mcs,
            "Ratio_MCS": round(ratio_mcs, 2),
            "Average_Tanimoto": round(avg_tanimoto, 2),
            "Average_MCS": round(avg_mcs, 2),
            "Average_LogP": round(avg_logp, 2),
            "Average_Molecular_Weight": round(avg_mw, 2),
        }

    def save_results(self):
        self.df.to_csv(self.output_file_path, index=False)
        print(f"Results saved to {self.output_file_path}")

    def plot_distributions(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # LogP
        axes[0, 0].hist(self.df["LogP"].dropna(), bins=20, color="blue", edgecolor="black")
        axes[0, 0].set_title("Distribution of LogP")

        # Molecular Weight
        axes[0, 1].hist(self.df["Molecular_Weight"].dropna(), bins=20, color="green", edgecolor="black")
        axes[0, 1].set_title("Distribution of Molecular Weight")

        # Tanimoto Similarity
        axes[1, 0].hist(self.df["Tanimoto_Similarity"].dropna(), bins=20, color="orange", edgecolor="black")
        axes[1, 0].set_title("Tanimoto Similarity Distribution")

        # MCS Ratio
        axes[1, 1].hist(self.df["MCS_Ratio"].dropna(), bins=20, color="purple", edgecolor="black")
        axes[1, 1].set_title("Max Common Substructure Ratio")

        plt.tight_layout()
        plt.show()

    def run(self):
        self.load_data()
        self.validate_molecules()
        self.calculate_properties()
        self.calculate_tanimoto_similarity()
        self.calculate_ecfp_similarity(radius=2, col_name="ECFP4_Similarity")
        self.calculate_ecfp_similarity(radius=3, col_name="ECFP6_Similarity")
        self.calculate_mcs_similarity()
        self.generate_summary()
        self.save_results()
        self.plot_distributions()

        return self.results, self.df