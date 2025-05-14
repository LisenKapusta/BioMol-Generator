# import pandas as pd
# import os
# from rdkit import Chem
# from rdkit.Chem import Descriptors, AllChem
# from rdkit.DataStructs import TanimotoSimilarity
# import matplotlib.pyplot as plt

# class MoleculeEvaluator:
#     def __init__(self, input_file_path, output_file_path):
#         """
#         Инициализация класса.
#         :param input_file_path: Путь к входному CSV-файлу для анализа.
#         :param output_file_path: Путь для сохранения результирующего файла.
#         """
#         self.input_file_path = input_file_path
#         self.output_file_path = output_file_path
#         self.df = None
#         self.results = {}

#     def load_data(self):
#         """Загрузка данных из CSV-файла."""
#         if not os.path.exists(self.input_file_path):
#             raise FileNotFoundError(f"File not found: {self.input_file_path}")
#         self.df = pd.read_csv(self.input_file_path)
#         self.df = self.df.dropna(subset='Predicted_Label')

#     def validate_molecules(self):
#         """Проверка химической валидности молекул."""
#         def is_valid_smiles(smiles):
#             mol = Chem.MolFromSmiles(smiles)
#             return mol is not None

#         self.df["Is_Valid"] = self.df["Predicted_Label"].apply(is_valid_smiles)

#     def calculate_properties(self):
#         """Вычисление физико-химических свойств молекул."""
#         def calculate_properties(smiles):
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is None:
#                 return None
#             logp = Descriptors.MolLogP(mol)
#             mol_weight = Descriptors.MolWt(mol)
#             return {"LogP": logp, "Molecular Weight": mol_weight}

#         self.df["Properties"] = self.df["Predicted_Label"].apply(calculate_properties)
#         self.df["LogP"] = self.df["Properties"].apply(lambda x: x["LogP"] if x else None)
#         self.df["Molecular_Weight"] = self.df["Properties"].apply(lambda x: x["Molecular Weight"] if x else None)

#     def calculate_tanimoto_similarity(self):
#         """Расчет Tanimoto Similarity для всех допустимых меток."""
#         valid_labels_dict = self.df.groupby("Sequence")["True_Label"].apply(set).to_dict()

#         def calculate_max_similarity(row):
#             mol_pred = Chem.MolFromSmiles(row["Predicted_Label"])
#             if mol_pred is None:
#                 return 0.0

#             sequence = row["Sequence"]
#             true_labels = valid_labels_dict.get(sequence, set())
#             max_similarity = 0.0

#             for true_label in true_labels:
#                 mol_true = Chem.MolFromSmiles(true_label)
#                 if mol_true is not None:
#                     fp_true = AllChem.GetMorganFingerprint(mol_true, 2)
#                     fp_pred = AllChem.GetMorganFingerprint(mol_pred, 2)
#                     similarity = TanimotoSimilarity(fp_true, fp_pred)
#                     max_similarity = max(max_similarity, similarity)

#             return max_similarity

#         self.df["Tanimoto_Similarity"] = self.df.apply(calculate_max_similarity, axis=1)
#         self.df["Is_Acceptable"] = self.df["Tanimoto_Similarity"] >= 0.8

#     def generate_summary(self):
#         """Генерация сводной статистики."""
#         valid_molecules = self.df["Is_Valid"].sum()
#         total_molecules = len(self.df)
#         valid_ratio = valid_molecules / total_molecules * 100

#         acceptable_molecules = self.df["Is_Acceptable"].sum()
#         acceptable_ratio = acceptable_molecules / total_molecules * 100

#         self.results = {
#             "Valid_Molecules": valid_molecules,
#             "Total_Molecules": total_molecules,
#             "Valid_Ratio": valid_ratio,
#             "Acceptable_Molecules": acceptable_molecules,
#             "Acceptable_Ratio": acceptable_ratio
#         }

#     def save_results(self):
#         """Сохранение результатов в CSV-файл."""
#         self.df.to_csv(self.output_file_path, index=False)
#         print(f"Results saved to {self.output_file_path}")

#     def plot_distributions(self):
#         """Генерация графиков распределения свойств."""
#         plt.figure(figsize=(12, 6))

#         # Распределение LogP
#         plt.subplot(1, 2, 1)
#         plt.hist(self.df["LogP"].dropna(), bins=20, color="blue", edgecolor="black")
#         plt.title("Distribution of LogP")
#         plt.xlabel("LogP")
#         plt.ylabel("Frequency")

#         # Распределение молекулярной массы
#         plt.subplot(1, 2, 2)
#         plt.hist(self.df["Molecular_Weight"].dropna(), bins=20, color="green", edgecolor="black")
#         plt.title("Distribution of Molecular Weight")
#         plt.xlabel("Molecular Weight")
#         plt.ylabel("Frequency")

#         plt.tight_layout()
#         plt.show()

#     def run(self):
#         """Основной метод для выполнения всех шагов."""
#         self.load_data()
#         self.validate_molecules()
#         self.calculate_properties()
#         self.calculate_tanimoto_similarity()
#         self.generate_summary()
#         self.save_results()
#         self.plot_distributions()

#         return self.results, self.df

import pandas as pd
import os
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MCS
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt


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
        self.df = self.df.dropna(subset='Predicted_Label')

    def validate_molecules(self):
        def is_valid_smiles(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                return mol is not None
            except:
                return False

        self.df["Is_Valid"] = self.df["Predicted_Label"].apply(is_valid_smiles)

    def calculate_properties(self):
        def calc_props(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"LogP": None, "Molecular Weight": None}
            return {
                "LogP": Descriptors.MolLogP(mol),
                "Molecular Weight": Descriptors.MolWt(mol)
            }

        self.df["Properties"] = self.df["Predicted_Label"].apply(calc_props)
        self.df["LogP"] = self.df["Properties"].apply(lambda x: x["LogP"])
        self.df["Molecular_Weight"] = self.df["Properties"].apply(lambda x: x["Molecular Weight"])

    def calculate_tanimoto_similarity(self):
        valid_labels_dict = self.df.groupby("Sequence")["True_Label"].apply(set).to_dict()

        def tanimoto(row):
            pred_smiles = row["Predicted_Label"]
            pred_mol = Chem.MolFromSmiles(pred_smiles)
            if pred_mol is None:
                return 0.0

            max_sim = 0.0
            true_smiles_list = valid_labels_dict.get(row["Sequence"], set())

            for true_smiles in true_smiles_list:
                true_mol = Chem.MolFromSmiles(true_smiles)
                if true_mol is None:
                    continue
                fp_true = AllChem.GetMorganFingerprint(true_mol, 2)
                fp_pred = AllChem.GetMorganFingerprint(pred_mol, 2)
                sim = DataStructs.TanimotoSimilarity(fp_true, fp_pred)
                max_sim = max(max_sim, sim)

            return max_sim

        self.df["Tanimoto_Similarity"] = self.df.apply(tanimoto, axis=1)
        self.df["Is_Acceptable_Tanimoto"] = self.df["Tanimoto_Similarity"] >= 0.8

    def calculate_ecfp_similarity(self, radius=2, col_name="ECFP4_Similarity"):
        """Вычисляет сходство на основе MorganFingerprint (ECFP4 или ECFP6)"""
        valid_labels_dict = self.df.groupby("Sequence")["True_Label"].apply(set).to_dict()

        def ecfp_similarity(row):
            pred_smiles = row["Predicted_Label"]
            pred_mol = Chem.MolFromSmiles(pred_smiles)
            if pred_mol is None:
                return 0.0

            max_sim = 0.0
            true_smiles_list = valid_labels_dict.get(row["Sequence"], set())

            for true_smiles in true_smiles_list:
                true_mol = Chem.MolFromSmiles(true_smiles)
                if true_mol is None:
                    continue
                fp_true = AllChem.GetMorganFingerprint(true_mol, radius)
                fp_pred = AllChem.GetMorganFingerprint(pred_mol, radius)
                sim = DataStructs.TanimotoSimilarity(fp_true, fp_pred)
                max_sim = max(max_sim, sim)

            return max_sim

        self.df[col_name] = self.df.apply(ecfp_similarity, axis=1)

    def calculate_mcs_similarity(self):
        """Вычисляет схожесть через Max Common Substructure"""
        valid_labels_dict = self.df.groupby("Sequence")["True_Label"].apply(set).to_dict()

        def mcs_score(row):
            pred_smiles = row["Predicted_Label"]
            pred_mol = Chem.MolFromSmiles(pred_smiles)
            if pred_mol is None:
                return 0.0

            max_score = 0.0
            true_smiles_list = valid_labels_dict.get(row["Sequence"], set())

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

        self.df["MCS_Ratio"] = self.df.apply(mcs_score, axis=1)
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