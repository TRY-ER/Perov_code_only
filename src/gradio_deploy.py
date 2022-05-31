# this scripts will depoloy the ml model with lite environment on browser with local server

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from scipy import stats as st

# this parameter defines the number of folds available in the divided_trained_models directory
fold_limit = 1

col_data = joblib.load("../deposition/col_data_pred.z")

input_list = []
use_col = []

#tesint the code 
for col in col_data.keys():
    if col != "PCE_categorical":
        use_col.append(col)
        if col_data[col]["type"] == "categorical":
            input_list.append(gr.Dropdown(choices=col_data[col]["u_val"].tolist(),
                                          label = col))
        elif col_data[col]["type"] == "numeric":
            input_list.append(gr.Number(value=col_data[col]["min"],
                                        label = f"{col} ::: enter value between {col_data[col]['min']} and {col_data[col]['max']}"))


def predict(Cell_area_measured_numeric,
           Cell_architecture_cat,
           Substrate_stack_sequence_0,
           Substrate_stack_sequence_1,
           Substrate_stack_sequence_2,
           Substrate_stack_sequence_3,
           Substrate_stack_sequence_4,
           ETL_stack_sequence_0,
           ETL_stack_sequence_1,
           ETL_stack_sequence_2,
           ETL_stack_sequence_3,
           ETL_stack_sequence_4,
           ETL_stack_sequence_5,
           ETL_stack_sequence_6,
           ETL_deposition_procedure_0,
           ETL_deposition_procedure_1,
           ETL_deposition_procedure_2,
           ETL_deposition_procedure_3,
           ETL_deposition_procedure_4,
           ETL_deposition_procedure_5,
           ETL_deposition_procedure_6,
           Perovskite_composition_a_ions_0,
           Perovskite_composition_a_ions_1,
           Perovskite_composition_a_ions_2,
           Perovskite_composition_a_ions_3,
           Perovskite_composition_a_ions_coefficients_0,
           Perovskite_composition_a_ions_coefficients_1,
           Perovskite_composition_a_ions_coefficients_2,
           Perovskite_composition_a_ions_coefficients_3,
           Perovskite_composition_b_ions_0,
           Perovskite_composition_b_ions_1,
           Perovskite_composition_b_ions_2,
           Perovskite_composition_b_ions_3,
           Perovskite_composition_b_ions_coefficients_0,
           Perovskite_composition_b_ions_coefficients_1,
           Perovskite_composition_b_ions_coefficients_2,
           Perovskite_composition_b_ions_coefficients_3,
           Perovskite_composition_c_ions_0,
           Perovskite_composition_c_ions_1,
           Perovskite_composition_c_ions_2,
           Perovskite_composition_c_ions_3,
           Perovskite_composition_c_ions_coefficients_0,
           Perovskite_composition_c_ions_coefficients_1,
           Perovskite_composition_c_ions_coefficients_2,
           Perovskite_composition_c_ions_coefficients_3,
           Perovskite_composition_inorganic_bool,
           Perovskite_composition_leadfree_bool,
           Perovskite_band_gap_graded_bool,
           Perovskite_deposition_number_of_deposition_steps_numeric,
           Perovskite_deposition_procedure_0,
           Perovskite_deposition_procedure_1,
           Perovskite_deposition_procedure_2,
           Perovskite_deposition_procedure_3,
           Perovskite_deposition_procedure_4,
           Perovskite_deposition_procedure_5,
           Perovskite_deposition_aggregation_state_of_reactants_0,
           Perovskite_deposition_aggregation_state_of_reactants_1,
           Perovskite_deposition_aggregation_state_of_reactants_2,
           Perovskite_deposition_aggregation_state_of_reactants_3,
           Perovskite_deposition_aggregation_state_of_reactants_4,
           Perovskite_deposition_aggregation_state_of_reactants_5,
           Perovskite_deposition_synthesis_atmosphere_0,
           Perovskite_deposition_synthesis_atmosphere_1,
           Perovskite_deposition_synthesis_atmosphere_2,
           Perovskite_deposition_synthesis_atmosphere_3,
           Perovskite_deposition_synthesis_atmosphere_4,
           Perovskite_deposition_synthesis_atmosphere_5,
           Perovskite_deposition_solvents_0,
           Perovskite_deposition_solvents_1,
           Perovskite_deposition_solvents_2,
           Perovskite_deposition_solvents_3,
           Perovskite_deposition_solvents_4,
           Perovskite_deposition_solvents_5,
           Perovskite_deposition_solvents_mixing_ratios_0,
           Perovskite_deposition_solvents_mixing_ratios_1,
           Perovskite_deposition_solvents_mixing_ratios_2,
           Perovskite_deposition_solvents_mixing_ratios_3,
           Perovskite_deposition_solvents_mixing_ratios_4,
           Perovskite_deposition_solvents_mixing_ratios_5,
           Perovskite_deposition_quenching_induced_crystallisation_bool,
           Perovskite_deposition_thermal_annealing_temperature_0,
           Perovskite_deposition_thermal_annealing_temperature_1,
           Perovskite_deposition_thermal_annealing_time_0,
           Perovskite_deposition_thermal_annealing_time_1,
           HTL_stack_sequence_0,
           HTL_stack_sequence_1,
           HTL_stack_sequence_2,
           HTL_additives_compounds_0,
           HTL_additives_compounds_1,
           HTL_additives_compounds_2,
           HTL_additives_compounds_3,
           HTL_additives_compounds_4,
           HTL_deposition_procedure_0,
           HTL_deposition_procedure_1,
           HTL_deposition_procedure_2,
           HTL_deposition_procedure_3,
           HTL_deposition_procedure_4,
           HTL_deposition_procedure_5,
           Backcontact_stack_sequence_0,
           Backcontact_stack_sequence_1,
           Backcontact_stack_sequence_2,
           Backcontact_stack_sequence_3,
           Backcontact_stack_sequence_4,
           Backcontact_thickness_list_0,
           Backcontact_thickness_list_1,
           Backcontact_thickness_list_2,
           Backcontact_thickness_list_3,
           Backcontact_thickness_list_4,
           Backcontact_deposition_procedure_0,
           Backcontact_deposition_procedure_1,
           Backcontact_deposition_procedure_2,
           Backcontact_deposition_procedure_3,
           Backcontact_deposition_procedure_4,
           Encapsulation_bool):
    main_vals= []
    try:
        lbl_enc = joblib.load('.../outputs/smote_label_enc/Cell_area_measured_numeric.z')
        Cell_area_measured_numeric = lbl_enc.transform([Cell_area_measured_numeric])
        main_vals.append(*Cell_area_measured_numeric)
    except:main_vals.append(Cell_area_measured_numeric)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Cell_architecture_cat.z')
        Cell_architecture_cat = lbl_enc.transform([Cell_architecture_cat])
        main_vals.append(*Cell_architecture_cat)
    except:main_vals.append(Cell_architecture_cat)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Substrate_stack_sequence_0.z')
        Substrate_stack_sequence_0 = lbl_enc.transform([Substrate_stack_sequence_0])
        main_vals.append(*Substrate_stack_sequence_0)
    except:main_vals.append(Substrate_stack_sequence_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Substrate_stack_sequence_1.z')
        Substrate_stack_sequence_1 = lbl_enc.transform([Substrate_stack_sequence_1])
        main_vals.append(*Substrate_stack_sequence_1)
    except:main_vals.append(Substrate_stack_sequence_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Substrate_stack_sequence_2.z')
        Substrate_stack_sequence_2 = lbl_enc.transform([Substrate_stack_sequence_2])
        main_vals.append(*Substrate_stack_sequence_2)
    except:main_vals.append(Substrate_stack_sequence_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Substrate_stack_sequence_3.z')
        Substrate_stack_sequence_3 = lbl_enc.transform([Substrate_stack_sequence_3])
        main_vals.append(*Substrate_stack_sequence_3)
    except:main_vals.append(Substrate_stack_sequence_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Substrate_stack_sequence_4.z')
        Substrate_stack_sequence_4 = lbl_enc.transform([Substrate_stack_sequence_4])
        main_vals.append(*Substrate_stack_sequence_4)
    except:main_vals.append(Substrate_stack_sequence_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_stack_sequence_0.z')
        ETL_stack_sequence_0 = lbl_enc.transform([ETL_stack_sequence_0])
        main_vals.append(*ETL_stack_sequence_0)
    except:main_vals.append(ETL_stack_sequence_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_stack_sequence_1.z')
        ETL_stack_sequence_1 = lbl_enc.transform([ETL_stack_sequence_1])
        main_vals.append(*ETL_stack_sequence_1)
    except:main_vals.append(ETL_stack_sequence_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_stack_sequence_2.z')
        ETL_stack_sequence_2 = lbl_enc.transform([ETL_stack_sequence_2])
        main_vals.append(*ETL_stack_sequence_2)
    except:main_vals.append(ETL_stack_sequence_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_stack_sequence_3.z')
        ETL_stack_sequence_3 = lbl_enc.transform([ETL_stack_sequence_3])
        main_vals.append(*ETL_stack_sequence_3)
    except:main_vals.append(ETL_stack_sequence_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_stack_sequence_4.z')
        ETL_stack_sequence_4 = lbl_enc.transform([ETL_stack_sequence_4])
        main_vals.append(*ETL_stack_sequence_4)
    except:main_vals.append(ETL_stack_sequence_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_stack_sequence_5.z')
        ETL_stack_sequence_5 = lbl_enc.transform([ETL_stack_sequence_5])
        main_vals.append(*ETL_stack_sequence_5)
    except:main_vals.append(ETL_stack_sequence_5)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_stack_sequence_6.z')
        ETL_stack_sequence_6 = lbl_enc.transform([ETL_stack_sequence_6])
        main_vals.append(*ETL_stack_sequence_6)
    except:main_vals.append(ETL_stack_sequence_6)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_deposition_procedure_0.z')
        ETL_deposition_procedure_0 = lbl_enc.transform([ETL_deposition_procedure_0])
        main_vals.append(*ETL_deposition_procedure_0)
    except:main_vals.append(ETL_deposition_procedure_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_deposition_procedure_1.z')
        ETL_deposition_procedure_1 = lbl_enc.transform([ETL_deposition_procedure_1])
        main_vals.append(*ETL_deposition_procedure_1)
    except:main_vals.append(ETL_deposition_procedure_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_deposition_procedure_2.z')
        ETL_deposition_procedure_2 = lbl_enc.transform([ETL_deposition_procedure_2])
        main_vals.append(*ETL_deposition_procedure_2)
    except:main_vals.append(ETL_deposition_procedure_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_deposition_procedure_3.z')
        ETL_deposition_procedure_3 = lbl_enc.transform([ETL_deposition_procedure_3])
        main_vals.append(*ETL_deposition_procedure_3)
    except:main_vals.append(ETL_deposition_procedure_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_deposition_procedure_4.z')
        ETL_deposition_procedure_4 = lbl_enc.transform([ETL_deposition_procedure_4])
        main_vals.append(*ETL_deposition_procedure_4)
    except:main_vals.append(ETL_deposition_procedure_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_deposition_procedure_5.z')
        ETL_deposition_procedure_5 = lbl_enc.transform([ETL_deposition_procedure_5])
        main_vals.append(*ETL_deposition_procedure_5)
    except:main_vals.append(ETL_deposition_procedure_5)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/ETL_deposition_procedure_6.z')
        ETL_deposition_procedure_6 = lbl_enc.transform([ETL_deposition_procedure_6])
        main_vals.append(*ETL_deposition_procedure_6)
    except:main_vals.append(ETL_deposition_procedure_6)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_a_ions_0.z')
        Perovskite_composition_a_ions_0 = lbl_enc.transform([Perovskite_composition_a_ions_0])
        main_vals.append(*Perovskite_composition_a_ions_0)
    except:main_vals.append(Perovskite_composition_a_ions_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_a_ions_1.z')
        Perovskite_composition_a_ions_1 = lbl_enc.transform([Perovskite_composition_a_ions_1])
        main_vals.append(*Perovskite_composition_a_ions_1)
    except:main_vals.append(Perovskite_composition_a_ions_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_a_ions_2.z')
        Perovskite_composition_a_ions_2 = lbl_enc.transform([Perovskite_composition_a_ions_2])
        main_vals.append(*Perovskite_composition_a_ions_2)
    except:main_vals.append(Perovskite_composition_a_ions_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_a_ions_3.z')
        Perovskite_composition_a_ions_3 = lbl_enc.transform([Perovskite_composition_a_ions_3])
        main_vals.append(*Perovskite_composition_a_ions_3)
    except:main_vals.append(Perovskite_composition_a_ions_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_a_ions_coefficients_0.z')
        Perovskite_composition_a_ions_coefficients_0 = lbl_enc.transform([Perovskite_composition_a_ions_coefficients_0])
        main_vals.append(*Perovskite_composition_a_ions_coefficients_0)
    except:main_vals.append(Perovskite_composition_a_ions_coefficients_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_a_ions_coefficients_1.z')
        Perovskite_composition_a_ions_coefficients_1 = lbl_enc.transform([Perovskite_composition_a_ions_coefficients_1])
        main_vals.append(*Perovskite_composition_a_ions_coefficients_1)
    except:main_vals.append(Perovskite_composition_a_ions_coefficients_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_a_ions_coefficients_2.z')
        Perovskite_composition_a_ions_coefficients_2 = lbl_enc.transform([Perovskite_composition_a_ions_coefficients_2])
        main_vals.append(*Perovskite_composition_a_ions_coefficients_2)
    except:main_vals.append(Perovskite_composition_a_ions_coefficients_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_a_ions_coefficients_3.z')
        Perovskite_composition_a_ions_coefficients_3 = lbl_enc.transform([Perovskite_composition_a_ions_coefficients_3])
        main_vals.append(*Perovskite_composition_a_ions_coefficients_3)
    except:main_vals.append(Perovskite_composition_a_ions_coefficients_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_b_ions_0.z')
        Perovskite_composition_b_ions_0 = lbl_enc.transform([Perovskite_composition_b_ions_0])
        main_vals.append(*Perovskite_composition_b_ions_0)
    except:main_vals.append(Perovskite_composition_b_ions_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_b_ions_1.z')
        Perovskite_composition_b_ions_1 = lbl_enc.transform([Perovskite_composition_b_ions_1])
        main_vals.append(*Perovskite_composition_b_ions_1)
    except:main_vals.append(Perovskite_composition_b_ions_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_b_ions_2.z')
        Perovskite_composition_b_ions_2 = lbl_enc.transform([Perovskite_composition_b_ions_2])
        main_vals.append(*Perovskite_composition_b_ions_2)
    except:main_vals.append(Perovskite_composition_b_ions_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_b_ions_3.z')
        Perovskite_composition_b_ions_3 = lbl_enc.transform([Perovskite_composition_b_ions_3])
        main_vals.append(*Perovskite_composition_b_ions_3)
    except:main_vals.append(Perovskite_composition_b_ions_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_b_ions_coefficients_0.z')
        Perovskite_composition_b_ions_coefficients_0 = lbl_enc.transform([Perovskite_composition_b_ions_coefficients_0])
        main_vals.append(*Perovskite_composition_b_ions_coefficients_0)
    except:main_vals.append(Perovskite_composition_b_ions_coefficients_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_b_ions_coefficients_1.z')
        Perovskite_composition_b_ions_coefficients_1 = lbl_enc.transform([Perovskite_composition_b_ions_coefficients_1])
        main_vals.append(*Perovskite_composition_b_ions_coefficients_1)
    except:main_vals.append(Perovskite_composition_b_ions_coefficients_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_b_ions_coefficients_2.z')
        Perovskite_composition_b_ions_coefficients_2 = lbl_enc.transform([Perovskite_composition_b_ions_coefficients_2])
        main_vals.append(*Perovskite_composition_b_ions_coefficients_2)
    except:main_vals.append(Perovskite_composition_b_ions_coefficients_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_b_ions_coefficients_3.z')
        Perovskite_composition_b_ions_coefficients_3 = lbl_enc.transform([Perovskite_composition_b_ions_coefficients_3])
        main_vals.append(*Perovskite_composition_b_ions_coefficients_3)
    except:main_vals.append(Perovskite_composition_b_ions_coefficients_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_c_ions_0.z')
        Perovskite_composition_c_ions_0 = lbl_enc.transform([Perovskite_composition_c_ions_0])
        main_vals.append(*Perovskite_composition_c_ions_0)
    except:main_vals.append(Perovskite_composition_c_ions_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_c_ions_1.z')
        Perovskite_composition_c_ions_1 = lbl_enc.transform([Perovskite_composition_c_ions_1])
        main_vals.append(*Perovskite_composition_c_ions_1)
    except:main_vals.append(Perovskite_composition_c_ions_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_c_ions_2.z')
        Perovskite_composition_c_ions_2 = lbl_enc.transform([Perovskite_composition_c_ions_2])
        main_vals.append(*Perovskite_composition_c_ions_2)
    except:main_vals.append(Perovskite_composition_c_ions_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_c_ions_3.z')
        Perovskite_composition_c_ions_3 = lbl_enc.transform([Perovskite_composition_c_ions_3])
        main_vals.append(*Perovskite_composition_c_ions_3)
    except:main_vals.append(Perovskite_composition_c_ions_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_c_ions_coefficients_0.z')
        Perovskite_composition_c_ions_coefficients_0 = lbl_enc.transform([Perovskite_composition_c_ions_coefficients_0])
        main_vals.append(*Perovskite_composition_c_ions_coefficients_0)
    except:main_vals.append(Perovskite_composition_c_ions_coefficients_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_c_ions_coefficients_1.z')
        Perovskite_composition_c_ions_coefficients_1 = lbl_enc.transform([Perovskite_composition_c_ions_coefficients_1])
        main_vals.append(*Perovskite_composition_c_ions_coefficients_1)
    except:main_vals.append(Perovskite_composition_c_ions_coefficients_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_c_ions_coefficients_2.z')
        Perovskite_composition_c_ions_coefficients_2 = lbl_enc.transform([Perovskite_composition_c_ions_coefficients_2])
        main_vals.append(*Perovskite_composition_c_ions_coefficients_2)
    except:main_vals.append(Perovskite_composition_c_ions_coefficients_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_c_ions_coefficients_3.z')
        Perovskite_composition_c_ions_coefficients_3 = lbl_enc.transform([Perovskite_composition_c_ions_coefficients_3])
        main_vals.append(*Perovskite_composition_c_ions_coefficients_3)
    except:main_vals.append(Perovskite_composition_c_ions_coefficients_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_inorganic_bool.z')
        Perovskite_composition_inorganic_bool = lbl_enc.transform([Perovskite_composition_inorganic_bool])
        main_vals.append(*Perovskite_composition_inorganic_bool)
    except:main_vals.append(Perovskite_composition_inorganic_bool)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_composition_leadfree_bool.z')
        Perovskite_composition_leadfree_bool = lbl_enc.transform([Perovskite_composition_leadfree_bool])
        main_vals.append(*Perovskite_composition_leadfree_bool)
    except:main_vals.append(Perovskite_composition_leadfree_bool)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_band_gap_graded_bool.z')
        Perovskite_band_gap_graded_bool = lbl_enc.transform([Perovskite_band_gap_graded_bool])
        main_vals.append(*Perovskite_band_gap_graded_bool)
    except:main_vals.append(Perovskite_band_gap_graded_bool)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_number_of_deposition_steps_numeric.z')
        Perovskite_deposition_number_of_deposition_steps_numeric = lbl_enc.transform([Perovskite_deposition_number_of_deposition_steps_numeric])
        main_vals.append(*Perovskite_deposition_number_of_deposition_steps_numeric)
    except:main_vals.append(Perovskite_deposition_number_of_deposition_steps_numeric)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_procedure_0.z')
        Perovskite_deposition_procedure_0 = lbl_enc.transform([Perovskite_deposition_procedure_0])
        main_vals.append(*Perovskite_deposition_procedure_0)
    except:main_vals.append(Perovskite_deposition_procedure_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_procedure_1.z')
        Perovskite_deposition_procedure_1 = lbl_enc.transform([Perovskite_deposition_procedure_1])
        main_vals.append(*Perovskite_deposition_procedure_1)
    except:main_vals.append(Perovskite_deposition_procedure_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_procedure_2.z')
        Perovskite_deposition_procedure_2 = lbl_enc.transform([Perovskite_deposition_procedure_2])
        main_vals.append(*Perovskite_deposition_procedure_2)
    except:main_vals.append(Perovskite_deposition_procedure_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_procedure_3.z')
        Perovskite_deposition_procedure_3 = lbl_enc.transform([Perovskite_deposition_procedure_3])
        main_vals.append(*Perovskite_deposition_procedure_3)
    except:main_vals.append(Perovskite_deposition_procedure_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_procedure_4.z')
        Perovskite_deposition_procedure_4 = lbl_enc.transform([Perovskite_deposition_procedure_4])
        main_vals.append(*Perovskite_deposition_procedure_4)
    except:main_vals.append(Perovskite_deposition_procedure_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_procedure_5.z')
        Perovskite_deposition_procedure_5 = lbl_enc.transform([Perovskite_deposition_procedure_5])
        main_vals.append(*Perovskite_deposition_procedure_5)
    except:main_vals.append(Perovskite_deposition_procedure_5)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_aggregation_state_of_reactants_0.z')
        Perovskite_deposition_aggregation_state_of_reactants_0 = lbl_enc.transform([Perovskite_deposition_aggregation_state_of_reactants_0])
        main_vals.append(*Perovskite_deposition_aggregation_state_of_reactants_0)
    except:main_vals.append(Perovskite_deposition_aggregation_state_of_reactants_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_aggregation_state_of_reactants_1.z')
        Perovskite_deposition_aggregation_state_of_reactants_1 = lbl_enc.transform([Perovskite_deposition_aggregation_state_of_reactants_1])
        main_vals.append(*Perovskite_deposition_aggregation_state_of_reactants_1)
    except:main_vals.append(Perovskite_deposition_aggregation_state_of_reactants_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_aggregation_state_of_reactants_2.z')
        Perovskite_deposition_aggregation_state_of_reactants_2 = lbl_enc.transform([Perovskite_deposition_aggregation_state_of_reactants_2])
        main_vals.append(*Perovskite_deposition_aggregation_state_of_reactants_2)
    except:main_vals.append(Perovskite_deposition_aggregation_state_of_reactants_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_aggregation_state_of_reactants_3.z')
        Perovskite_deposition_aggregation_state_of_reactants_3 = lbl_enc.transform([Perovskite_deposition_aggregation_state_of_reactants_3])
        main_vals.append(*Perovskite_deposition_aggregation_state_of_reactants_3)
    except:main_vals.append(Perovskite_deposition_aggregation_state_of_reactants_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_aggregation_state_of_reactants_4.z')
        Perovskite_deposition_aggregation_state_of_reactants_4 = lbl_enc.transform([Perovskite_deposition_aggregation_state_of_reactants_4])
        main_vals.append(*Perovskite_deposition_aggregation_state_of_reactants_4)
    except:main_vals.append(Perovskite_deposition_aggregation_state_of_reactants_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_aggregation_state_of_reactants_5.z')
        Perovskite_deposition_aggregation_state_of_reactants_5 = lbl_enc.transform([Perovskite_deposition_aggregation_state_of_reactants_5])
        main_vals.append(*Perovskite_deposition_aggregation_state_of_reactants_5)
    except:main_vals.append(Perovskite_deposition_aggregation_state_of_reactants_5)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_synthesis_atmosphere_0.z')
        Perovskite_deposition_synthesis_atmosphere_0 = lbl_enc.transform([Perovskite_deposition_synthesis_atmosphere_0])
        main_vals.append(*Perovskite_deposition_synthesis_atmosphere_0)
    except:main_vals.append(Perovskite_deposition_synthesis_atmosphere_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_synthesis_atmosphere_1.z')
        Perovskite_deposition_synthesis_atmosphere_1 = lbl_enc.transform([Perovskite_deposition_synthesis_atmosphere_1])
        main_vals.append(*Perovskite_deposition_synthesis_atmosphere_1)
    except:main_vals.append(Perovskite_deposition_synthesis_atmosphere_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_synthesis_atmosphere_2.z')
        Perovskite_deposition_synthesis_atmosphere_2 = lbl_enc.transform([Perovskite_deposition_synthesis_atmosphere_2])
        main_vals.append(*Perovskite_deposition_synthesis_atmosphere_2)
    except:main_vals.append(Perovskite_deposition_synthesis_atmosphere_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_synthesis_atmosphere_3.z')
        Perovskite_deposition_synthesis_atmosphere_3 = lbl_enc.transform([Perovskite_deposition_synthesis_atmosphere_3])
        main_vals.append(*Perovskite_deposition_synthesis_atmosphere_3)
    except:main_vals.append(Perovskite_deposition_synthesis_atmosphere_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_synthesis_atmosphere_4.z')
        Perovskite_deposition_synthesis_atmosphere_4 = lbl_enc.transform([Perovskite_deposition_synthesis_atmosphere_4])
        main_vals.append(*Perovskite_deposition_synthesis_atmosphere_4)
    except:main_vals.append(Perovskite_deposition_synthesis_atmosphere_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_synthesis_atmosphere_5.z')
        Perovskite_deposition_synthesis_atmosphere_5 = lbl_enc.transform([Perovskite_deposition_synthesis_atmosphere_5])
        main_vals.append(*Perovskite_deposition_synthesis_atmosphere_5)
    except:main_vals.append(Perovskite_deposition_synthesis_atmosphere_5)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_solvents_0.z')
        Perovskite_deposition_solvents_0 = lbl_enc.transform([Perovskite_deposition_solvents_0])
        main_vals.append(*Perovskite_deposition_solvents_0)
    except:main_vals.append(Perovskite_deposition_solvents_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_solvents_1.z')
        Perovskite_deposition_solvents_1 = lbl_enc.transform([Perovskite_deposition_solvents_1])
        main_vals.append(*Perovskite_deposition_solvents_1)
    except:main_vals.append(Perovskite_deposition_solvents_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_solvents_2.z')
        Perovskite_deposition_solvents_2 = lbl_enc.transform([Perovskite_deposition_solvents_2])
        main_vals.append(*Perovskite_deposition_solvents_2)
    except:main_vals.append(Perovskite_deposition_solvents_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_solvents_3.z')
        Perovskite_deposition_solvents_3 = lbl_enc.transform([Perovskite_deposition_solvents_3])
        main_vals.append(*Perovskite_deposition_solvents_3)
    except:main_vals.append(Perovskite_deposition_solvents_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_solvents_4.z')
        Perovskite_deposition_solvents_4 = lbl_enc.transform([Perovskite_deposition_solvents_4])
        main_vals.append(*Perovskite_deposition_solvents_4)
    except:main_vals.append(Perovskite_deposition_solvents_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_solvents_5.z')
        Perovskite_deposition_solvents_5 = lbl_enc.transform([Perovskite_deposition_solvents_5])
        main_vals.append(*Perovskite_deposition_solvents_5)
    except:main_vals.append(Perovskite_deposition_solvents_5)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_solvents_mixing_ratios_0.z')
        Perovskite_deposition_solvents_mixing_ratios_0 = lbl_enc.transform([Perovskite_deposition_solvents_mixing_ratios_0])
        main_vals.append(*Perovskite_deposition_solvents_mixing_ratios_0)
    except:main_vals.append(Perovskite_deposition_solvents_mixing_ratios_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_solvents_mixing_ratios_1.z')
        Perovskite_deposition_solvents_mixing_ratios_1 = lbl_enc.transform([Perovskite_deposition_solvents_mixing_ratios_1])
        main_vals.append(*Perovskite_deposition_solvents_mixing_ratios_1)
    except:main_vals.append(Perovskite_deposition_solvents_mixing_ratios_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_solvents_mixing_ratios_2.z')
        Perovskite_deposition_solvents_mixing_ratios_2 = lbl_enc.transform([Perovskite_deposition_solvents_mixing_ratios_2])
        main_vals.append(*Perovskite_deposition_solvents_mixing_ratios_2)
    except:main_vals.append(Perovskite_deposition_solvents_mixing_ratios_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_solvents_mixing_ratios_3.z')
        Perovskite_deposition_solvents_mixing_ratios_3 = lbl_enc.transform([Perovskite_deposition_solvents_mixing_ratios_3])
        main_vals.append(*Perovskite_deposition_solvents_mixing_ratios_3)
    except:main_vals.append(Perovskite_deposition_solvents_mixing_ratios_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_solvents_mixing_ratios_4.z')
        Perovskite_deposition_solvents_mixing_ratios_4 = lbl_enc.transform([Perovskite_deposition_solvents_mixing_ratios_4])
        main_vals.append(*Perovskite_deposition_solvents_mixing_ratios_4)
    except:main_vals.append(Perovskite_deposition_solvents_mixing_ratios_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_solvents_mixing_ratios_5.z')
        Perovskite_deposition_solvents_mixing_ratios_5 = lbl_enc.transform([Perovskite_deposition_solvents_mixing_ratios_5])
        main_vals.append(*Perovskite_deposition_solvents_mixing_ratios_5)
    except:main_vals.append(Perovskite_deposition_solvents_mixing_ratios_5)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_quenching_induced_crystallisation_bool.z')
        Perovskite_deposition_quenching_induced_crystallisation_bool = lbl_enc.transform([Perovskite_deposition_quenching_induced_crystallisation_bool])
        main_vals.append(*Perovskite_deposition_quenching_induced_crystallisation_bool)
    except:main_vals.append(Perovskite_deposition_quenching_induced_crystallisation_bool)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_thermal_annealing_temperature_0.z')
        Perovskite_deposition_thermal_annealing_temperature_0 = lbl_enc.transform([Perovskite_deposition_thermal_annealing_temperature_0])
        main_vals.append(*Perovskite_deposition_thermal_annealing_temperature_0)
    except:main_vals.append(Perovskite_deposition_thermal_annealing_temperature_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_thermal_annealing_temperature_1.z')
        Perovskite_deposition_thermal_annealing_temperature_1 = lbl_enc.transform([Perovskite_deposition_thermal_annealing_temperature_1])
        main_vals.append(*Perovskite_deposition_thermal_annealing_temperature_1)
    except:main_vals.append(Perovskite_deposition_thermal_annealing_temperature_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_thermal_annealing_time_0.z')
        Perovskite_deposition_thermal_annealing_time_0 = lbl_enc.transform([Perovskite_deposition_thermal_annealing_time_0])
        main_vals.append(*Perovskite_deposition_thermal_annealing_time_0)
    except:main_vals.append(Perovskite_deposition_thermal_annealing_time_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Perovskite_deposition_thermal_annealing_time_1.z')
        Perovskite_deposition_thermal_annealing_time_1 = lbl_enc.transform([Perovskite_deposition_thermal_annealing_time_1])
        main_vals.append(*Perovskite_deposition_thermal_annealing_time_1)
    except:main_vals.append(Perovskite_deposition_thermal_annealing_time_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_stack_sequence_0.z')
        HTL_stack_sequence_0 = lbl_enc.transform([HTL_stack_sequence_0])
        main_vals.append(*HTL_stack_sequence_0)
    except:main_vals.append(HTL_stack_sequence_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_stack_sequence_1.z')
        HTL_stack_sequence_1 = lbl_enc.transform([HTL_stack_sequence_1])
        main_vals.append(*HTL_stack_sequence_1)
    except:main_vals.append(HTL_stack_sequence_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_stack_sequence_2.z')
        HTL_stack_sequence_2 = lbl_enc.transform([HTL_stack_sequence_2])
        main_vals.append(*HTL_stack_sequence_2)
    except:main_vals.append(HTL_stack_sequence_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_additives_compounds_0.z')
        HTL_additives_compounds_0 = lbl_enc.transform([HTL_additives_compounds_0])
        main_vals.append(*HTL_additives_compounds_0)
    except:main_vals.append(HTL_additives_compounds_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_additives_compounds_1.z')
        HTL_additives_compounds_1 = lbl_enc.transform([HTL_additives_compounds_1])
        main_vals.append(*HTL_additives_compounds_1)
    except:main_vals.append(HTL_additives_compounds_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_additives_compounds_2.z')
        HTL_additives_compounds_2 = lbl_enc.transform([HTL_additives_compounds_2])
        main_vals.append(*HTL_additives_compounds_2)
    except:main_vals.append(HTL_additives_compounds_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_additives_compounds_3.z')
        HTL_additives_compounds_3 = lbl_enc.transform([HTL_additives_compounds_3])
        main_vals.append(*HTL_additives_compounds_3)
    except:main_vals.append(HTL_additives_compounds_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_additives_compounds_4.z')
        HTL_additives_compounds_4 = lbl_enc.transform([HTL_additives_compounds_4])
        main_vals.append(*HTL_additives_compounds_4)
    except:main_vals.append(HTL_additives_compounds_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_deposition_procedure_0.z')
        HTL_deposition_procedure_0 = lbl_enc.transform([HTL_deposition_procedure_0])
        main_vals.append(*HTL_deposition_procedure_0)
    except:main_vals.append(HTL_deposition_procedure_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_deposition_procedure_1.z')
        HTL_deposition_procedure_1 = lbl_enc.transform([HTL_deposition_procedure_1])
        main_vals.append(*HTL_deposition_procedure_1)
    except:main_vals.append(HTL_deposition_procedure_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_deposition_procedure_2.z')
        HTL_deposition_procedure_2 = lbl_enc.transform([HTL_deposition_procedure_2])
        main_vals.append(*HTL_deposition_procedure_2)
    except:main_vals.append(HTL_deposition_procedure_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_deposition_procedure_3.z')
        HTL_deposition_procedure_3 = lbl_enc.transform([HTL_deposition_procedure_3])
        main_vals.append(*HTL_deposition_procedure_3)
    except:main_vals.append(HTL_deposition_procedure_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_deposition_procedure_4.z')
        HTL_deposition_procedure_4 = lbl_enc.transform([HTL_deposition_procedure_4])
        main_vals.append(*HTL_deposition_procedure_4)
    except:main_vals.append(HTL_deposition_procedure_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/HTL_deposition_procedure_5.z')
        HTL_deposition_procedure_5 = lbl_enc.transform([HTL_deposition_procedure_5])
        main_vals.append(*HTL_deposition_procedure_5)
    except:main_vals.append(HTL_deposition_procedure_5)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_stack_sequence_0.z')
        Backcontact_stack_sequence_0 = lbl_enc.transform([Backcontact_stack_sequence_0])
        main_vals.append(*Backcontact_stack_sequence_0)
    except:main_vals.append(Backcontact_stack_sequence_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_stack_sequence_1.z')
        Backcontact_stack_sequence_1 = lbl_enc.transform([Backcontact_stack_sequence_1])
        main_vals.append(*Backcontact_stack_sequence_1)
    except:main_vals.append(Backcontact_stack_sequence_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_stack_sequence_2.z')
        Backcontact_stack_sequence_2 = lbl_enc.transform([Backcontact_stack_sequence_2])
        main_vals.append(*Backcontact_stack_sequence_2)
    except:main_vals.append(Backcontact_stack_sequence_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_stack_sequence_3.z')
        Backcontact_stack_sequence_3 = lbl_enc.transform([Backcontact_stack_sequence_3])
        main_vals.append(*Backcontact_stack_sequence_3)
    except:main_vals.append(Backcontact_stack_sequence_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_stack_sequence_4.z')
        Backcontact_stack_sequence_4 = lbl_enc.transform([Backcontact_stack_sequence_4])
        main_vals.append(*Backcontact_stack_sequence_4)
    except:main_vals.append(Backcontact_stack_sequence_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_thickness_list_0.z')
        Backcontact_thickness_list_0 = lbl_enc.transform([Backcontact_thickness_list_0])
        main_vals.append(*Backcontact_thickness_list_0)
    except:main_vals.append(Backcontact_thickness_list_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_thickness_list_1.z')
        Backcontact_thickness_list_1 = lbl_enc.transform([Backcontact_thickness_list_1])
        main_vals.append(*Backcontact_thickness_list_1)
    except:main_vals.append(Backcontact_thickness_list_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_thickness_list_2.z')
        Backcontact_thickness_list_2 = lbl_enc.transform([Backcontact_thickness_list_2])
        main_vals.append(*Backcontact_thickness_list_2)
    except:main_vals.append(Backcontact_thickness_list_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_thickness_list_3.z')
        Backcontact_thickness_list_3 = lbl_enc.transform([Backcontact_thickness_list_3])
        main_vals.append(*Backcontact_thickness_list_3)
    except:main_vals.append(Backcontact_thickness_list_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_thickness_list_4.z')
        Backcontact_thickness_list_4 = lbl_enc.transform([Backcontact_thickness_list_4])
        main_vals.append(*Backcontact_thickness_list_4)
    except:main_vals.append(Backcontact_thickness_list_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_deposition_procedure_0.z')
        Backcontact_deposition_procedure_0 = lbl_enc.transform([Backcontact_deposition_procedure_0])
        main_vals.append(*Backcontact_deposition_procedure_0)
    except:main_vals.append(Backcontact_deposition_procedure_0)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_deposition_procedure_1.z')
        Backcontact_deposition_procedure_1 = lbl_enc.transform([Backcontact_deposition_procedure_1])
        main_vals.append(*Backcontact_deposition_procedure_1)
    except:main_vals.append(Backcontact_deposition_procedure_1)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_deposition_procedure_2.z')
        Backcontact_deposition_procedure_2 = lbl_enc.transform([Backcontact_deposition_procedure_2])
        main_vals.append(*Backcontact_deposition_procedure_2)
    except:main_vals.append(Backcontact_deposition_procedure_2)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_deposition_procedure_3.z')
        Backcontact_deposition_procedure_3 = lbl_enc.transform([Backcontact_deposition_procedure_3])
        main_vals.append(*Backcontact_deposition_procedure_3)
    except:main_vals.append(Backcontact_deposition_procedure_3)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Backcontact_deposition_procedure_4.z')
        Backcontact_deposition_procedure_4 = lbl_enc.transform([Backcontact_deposition_procedure_4])
        main_vals.append(*Backcontact_deposition_procedure_4)
    except:main_vals.append(Backcontact_deposition_procedure_4)
    try:
        lbl_enc = joblib.load('../outputs/smote_label_enc/Encapsulation_bool.z')
        Encapsulation_bool = lbl_enc.transform([Encapsulation_bool])
        main_vals.append(*Encapsulation_bool)
    except:main_vals.append(Encapsulation_bool)
    # print(main_vals)
    main_vals = np.array(main_vals)
    temp_df  = {}
    for col, value in zip(use_col,main_vals):
        temp_df[col] = value
    temp_df = pd.DataFrame([temp_df])
    scaler = joblib.load("../outputs/scaler/standard_scaler.z")
    scaled_df = scaler.transform(temp_df)
    pred_list = []
    for fold in range(fold_limit):
        model = joblib.load(f"../divided_trained_models/tabnet_adam/models_main/fold_{fold}/{fold}_model.z")
        result = model.predict(scaled_df)
        pred_list.append(result)
    main_result = list(st.mode(pred_list))[0]
    tar_lbl = joblib.load("../outputs/smote_label_enc/PCE_categorical.z")
    result_main = tar_lbl.inverse_transform([int(*main_result)])

    result_list = ["very_low", "low", "high", "higher","highest","best"]
    match_exp = ["5-10", "10-15", "15-20", "20-25", "25-30", ">30"]

    for res,match in zip(result_list,match_exp):
        if res == str(*result_main):
            desc_val = match

    return f"{str(*result_main)} >> [The value is between {desc_val}]"


# print(predict(0.0004, 'nip', 'SLG', 'FTO', 'None', 'None', 'None', 'TiO2-c', 'TiO2-mp', 'None', 'None', 'None', 'None', 'None', 'Spray-pyrolys', 'Spin-coating', 'None', 'None', 'None', 'None', 'None', 'Cs', 'None', 'None', 'None', 0.005, 0.0, 0.0, 0.0, 'Sn', 'None', 'None', 'None', 0.0, 0.0, 0.0, 0.0, 'I', 'None', 'None', 'None', -4.44e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'Space-limited inverse temperature crystallization', 'Ultrasonic spray', 'None', 'None', 'None', 'None', 'Liquid', 'None', 'None', 'None', 'None', 'None', 'N2', 'None', 'None', 'None', 'None', 'None', 'DMSO', 'None', 'None', 'None', 'None', 7.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.55e-15, 0.0, 'Selenium', 'None', 'None', 'Li(CF3SO2)2N', 'TBP', 'None', 'None', 'None', 'Spin-coating', 'None', 'None', 'None', 'None', 'None', 'Au', 'None', 'None', 'None', 'None', 0.5, 0.0, 0.0, 0.0, 0.0, 'Evaporation', 'None', 'None', 'None', 'None', 0.0))


app = gr.Interface(fn = predict,
                   inputs= input_list,
                   outputs = "text").launch()