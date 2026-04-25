"""
All 26 specialist doctor agents defined as a data-driven registry.

Each entry contains:
  role               — official title (e.g. "Cardiologist")
  conditions         — conditions this specialty treats
  resources          — equipment/beds/drugs they typically request
  protocols          — clinical protocols they follow
  consultation_partners — specialties they commonly call for second opinion
  urgency_weight     — 0-1, how often their cases are high/critical urgency
  deception_profile  — which adversarial strategies are realistic for this specialty
"""

SPECIALISTS: dict[str, dict] = {

    "Cardiology": {
        "role": "Cardiologist",
        "conditions": [
            "myocardial_infarction", "heart_failure", "arrhythmia",
            "angina", "cardiomyopathy", "pericarditis", "cardiac_arrest",
        ],
        "resources": [
            "ECG_machine", "echocardiogram", "cardiac_catheterization_lab",
            "defibrillator", "cardiac_ICU_bed", "troponin_assay",
            "coronary_stent", "pacemaker",
        ],
        "protocols": ["ACLS", "STEMI_protocol", "heart_failure_management", "anticoagulation"],
        "consultation_partners": ["Anesthesiology", "Vascular Surgery", "Radiology", "Hematology", "Nephrology"],
        "urgency_weight": 0.75,
        "deception_profile": ["inflation", "masking"],
    },

    "Neurology": {
        "role": "Neurologist",
        "conditions": [
            "stroke", "epilepsy", "migraine", "multiple_sclerosis",
            "Parkinson_disease", "dementia", "neuropathy", "meningitis",
        ],
        "resources": [
            "MRI_brain", "CT_scan_head", "EEG_unit", "neurology_ICU_bed",
            "tPA_drug", "lumbar_puncture_kit", "nerve_conduction_study",
        ],
        "protocols": ["stroke_protocol", "seizure_management", "tPA_thrombolysis"],
        "consultation_partners": ["Neurosurgery", "Radiology", "Psychiatry", "Infectious Disease"],
        "urgency_weight": 0.65,
        "deception_profile": ["masking", "inflation"],
    },

    "Neurosurgery": {
        "role": "Neurosurgeon",
        "conditions": [
            "brain_tumor", "subdural_hematoma", "epidural_hematoma",
            "spinal_cord_injury", "hydrocephalus", "cerebral_aneurysm",
        ],
        "resources": [
            "neurosurgery_OR_suite", "neurosurgical_ICU_bed", "craniotomy_kit",
            "ICP_monitor", "spinal_fixation_hardware", "neurosurgical_microscope",
        ],
        "protocols": ["craniotomy_protocol", "spinal_surgery_protocol", "ICP_management"],
        "consultation_partners": ["Neurology", "Anesthesiology", "Radiology", "Oncology"],
        "urgency_weight": 0.85,
        "deception_profile": ["inflation", "collusion"],
    },

    "Orthopedics": {
        "role": "Orthopedic Surgeon",
        "conditions": [
            "fracture", "joint_replacement", "spinal_stenosis",
            "ACL_tear", "osteoporosis", "osteomyelitis", "scoliosis",
        ],
        "resources": [
            "orthopedics_OR_suite", "X_ray_unit", "bone_fixation_hardware",
            "cast_supplies", "orthopedic_bed", "arthroscopy_kit",
            "prosthetic_joint",
        ],
        "protocols": ["fracture_reduction", "joint_replacement_protocol", "DVT_prophylaxis"],
        "consultation_partners": ["Anesthesiology", "Radiology", "Vascular Surgery", "Rheumatology"],
        "urgency_weight": 0.45,
        "deception_profile": ["inflation", "masking"],
    },

    "Pediatrics": {
        "role": "Pediatrician",
        "conditions": [
            "respiratory_infection", "febrile_seizure", "dehydration",
            "failure_to_thrive", "neonatal_jaundice", "kawasaki_disease",
            "RSV_bronchiolitis",
        ],
        "resources": [
            "pediatric_ward_bed", "nebulizer", "pediatric_IV_kit",
            "phototherapy_unit", "pediatric_ventilator", "oral_rehydration_kit",
        ],
        "protocols": ["pediatric_resuscitation", "neonatal_protocol", "fever_management_pediatric"],
        "consultation_partners": ["Neurology", "Hematology", "Endocrinology", "Infectious Disease"],
        "urgency_weight": 0.55,
        "deception_profile": ["inflation", "ghost"],
    },

    "Gynecology": {
        "role": "Gynecologist",
        "conditions": [
            "ovarian_cyst", "endometriosis", "uterine_fibroid",
            "cervical_cancer", "PCOS", "pelvic_inflammatory_disease",
        ],
        "resources": [
            "laparoscopy_kit", "gynecology_OR_suite", "pelvic_ultrasound",
            "gynecology_bed", "hysteroscopy_kit", "colposcopy_unit",
        ],
        "protocols": ["laparoscopic_surgery", "hormone_therapy", "colposcopy_protocol"],
        "consultation_partners": ["Oncology", "Radiology", "Pathology", "Obstetrics"],
        "urgency_weight": 0.35,
        "deception_profile": ["masking", "inflation"],
    },

    "Obstetrics": {
        "role": "Obstetrician",
        "conditions": [
            "high_risk_pregnancy", "preeclampsia", "placenta_previa",
            "gestational_diabetes", "preterm_labor", "ectopic_pregnancy",
            "postpartum_hemorrhage",
        ],
        "resources": [
            "delivery_suite", "fetal_monitor", "NICU_bed",
            "epidural_kit", "cesarean_kit", "magnesium_sulfate_drip",
            "neonatal_resuscitation_kit",
        ],
        "protocols": ["labor_management", "c_section_protocol", "preeclampsia_protocol", "PPH_protocol"],
        "consultation_partners": ["Anesthesiology", "Pediatrics", "Nephrology", "Hematology"],
        "urgency_weight": 0.70,
        "deception_profile": ["inflation", "masking"],
    },

    "Dermatology": {
        "role": "Dermatologist",
        "conditions": [
            "melanoma", "psoriasis", "eczema", "cellulitis",
            "contact_dermatitis", "pemphigus", "basal_cell_carcinoma",
        ],
        "resources": [
            "dermatology_procedure_room", "biopsy_kit",
            "phototherapy_unit", "dermatology_bed",
            "MOHS_surgery_kit", "cryotherapy_unit",
        ],
        "protocols": ["skin_biopsy_protocol", "MOHS_surgery", "phototherapy_protocol"],
        "consultation_partners": ["Oncology", "Pathology", "Rheumatology", "Plastic Surgery"],
        "urgency_weight": 0.20,
        "deception_profile": ["inflation"],
    },

    "Ophthalmology": {
        "role": "Ophthalmologist",
        "conditions": [
            "glaucoma", "retinal_detachment", "cataracts",
            "diabetic_retinopathy", "macular_degeneration",
            "corneal_ulcer", "uveitis",
        ],
        "resources": [
            "ophthalmology_OR_suite", "slit_lamp", "OCT_scanner",
            "laser_unit", "intravitreal_injection_kit", "tonometer",
        ],
        "protocols": ["cataract_surgery", "retinal_detachment_repair", "anti_VEGF_protocol"],
        "consultation_partners": ["Neurology", "Endocrinology", "Radiology", "Rheumatology"],
        "urgency_weight": 0.45,
        "deception_profile": ["inflation", "masking"],
    },

    "Otolaryngology": {
        "role": "ENT Specialist",
        "conditions": [
            "tonsillitis", "sinusitis", "hearing_loss",
            "head_neck_cancer", "sleep_apnea", "epistaxis",
            "laryngitis", "otitis_media",
        ],
        "resources": [
            "ENT_OR_suite", "nasal_endoscopy_kit", "audiometry_unit",
            "ENT_bed", "adenoid_removal_kit", "tracheotomy_kit",
        ],
        "protocols": ["tonsillectomy", "sinus_surgery", "sleep_apnea_CPAP_protocol"],
        "consultation_partners": ["Oncology", "Radiology", "Anesthesiology", "Neurology"],
        "urgency_weight": 0.30,
        "deception_profile": ["inflation"],
    },

    "Gastroenterology": {
        "role": "Gastroenterologist",
        "conditions": [
            "GI_bleeding", "inflammatory_bowel_disease", "liver_cirrhosis",
            "pancreatitis", "colorectal_cancer", "GERD", "hepatitis",
        ],
        "resources": [
            "endoscopy_unit", "colonoscopy_kit", "GI_ward_bed",
            "liver_biopsy_kit", "ERCP_kit", "abdominal_ultrasound",
        ],
        "protocols": ["upper_GI_endoscopy", "ERCP_protocol", "colonoscopy", "variceal_banding"],
        "consultation_partners": ["Oncology", "Radiology", "Pathology", "Infectious Disease", "Vascular Surgery"],
        "urgency_weight": 0.55,
        "deception_profile": ["masking", "inflation"],
    },

    "Pulmonology": {
        "role": "Pulmonologist",
        "conditions": [
            "COPD", "pneumonia", "pulmonary_embolism", "lung_cancer",
            "asthma", "respiratory_failure", "ARDS", "pleural_effusion",
        ],
        "resources": [
            "mechanical_ventilator", "bronchoscopy_kit", "pulmonology_ICU_bed",
            "oxygen_concentrator", "CPAP_machine", "chest_drain_kit",
            "pulmonary_function_test_unit",
        ],
        "protocols": ["mechanical_ventilation", "ARDS_protocol", "bronchoscopy", "PE_thrombolysis"],
        "consultation_partners": ["Cardiology", "Oncology", "Infectious Disease", "Anesthesiology"],
        "urgency_weight": 0.72,
        "deception_profile": ["inflation", "masking"],
    },

    "Nephrology": {
        "role": "Nephrologist",
        "conditions": [
            "acute_kidney_injury", "chronic_kidney_disease",
            "glomerulonephritis", "nephrotic_syndrome",
            "renal_calculi", "dialysis_dependent_failure",
        ],
        "resources": [
            "dialysis_machine", "nephrology_ward_bed", "kidney_biopsy_kit",
            "hemodialysis_catheter", "peritoneal_dialysis_kit",
            "renal_ultrasound",
        ],
        "protocols": ["hemodialysis", "peritoneal_dialysis", "AKI_management", "renal_transplant_prep"],
        "consultation_partners": ["Cardiology", "Urology", "Endocrinology", "Obstetrics"],
        "urgency_weight": 0.55,
        "deception_profile": ["inflation", "masking"],
    },

    "Urology": {
        "role": "Urologist",
        "conditions": [
            "kidney_stones", "prostate_cancer", "bladder_cancer",
            "urinary_tract_infection", "urinary_incontinence", "BPH",
            "testicular_torsion",
        ],
        "resources": [
            "urology_OR_suite", "cystoscopy_kit", "lithotripsy_unit",
            "urology_ward_bed", "urodynamics_equipment", "robotic_surgery_system",
        ],
        "protocols": ["lithotripsy", "TURP", "radical_prostatectomy", "cystoscopy_protocol"],
        "consultation_partners": ["Nephrology", "Oncology", "Radiology", "Anesthesiology"],
        "urgency_weight": 0.35,
        "deception_profile": ["inflation"],
    },

    "Endocrinology": {
        "role": "Endocrinologist",
        "conditions": [
            "diabetes_mellitus", "thyroid_disorder", "adrenal_insufficiency",
            "pituitary_tumor", "osteoporosis", "Cushings_syndrome",
            "hyperparathyroidism",
        ],
        "resources": [
            "endocrinology_ward_bed", "glucose_monitor", "insulin_pump",
            "thyroid_biopsy_kit", "DEXA_scan", "hormone_assay_kit",
        ],
        "protocols": ["insulin_protocol", "thyroid_ablation", "adrenal_crisis_management"],
        "consultation_partners": ["Cardiology", "Nephrology", "Ophthalmology", "Neurology", "Oncology"],
        "urgency_weight": 0.30,
        "deception_profile": ["masking", "inflation"],
    },

    "Oncology": {
        "role": "Oncologist",
        "conditions": [
            "breast_cancer", "lung_cancer", "colon_cancer", "lymphoma",
            "leukemia", "brain_tumor", "ovarian_cancer", "pancreatic_cancer",
        ],
        "resources": [
            "chemotherapy_infusion_chair", "oncology_ward_bed",
            "bone_marrow_biopsy_kit", "radiation_therapy_unit",
            "targeted_therapy_drug", "immunotherapy_drug",
        ],
        "protocols": ["chemotherapy_protocol", "bone_marrow_transplant", "palliative_care", "immunotherapy_protocol"],
        "consultation_partners": ["Hematology", "Radiology", "Pathology", "Neurosurgery", "Gastroenterology"],
        "urgency_weight": 0.55,
        "deception_profile": ["inflation", "masking", "collusion"],
    },

    "Hematology": {
        "role": "Hematologist",
        "conditions": [
            "anemia", "hemophilia", "thrombocytopenia", "leukemia",
            "sickle_cell_disease", "coagulation_disorder", "DVT", "aplastic_anemia",
        ],
        "resources": [
            "blood_transfusion_unit", "bone_marrow_aspiration_kit",
            "hematology_ward_bed", "coagulation_factor_VIII",
            "platelet_transfusion_unit", "apheresis_machine",
        ],
        "protocols": ["transfusion_protocol", "anticoagulation_management", "bone_marrow_biopsy", "sickle_cell_protocol"],
        "consultation_partners": ["Oncology", "Cardiology", "Neurosurgery", "Obstetrics"],
        "urgency_weight": 0.55,
        "deception_profile": ["masking", "inflation"],
    },

    "Rheumatology": {
        "role": "Rheumatologist",
        "conditions": [
            "rheumatoid_arthritis", "lupus", "gout",
            "ankylosing_spondylitis", "vasculitis", "sjogrens_syndrome",
            "fibromyalgia",
        ],
        "resources": [
            "rheumatology_ward_bed", "joint_aspiration_kit",
            "biologic_therapy_infusion", "synovial_fluid_analysis_kit",
            "anti_CCP_assay",
        ],
        "protocols": ["DMARDs_protocol", "biologic_therapy", "joint_injection", "pulse_steroids"],
        "consultation_partners": ["Nephrology", "Cardiology", "Dermatology", "Ophthalmology"],
        "urgency_weight": 0.30,
        "deception_profile": ["masking"],
    },

    "Psychiatry": {
        "role": "Psychiatrist",
        "conditions": [
            "major_depression", "schizophrenia", "bipolar_disorder",
            "PTSD", "substance_abuse", "suicidal_ideation", "OCD",
        ],
        "resources": [
            "psychiatric_ward_bed", "ECT_unit", "antipsychotic_medications",
            "mood_stabilizer_drugs", "psychiatric_assessment_kit",
        ],
        "protocols": ["crisis_intervention", "ECT_protocol", "involuntary_hold", "medication_management"],
        "consultation_partners": ["Neurology", "Endocrinology", "General Medicine"],
        "urgency_weight": 0.35,
        "deception_profile": ["masking", "inflation"],
    },

    "Radiology": {
        "role": "Radiologist",
        "conditions": [
            "imaging_interpretation", "interventional_procedures",
            "diagnostic_support", "image_guided_biopsy",
        ],
        "resources": [
            "MRI_machine", "CT_scanner", "X_ray_unit",
            "angiography_suite", "ultrasound_unit", "PET_scan",
            "interventional_radiology_suite",
        ],
        "protocols": ["contrast_CT_protocol", "MRI_brain_protocol", "interventional_radiology", "biopsy_guidance"],
        "consultation_partners": ["Cardiology", "Neurology", "Oncology", "Neurosurgery", "Vascular Surgery"],
        "urgency_weight": 0.45,
        "deception_profile": ["inflation"],
    },

    "Anesthesiology": {
        "role": "Anesthesiologist",
        "conditions": [
            "pre_operative_assessment", "pain_management",
            "critical_care_sedation", "post_operative_recovery",
        ],
        "resources": [
            "anesthesia_machine", "OR_anesthesia_slot", "ICU_bed",
            "epidural_kit", "opioid_analgesics", "neuromuscular_blockers",
            "ventilator",
        ],
        "protocols": ["general_anesthesia", "regional_anesthesia", "pain_management_protocol", "rapid_sequence_induction"],
        "consultation_partners": ["Cardiology", "Pulmonology", "Neurology", "Hematology"],
        "urgency_weight": 0.65,
        "deception_profile": ["inflation"],
    },

    "General Medicine": {
        "role": "General Physician",
        "conditions": [
            "hypertension", "type2_diabetes", "upper_respiratory_infection",
            "fever_of_unknown_origin", "hyperlipidemia", "malaria",
            "typhoid", "general_weakness",
        ],
        "resources": [
            "general_ward_bed", "basic_blood_panel", "X_ray_chest",
            "IV_fluids", "antibiotics", "antipyretics",
        ],
        "protocols": ["fever_management", "hypertension_protocol", "antibiotic_empiric_therapy"],
        "consultation_partners": ["Cardiology", "Endocrinology", "Pulmonology", "Neurology", "Infectious Disease"],
        "urgency_weight": 0.30,
        "deception_profile": ["inflation", "ghost"],
    },

    "Pathology": {
        "role": "Pathologist",
        "conditions": [
            "biopsy_analysis", "autopsy", "lab_diagnostics",
            "histopathology", "cytopathology", "forensic_pathology",
        ],
        "resources": [
            "pathology_lab", "biopsy_processing_kit", "microscopy_unit",
            "immunohistochemistry_kit", "flow_cytometry_unit",
            "molecular_pathology_kit",
        ],
        "protocols": ["biopsy_processing", "frozen_section", "IHC_staining"],
        "consultation_partners": ["Oncology", "Dermatology", "Gastroenterology", "Hematology"],
        "urgency_weight": 0.20,
        "deception_profile": ["inflation"],
    },

    "Plastic Surgery": {
        "role": "Plastic Surgeon",
        "conditions": [
            "burn_injury", "reconstructive_surgery", "cleft_palate",
            "hand_injury", "post_mastectomy_reconstruction", "keloid",
        ],
        "resources": [
            "plastic_surgery_OR_suite", "skin_graft_kit",
            "microsurgery_equipment", "tissue_expander",
            "burn_dressing_kit", "laser_resurfacing_unit",
        ],
        "protocols": ["burn_management", "reconstructive_surgery_protocol", "skin_grafting", "microsurgery_protocol"],
        "consultation_partners": ["Anesthesiology", "Vascular Surgery", "Dermatology", "Infectious Disease"],
        "urgency_weight": 0.45,
        "deception_profile": ["inflation"],
    },

    "Vascular Surgery": {
        "role": "Vascular Surgeon",
        "conditions": [
            "aortic_aneurysm", "peripheral_artery_disease",
            "deep_vein_thrombosis", "carotid_stenosis",
            "limb_ischemia", "mesenteric_ischemia",
        ],
        "resources": [
            "vascular_OR_suite", "angiography_suite", "vascular_ICU_bed",
            "vascular_stent_kit", "bypass_graft_kit", "doppler_ultrasound",
        ],
        "protocols": ["aortic_repair_protocol", "carotid_endarterectomy", "bypass_surgery", "thrombolysis"],
        "consultation_partners": ["Cardiology", "Radiology", "Anesthesiology", "Hematology", "Nephrology"],
        "urgency_weight": 0.75,
        "deception_profile": ["inflation", "masking"],
    },

    "Infectious Disease": {
        "role": "Infectious Disease Specialist",
        "conditions": [
            "sepsis", "HIV_AIDS", "tuberculosis", "malaria",
            "antibiotic_resistant_infection", "COVID_19",
            "hepatitis_B", "hepatitis_C", "endocarditis",
        ],
        "resources": [
            "isolation_room", "infectious_disease_ward_bed",
            "IV_broad_spectrum_antibiotics", "antiviral_medications",
            "blood_culture_kit", "PPE_kit", "antifungal_drugs",
        ],
        "protocols": ["sepsis_bundle", "antibiotic_stewardship", "isolation_protocol", "contact_tracing"],
        "consultation_partners": ["Pulmonology", "Hematology", "Nephrology", "Radiology", "General Medicine"],
        "urgency_weight": 0.65,
        "deception_profile": ["inflation", "ghost"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Symptom → Specialist mappings (used by Router Agent)
# ─────────────────────────────────────────────────────────────────────────────

SYMPTOM_SPECIALIST_MAP: dict[str, list[str]] = {
    "chest pain":               ["Cardiology", "Pulmonology"],
    "palpitations":             ["Cardiology"],
    "shortness of breath":      ["Pulmonology", "Cardiology"],
    "coughing blood":           ["Pulmonology", "Oncology"],
    "stroke":                   ["Neurology", "Neurosurgery", "Radiology"],
    "seizure":                  ["Neurology"],
    "headache":                 ["Neurology"],
    "memory loss":              ["Neurology", "Psychiatry"],
    "head trauma":              ["Neurosurgery", "Neurology", "Radiology"],
    "spinal injury":            ["Neurosurgery", "Orthopedics"],
    "fracture":                 ["Orthopedics", "Anesthesiology"],
    "joint pain":               ["Orthopedics", "Rheumatology"],
    "back pain":                ["Orthopedics", "Neurology"],
    "pediatric":                ["Pediatrics"],
    "child fever":              ["Pediatrics", "Infectious Disease"],
    "neonatal":                 ["Pediatrics", "Obstetrics"],
    "pelvic pain":              ["Gynecology", "Obstetrics"],
    "pregnancy":                ["Obstetrics", "Gynecology"],
    "labor":                    ["Obstetrics", "Anesthesiology"],
    "skin lesion":              ["Dermatology", "Pathology"],
    "rash":                     ["Dermatology", "Rheumatology", "Infectious Disease"],
    "eye pain":                 ["Ophthalmology"],
    "vision loss":              ["Ophthalmology", "Neurology"],
    "ear pain":                 ["Otolaryngology"],
    "throat pain":              ["Otolaryngology"],
    "hearing loss":             ["Otolaryngology"],
    "abdominal pain":           ["Gastroenterology", "General Medicine"],
    "GI bleeding":              ["Gastroenterology", "Vascular Surgery"],
    "jaundice":                 ["Gastroenterology", "Infectious Disease"],
    "kidney failure":           ["Nephrology"],
    "urinary problems":         ["Urology", "Nephrology"],
    "blood in urine":           ["Urology", "Nephrology"],
    "diabetes":                 ["Endocrinology", "General Medicine"],
    "thyroid":                  ["Endocrinology"],
    "cancer":                   ["Oncology", "Radiology", "Pathology"],
    "tumor":                    ["Oncology", "Radiology"],
    "leukemia":                 ["Hematology", "Oncology"],
    "bleeding disorder":        ["Hematology"],
    "anemia":                   ["Hematology", "General Medicine"],
    "joint swelling":           ["Rheumatology", "Orthopedics"],
    "psychiatric symptoms":     ["Psychiatry"],
    "depression":               ["Psychiatry"],
    "hallucinations":           ["Psychiatry", "Neurology"],
    "imaging needed":           ["Radiology"],
    "pre-operative":            ["Anesthesiology"],
    "pain management":          ["Anesthesiology"],
    "fever":                    ["General Medicine", "Infectious Disease"],
    "general weakness":         ["General Medicine"],
    "biopsy":                   ["Pathology"],
    "burn":                     ["Plastic Surgery", "Infectious Disease"],
    "reconstruction":           ["Plastic Surgery"],
    "vascular":                 ["Vascular Surgery", "Cardiology"],
    "aneurysm":                 ["Vascular Surgery", "Radiology"],
    "sepsis":                   ["Infectious Disease", "General Medicine"],
    "infection":                ["Infectious Disease", "General Medicine"],
    "HIV":                      ["Infectious Disease"],
    "tuberculosis":             ["Infectious Disease", "Pulmonology"],
}


# Specific condition → primary specialist (one-to-one, highest-priority match)
CONDITION_SPECIALIST_MAP: dict[str, str] = {
    "myocardial_infarction":    "Cardiology",
    "heart_failure":            "Cardiology",
    "arrhythmia":               "Cardiology",
    "stroke":                   "Neurology",
    "epilepsy":                 "Neurology",
    "brain_tumor":              "Neurosurgery",
    "subdural_hematoma":        "Neurosurgery",
    "fracture":                 "Orthopedics",
    "RSV_bronchiolitis":        "Pediatrics",
    "preeclampsia":             "Obstetrics",
    "high_risk_pregnancy":      "Obstetrics",
    "melanoma":                 "Dermatology",
    "retinal_detachment":       "Ophthalmology",
    "tonsillitis":              "Otolaryngology",
    "liver_cirrhosis":          "Gastroenterology",
    "GI_bleeding":              "Gastroenterology",
    "COPD":                     "Pulmonology",
    "pulmonary_embolism":       "Pulmonology",
    "acute_kidney_injury":      "Nephrology",
    "kidney_stones":            "Urology",
    "diabetes_mellitus":        "Endocrinology",
    "thyroid_disorder":         "Endocrinology",
    "breast_cancer":            "Oncology",
    "lung_cancer":              "Oncology",
    "hemophilia":               "Hematology",
    "leukemia":                 "Hematology",
    "rheumatoid_arthritis":     "Rheumatology",
    "lupus":                    "Rheumatology",
    "schizophrenia":            "Psychiatry",
    "major_depression":         "Psychiatry",
    "sepsis":                   "Infectious Disease",
    "tuberculosis":             "Infectious Disease",
    "aortic_aneurysm":         "Vascular Surgery",
    "burn_injury":              "Plastic Surgery",
    "biopsy_analysis":          "Pathology",
}


def get_specialist(name: str) -> dict:
    """Return specialist config dict; raises KeyError if name is not in registry."""
    if name not in SPECIALISTS:
        raise KeyError(f"Unknown specialist: '{name}'. Valid: {list(SPECIALISTS.keys())}")
    return SPECIALISTS[name]
