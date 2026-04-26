"""
Phase 2.1 — Generate Synthetic Medical Speech Training Data for Whisper Fine-Tuning

Produces 2,600+ audio/transcript pairs from 26 specialist question banks.
Each phrase is rendered at 3 TTS speed variants (slow/normal/fast) across
2 voice types → 5 variations per phrase.

Output structure:
  notebooks/data/medical_speech/
    train/   (80%)
    val/     (20%)
    metadata.csv

Run:
  python -m notebooks.generate_medical_speech_data

Requirements (already in requirements.txt):
  pyttsx3, scipy

Time: ~15-25 min on CPU for 2,600 files
"""

import os
import csv
import json
import random
import shutil
import tempfile
import threading
from pathlib import Path

# Medical phrase banks per specialty
# 20 questions × 26 specialties = 520 base phrases
# × 5 TTS variants = 2,600 audio files

SPECIALIST_PHRASES: dict[str, list[str]] = {

    "Cardiology": [
        "Does your chest pain radiate to your left arm or jaw?",
        "On a scale of one to ten, how severe is your chest discomfort?",
        "Are you experiencing shortness of breath at rest or only with exertion?",
        "Do you have a history of myocardial infarction or coronary artery disease?",
        "Are you currently taking aspirin, clopidogrel, or any anticoagulant?",
        "Has your electrocardiogram been done recently?",
        "Do you notice any palpitations or irregular heartbeat?",
        "Is there any swelling in your ankles or legs?",
        "Have you ever had an echocardiogram or stress test?",
        "Do you have hypertension or hyperlipidemia?",
        "Are you experiencing diaphoresis along with your chest pain?",
        "Have you had any syncopal episodes or loss of consciousness?",
        "What is your current troponin level?",
        "Are you experiencing any jugular venous distension?",
        "Do you have a family history of sudden cardiac death?",
        "Have you been prescribed nitroglycerin for angina?",
        "Is your pain relieved by rest or sublingual nitrates?",
        "Are you a current smoker or do you have a history of tobacco use?",
        "Have you noticed any orthopnea or paroxysmal nocturnal dyspnea?",
        "What is your current ejection fraction from the last echocardiogram?",
    ],

    "Neurology": [
        "When did you first notice the weakness in your arm or leg?",
        "Is your headache sudden onset, like a thunderclap?",
        "Do you have any slurred speech or facial drooping?",
        "Have you had any seizures or convulsions recently?",
        "Are you experiencing any visual disturbances or diplopia?",
        "Do you have a history of multiple sclerosis or epilepsy?",
        "Is there any numbness or tingling in your extremities?",
        "Have you noticed any cognitive decline or memory impairment?",
        "What medications are you on for your Parkinson disease?",
        "Are you experiencing any tremor at rest or during movement?",
        "Have you had a CT scan or MRI of the brain recently?",
        "Is there any neck stiffness or photophobia with your headache?",
        "Do you have a history of transient ischemic attack or stroke?",
        "Are you taking any anticonvulsant medications like levetiracetam?",
        "Have you noticed any gait instability or frequent falls?",
        "Is the headache unilateral or bilateral?",
        "Do you experience any aura before your migraine episodes?",
        "Are you having any difficulty with swallowing or dysphagia?",
        "Have you been evaluated for lumbar radiculopathy?",
        "Is there any incontinence associated with your neurological symptoms?",
    ],

    "Pulmonology": [
        "How long have you had this shortness of breath?",
        "Do you produce any sputum? If so, what color is it?",
        "Have you been exposed to asbestos or occupational lung irritants?",
        "Are you currently on any bronchodilators or inhaled corticosteroids?",
        "Do you have a history of chronic obstructive pulmonary disease or asthma?",
        "What is your oxygen saturation on room air?",
        "Have you had a recent chest X-ray or CT pulmonary angiography?",
        "Are you experiencing any hemoptysis or coughing up blood?",
        "Do you have wheezing that is relieved with salbutamol?",
        "Have you had a spirometry test to assess your lung function?",
        "Is your dyspnea worse at night or early morning?",
        "Do you have any risk factors for pulmonary embolism such as recent surgery?",
        "Are you a current or former smoker? How many pack-years?",
        "Have you been tested for alpha-one antitrypsin deficiency?",
        "Do you use a continuous positive airway pressure machine for sleep apnea?",
        "Is there any pleuritic chest pain with inspiration?",
        "Have you had a bronchoscopy or bronchoalveolar lavage?",
        "Are you experiencing any fever or night sweats with your cough?",
        "Do you have a history of tuberculosis or recent exposure?",
        "What is your peak expiratory flow rate?",
    ],

    "Endocrinology": [
        "What is your most recent fasting blood glucose level?",
        "Are you currently on insulin, metformin, or any other antidiabetic medication?",
        "Have you been experiencing excessive thirst or polyuria?",
        "What is your most recent HbA1c value?",
        "Do you have a history of thyroid disease or take levothyroxine?",
        "Are you experiencing any heat intolerance, palpitations, or tremors?",
        "Have you been diagnosed with Cushing syndrome or adrenal insufficiency?",
        "Do you have any symptoms of hypoglycemia such as sweating or confusion?",
        "Are you monitoring your blood sugar at home? What are the readings?",
        "Have you had any diabetic ketoacidosis episodes previously?",
        "Do you have any complications from diabetes such as neuropathy or retinopathy?",
        "Have you had thyroid function tests done recently?",
        "Are you experiencing weight gain or fatigue that could indicate hypothyroidism?",
        "Do you have a history of polycystic ovary syndrome or insulin resistance?",
        "Have you been evaluated for metabolic syndrome?",
        "Are you taking any corticosteroids that might affect your glucose levels?",
        "Do you have any family history of type one or type two diabetes?",
        "Have you had a recent urine microalbumin or creatinine ratio test?",
        "Are you on any medications that can cause hyperglycemia?",
        "Have you noticed any changes in your skin such as acanthosis nigricans?",
    ],

    "Orthopedics": [
        "Where exactly is your pain and does it radiate down your leg?",
        "Did the injury happen suddenly or has it been a gradual onset?",
        "Have you had any X-rays or MRI of the affected joint?",
        "Do you have any crepitus or grinding sensation in the joint?",
        "Are you able to bear weight on the affected limb?",
        "Have you tried any physiotherapy or conservative treatment?",
        "Do you have a history of osteoporosis or previous fractures?",
        "Is there any swelling, bruising, or deformity at the injury site?",
        "Have you had any joint replacement surgery previously?",
        "Are you currently taking any nonsteroidal anti-inflammatory drugs?",
        "Do you have any morning stiffness that lasts more than thirty minutes?",
        "Have you been diagnosed with osteoarthritis or rheumatoid arthritis?",
        "Is the pain worse with activity or at rest?",
        "Do you have any neurological symptoms such as numbness in your toes?",
        "Have you had a bone density scan or DEXA scan recently?",
        "Is there locking or giving way of the knee joint?",
        "Do you have a history of DVT or pulmonary embolism after surgery?",
        "Have you had any compartment syndrome or vascular injury?",
        "What is your range of motion in the affected limb?",
        "Have you received any corticosteroid injections for this condition?",
    ],

    "Gastroenterology": [
        "Have you noticed any blood in your stool or melena?",
        "Where is your abdominal pain and does it radiate to your back?",
        "Do you have any difficulty swallowing or odynophagia?",
        "Have you had a colonoscopy or upper endoscopy recently?",
        "Do you have a history of peptic ulcer disease or gastritis?",
        "Are you taking proton pump inhibitors or H2 blockers?",
        "Have you experienced any significant weight loss recently?",
        "Do you have any nausea, vomiting, or symptoms of bowel obstruction?",
        "Is there any jaundice or yellowing of your sclera?",
        "Have you been tested for Helicobacter pylori infection?",
        "Do you have inflammatory bowel disease such as Crohn disease or ulcerative colitis?",
        "Have you had any liver function tests or hepatitis serology?",
        "Is there any change in your bowel habits such as alternating constipation and diarrhea?",
        "Do you have a history of pancreatitis or cholelithiasis?",
        "Are you experiencing any bloating or abdominal distension?",
        "Have you had any hepatitis B or hepatitis C exposure?",
        "Do you drink alcohol? If so, how many units per week?",
        "Have you had an ERCP or MRCP for evaluation of your biliary system?",
        "Is there any family history of colorectal cancer or inflammatory bowel disease?",
        "Have you noticed any pale stools or dark urine?",
    ],

    "Nephrology": [
        "What is your most recent serum creatinine and eGFR?",
        "Are you experiencing any peripheral edema or facial puffiness?",
        "Do you have a history of chronic kidney disease or dialysis?",
        "Have you had any blood or foam in your urine recently?",
        "Are you hypertensive and on any ACE inhibitors or ARBs?",
        "Have you had a renal ultrasound or kidney biopsy?",
        "Do you have diabetes or any other cause of nephropathy?",
        "Are you experiencing any uremic symptoms such as nausea or pruritus?",
        "What is your most recent urine protein to creatinine ratio?",
        "Have you had any acute kidney injury episodes requiring hospitalization?",
        "Do you take any nephrotoxic medications such as NSAIDs or contrast agents?",
        "Is there any family history of polycystic kidney disease?",
        "Have you been evaluated for renovascular hypertension?",
        "Do you have any electrolyte abnormalities such as hyperkalemia?",
        "Have you had a parathyroid hormone level checked for renal osteodystrophy?",
        "Are you on any dietary restrictions for protein or phosphorus?",
        "Have you had urological evaluation for obstructive uropathy?",
        "What is your current fluid and urine output balance?",
        "Are you on any immunosuppressants for glomerulonephritis?",
        "Have you had renal vein thrombosis evaluation given your nephrotic syndrome?",
    ],

    "Oncology": [
        "When was your cancer diagnosis confirmed and what is the histological type?",
        "What is the current stage of your cancer based on TNM classification?",
        "Have you completed any chemotherapy cycles and what regimen are you on?",
        "Are you experiencing any cytopenias from your chemotherapy?",
        "Have you had a PET scan or CT scan for restaging recently?",
        "Do you have any molecular markers such as EGFR or HER2 status?",
        "Are you experiencing any neuropathy from platinum-based chemotherapy?",
        "Have you been evaluated for immunotherapy with checkpoint inhibitors?",
        "Is there any tumor lysis syndrome risk with your current treatment?",
        "Do you have any bone metastases and are you on bisphosphonates?",
        "Are you on any targeted therapy such as imatinib or erlotinib?",
        "Have you had a bone marrow biopsy to assess your disease burden?",
        "Are you experiencing any mucositis or oral complications from treatment?",
        "Do you have febrile neutropenia and are you on prophylactic antibiotics?",
        "Have you been enrolled in any clinical trial for your cancer type?",
        "Is there any BRCA mutation or Lynch syndrome in your family history?",
        "Have you had tumor board review of your case?",
        "Are you experiencing any cachexia or nutritional deficiencies?",
        "Have you completed radiation therapy and what was the field and dose?",
        "Is there any consideration for palliative care or hospice evaluation?",
    ],

    "Psychiatry": [
        "Are you experiencing any auditory or visual hallucinations?",
        "Have you had any thoughts of self-harm or suicidal ideation?",
        "Are you currently on any antidepressants or antipsychotics?",
        "How long have you been experiencing these mood changes?",
        "Do you have a history of bipolar disorder or schizophrenia?",
        "Are you experiencing any panic attacks or severe anxiety?",
        "Have you had any manic or hypomanic episodes?",
        "Do you have any post-traumatic stress disorder symptoms?",
        "Have you been admitted to a psychiatric unit previously?",
        "Are you able to maintain your activities of daily living?",
        "Do you have any substance use disorder or alcohol dependence?",
        "Are you experiencing any obsessive thoughts or compulsive behaviors?",
        "Have you responded to cognitive behavioral therapy in the past?",
        "Do you have any sleep disturbances such as insomnia or hypersomnia?",
        "Are you experiencing any dissociative episodes?",
        "Is there any family history of psychiatric illness?",
        "Are you currently engaged in any psychotherapy or counseling?",
        "Do you feel hopeless about the future?",
        "Have you had a structured suicide risk assessment done?",
        "Are you experiencing any side effects from your current psychiatric medications?",
    ],

    "Infectious Disease": [
        "Have you had any recent travel to malaria-endemic regions?",
        "What is your HIV status and most recent CD4 count?",
        "Are you fully vaccinated including for COVID-19 and influenza?",
        "Do you have any recent animal bites or tick exposures?",
        "Have you been started on empirical antibiotics and what coverage?",
        "What does your blood culture show and is there any bacteremia?",
        "Are you immunocompromised due to HIV, transplant, or chemotherapy?",
        "Have you had any exposure to tuberculosis or a positive tuberculin test?",
        "Is there any surgical site infection or wound dehiscence?",
        "What is your current procalcitonin or C-reactive protein level?",
        "Have you been evaluated for sepsis using the qSOFA criteria?",
        "Do you have any fungal infections such as candidiasis or aspergillosis?",
        "Are you experiencing any urinary symptoms suggesting pyelonephritis?",
        "Have you had a lumbar puncture for meningitis workup?",
        "Is there any hepatitis or liver involvement with your current infection?",
        "Do you have any indwelling devices such as catheters or prosthetic valves?",
        "Have you had MRSA decolonization with mupirocin?",
        "Are you on appropriate prophylaxis for opportunistic infections?",
        "Have you been screened for sexually transmitted infections?",
        "What is the duration and character of your fever?",
    ],

    "Dermatology": [
        "How long have you had this skin lesion and has it changed in size or color?",
        "Does the rash itch, burn, or is it painless?",
        "Have you had any recent changes in medications or new exposures?",
        "Do you have a personal or family history of melanoma or skin cancer?",
        "Have you been treated with topical corticosteroids or antifungals?",
        "Is there any associated joint pain suggesting psoriatic arthritis?",
        "Have you had a skin biopsy for histopathological diagnosis?",
        "Are you applying any sunscreen or photoprotection regularly?",
        "Do you have any autoimmune conditions associated with your skin disease?",
        "Is this lesion ulcerated or has it bled spontaneously?",
        "Have you had patch testing for contact dermatitis?",
        "Do you have a history of atopic dermatitis or eczema?",
        "Is there any mucosal involvement with your skin condition?",
        "Have you had phototherapy or PUVA treatment for your psoriasis?",
        "Are you experiencing hair loss or nail changes along with this rash?",
        "Do you have any pruritus at night suggesting scabies infestation?",
        "Have you been evaluated for bullous pemphigoid or pemphigus vulgaris?",
        "Is there any hyperpigmentation or depigmentation post-inflammation?",
        "Have you used any biologic therapy such as dupilumab for your eczema?",
        "What dermatoscopic findings were noted on your pigmented lesion?",
    ],

    "Rheumatology": [
        "How long does your morning stiffness last?",
        "Are any of your joints swollen, warm, or erythematous?",
        "Have you had your rheumatoid factor or anti-CCP antibody checked?",
        "Do you have any extra-articular manifestations such as dry eyes or mouth?",
        "Are you currently on methotrexate, hydroxychloroquine, or leflunomide?",
        "Have you had any serositis such as pleuritis or pericarditis?",
        "What does your ANA and anti-dsDNA show for lupus monitoring?",
        "Is there any renal involvement with your connective tissue disease?",
        "Have you had any joint aspiration for crystal analysis?",
        "Are you on biologics such as TNF inhibitors or IL-6 blockers?",
        "Do you have any livedo reticularis or Raynaud phenomenon?",
        "Have you been evaluated for antiphospholipid syndrome?",
        "Is there any fibromyalgia component to your pain syndrome?",
        "Have you had any uveitis associated with your arthritis?",
        "Do you have any sacroiliitis on MRI suggesting ankylosing spondylitis?",
        "Are you experiencing any sicca symptoms with your connective tissue disease?",
        "Have you had a skin or muscle biopsy for myositis workup?",
        "What is your current disease activity score for rheumatoid arthritis?",
        "Are you taking folic acid supplementation with your methotrexate?",
        "Have you been screened for latent TB before starting biologic therapy?",
    ],

    "General Medicine": [
        "What brings you to the hospital today and when did it start?",
        "Do you have any chronic medical conditions I should know about?",
        "Are you currently on any prescription medications or supplements?",
        "Do you have any known drug allergies?",
        "Have you had any recent hospitalizations or surgeries?",
        "What is your vaccination status including tetanus and influenza?",
        "Do you smoke, drink alcohol, or use any recreational substances?",
        "Is there any significant family history of heart disease or cancer?",
        "Are you experiencing any fever, night sweats, or unintentional weight loss?",
        "Do you have any difficulty with activities of daily living?",
        "What is your occupation and do you have any occupational exposures?",
        "Are you on any blood thinners or antiplatelet agents?",
        "Have you had any recent laboratory tests or imaging?",
        "Do you have any urinary or bowel symptoms I should be aware of?",
        "Are you experiencing any joint pains or musculoskeletal symptoms?",
        "Have you traveled internationally in the last six months?",
        "Do you have any symptoms of depression or anxiety?",
        "Are you up to date with your age-appropriate cancer screening?",
        "Do you have any symptoms suggesting vitamin D or B12 deficiency?",
        "Is there anything else you would like to tell me about your health?",
    ],

    "Emergency Medicine": [
        "What time did your symptoms start and how rapidly did they progress?",
        "Do you have any chest pain, shortness of breath, or altered mental status?",
        "What is your current GCS score and are you oriented to time place and person?",
        "Have you taken any medications, toxins, or substances in the last 24 hours?",
        "What is your current blood pressure, pulse, and oxygen saturation?",
        "Do you have any active bleeding or trauma?",
        "Are you allergic to any medications including contrast dye?",
        "Have you had any seizures or loss of consciousness?",
        "What is your pain score on a numeric rating scale?",
        "Is there any mechanism of injury for this trauma?",
        "Do you have a history of anaphylaxis or severe allergic reactions?",
        "Are you pregnant or could you be pregnant?",
        "What is your last oral intake?",
        "Do you have any implanted devices such as pacemakers or stents?",
        "Are you on anticoagulation therapy?",
        "Do you have any neurological deficits suggesting stroke?",
        "Have you had any vomiting, diarrhea, or signs of dehydration?",
        "Is there any significant past medical history that I should factor into your emergency care?",
        "Are you experiencing any signs of septic shock such as hypotension or tachycardia?",
        "What is the priority triage level assigned to this patient?",
    ],

    "Hematology": [
        "Are you experiencing any easy bruising or prolonged bleeding from minor cuts?",
        "What is your current hemoglobin, platelet count, and white cell differential?",
        "Have you had a bone marrow biopsy or aspiration?",
        "Are you on any anticoagulation therapy for thromboembolism?",
        "Do you have a history of sickle cell disease or thalassemia?",
        "Have you been evaluated for hemophilia A or B?",
        "Are you experiencing any lymphadenopathy or splenomegaly?",
        "What is your INR and aPTT for coagulopathy assessment?",
        "Have you had any deep vein thrombosis or pulmonary embolism?",
        "Are you on direct oral anticoagulants or vitamin K antagonists?",
        "Do you have any symptoms of hyperviscosity such as visual changes or headache?",
        "Have you been evaluated for myeloproliferative neoplasm?",
        "What does your peripheral blood smear show?",
        "Are you experiencing any petechiae or purpura?",
        "Have you had thrombocytopenic episodes requiring platelet transfusion?",
        "Do you have factor V Leiden or prothrombin gene mutation?",
        "Have you had iron studies or ferritin level checked for anemia workup?",
        "Is there any family history of inherited bleeding disorders?",
        "Have you had a Coombs test for autoimmune hemolytic anemia?",
        "What is your CHADS-VASc score for atrial fibrillation anticoagulation?",
    ],

    "Anesthesiology": [
        "Have you had any previous adverse reactions to general or regional anesthesia?",
        "Are you fasting and when was your last oral intake?",
        "What is your Mallampati classification and predicted airway difficulty?",
        "Do you have any history of malignant hyperthermia?",
        "Are you on any medications that affect coagulation?",
        "Do you have any loose teeth, dental prosthetics, or temporomandibular joint disease?",
        "What is your ASA physical status classification?",
        "Do you have any history of obstructive sleep apnea?",
        "Have you had any awareness under anesthesia in previous procedures?",
        "Are you allergic to latex, soy, eggs, or any anesthetic agents?",
        "What is your current hemodynamic status for regional anesthesia planning?",
        "Do you have any coagulopathy that would contraindicate neuraxial blockade?",
        "Have you been on steroids requiring perioperative stress dosing?",
        "Is there any history of difficult mask ventilation or intubation?",
        "What are your current electrolyte values including potassium?",
        "Do you have any pacemaker or implantable defibrillator that may affect electrocautery?",
        "Have you had any postoperative nausea and vomiting with previous anesthetics?",
        "Is there any indication for neuromonitoring during this procedure?",
        "What is the planned surgical position and duration?",
        "Have you consented to blood transfusion if required intraoperatively?",
    ],

    "Radiology": [
        "Are you allergic to iodinated contrast media?",
        "What is your renal function before contrast-enhanced CT?",
        "Have you had any prior radiation exposure or imaging studies?",
        "Is there any metallic implant that would contraindicate MRI?",
        "Are you claustrophobic and would you require sedation for MRI?",
        "What is the clinical indication for this imaging study?",
        "Have you had a mammogram recently for breast cancer screening?",
        "Is this a diagnostic ultrasound for guided biopsy or aspiration?",
        "Have you had PET-CT imaging for oncology staging?",
        "What was the result of your recent chest X-ray?",
        "Is contrast enhancement required for this abdominal CT scan?",
        "Have you been evaluated with DEXA scan for bone mineral density?",
        "Do you need fluoroscopy-guided intervention such as angioplasty?",
        "What is the clinical suspicion for pulmonary embolism requiring CTPA?",
        "Have you had any MRI of the brain with gadolinium?",
        "Is there any suspicion of aortic dissection requiring urgent CTA?",
        "What protocol should be used for the liver MRI with hepatobiliary contrast?",
        "Is interventional radiology consultation needed for embolization?",
        "Have you had any nuclear medicine studies such as thyroid scan?",
        "What is the BIRADS classification of this breast imaging finding?",
    ],

    "Gynecology": [
        "What is the date of your last menstrual period?",
        "Are you experiencing any abnormal uterine bleeding or intermenstrual spotting?",
        "Have you had a recent cervical smear or colposcopy?",
        "Are you pregnant or could you be pregnant?",
        "Do you have a history of polycystic ovary syndrome or endometriosis?",
        "What contraception are you currently using?",
        "Have you had any previous pelvic inflammatory disease or STIs?",
        "Are you experiencing any pelvic pain or dyspareunia?",
        "Have you completed your HPV vaccination series?",
        "Do you have any symptoms of menopause such as hot flushes or vaginal dryness?",
        "Have you had any ovarian cysts or fibroids diagnosed on ultrasound?",
        "Is there any family history of ovarian or breast cancer with BRCA mutation?",
        "Have you had any ectopic pregnancy or tubal surgery previously?",
        "Are you experiencing any urinary incontinence or prolapse symptoms?",
        "What was the result of your most recent CA-125 level?",
        "Have you had any hysteroscopy or laparoscopy procedures?",
        "Are you on hormone replacement therapy and what formulation?",
        "Do you have any concerns about fertility or conception?",
        "Have you been evaluated for premature ovarian insufficiency?",
        "Is there any postmenopausal bleeding that requires endometrial sampling?",
    ],

    "Urology": [
        "Are you experiencing any hematuria macroscopic or microscopic?",
        "Do you have any lower urinary tract symptoms such as urgency or hesitancy?",
        "What is your most recent PSA level and prostate volume?",
        "Have you had any kidney stones or renal colic previously?",
        "Are you experiencing any erectile dysfunction or ejaculatory problems?",
        "Have you had a urine culture showing any organisms?",
        "Do you have a history of bladder cancer or urothelial carcinoma?",
        "Have you had a cystoscopy or urodynamic study recently?",
        "Is there any hydronephrosis on ultrasound suggesting obstruction?",
        "Are you on alpha-blockers or 5-alpha reductase inhibitors for BPH?",
        "Have you had any testicular pain, swelling, or torsion?",
        "Is there any stress incontinence or overflow incontinence?",
        "Have you had a transrectal ultrasound guided biopsy of the prostate?",
        "Do you have any urethral stricture or recurrent UTIs?",
        "Have you been evaluated for vesicoureteral reflux?",
        "Is there any indication for shock wave lithotripsy or ureteroscopy?",
        "What is your International Prostate Symptom Score?",
        "Have you had any renal cell carcinoma or upper tract urothelial cancer?",
        "Are you experiencing any penile discharge or scrotal pain?",
        "Have you had any incontinence surgery such as tension-free vaginal tape?",
    ],

    "Ophthalmology": [
        "Are you experiencing any sudden loss of vision or flashes of light?",
        "Do you have a history of glaucoma or elevated intraocular pressure?",
        "Are you currently using any topical eye drops such as timolol or latanoprost?",
        "Have you had any recent eye surgery such as cataract or retinal detachment repair?",
        "Do you have diabetic retinopathy and has it been graded recently?",
        "Are you experiencing any double vision or diplopia?",
        "Have you had a slit lamp examination and what were the findings?",
        "Is there any anterior chamber inflammation or uveitis?",
        "What is your current visual acuity in each eye?",
        "Have you been evaluated for age-related macular degeneration?",
        "Do you have any corneal disease such as keratoconus or bullous keratopathy?",
        "Are you on anti-VEGF injections for wet macular degeneration?",
        "Have you had visual field testing for glaucoma monitoring?",
        "Is there any proptosis or exophthalmos suggesting thyroid eye disease?",
        "Have you had fundus photography or OCT imaging recently?",
        "Do you have any symptoms of dry eye disease?",
        "Have you had any chemical or thermal injury to the eye?",
        "Is there any indication for panretinal photocoagulation?",
        "What was your most recent cup to disc ratio on optic nerve assessment?",
        "Have you been evaluated for strabismus or amblyopia?",
    ],

    "Neurosurgery": [
        "What imaging has been done for your intracranial lesion?",
        "Are you experiencing any signs of raised intracranial pressure?",
        "Have you had any recent change in level of consciousness or GCS?",
        "Do you have any focal neurological deficit that has worsened?",
        "Have you been evaluated for lumbar disc herniation with radiculopathy?",
        "Is there any indication for emergency craniotomy?",
        "What is the midline shift on your CT scan?",
        "Are you on prophylactic antiepileptic drugs perioperatively?",
        "Have you had a cerebral angiogram for your intracranial aneurysm?",
        "What does the spine MRI show for your myelopathy?",
        "Is there any cerebrospinal fluid leak or rhinorrhea?",
        "Have you had a ventriculoperitoneal shunt placement?",
        "What is the planned approach for your tumor resection?",
        "Are you a candidate for stereotactic radiosurgery?",
        "Have you had any epidural or subdural hematoma drainage?",
        "Is there any hydrocephalus requiring external ventricular drain placement?",
        "Have you had neuronavigation planning for your surgery?",
        "What are the surgical risks specific to the eloquent cortex proximity?",
        "Have you had intraoperative neurophysiological monitoring planned?",
        "Is there any need for postoperative ICU monitoring?",
    ],

    "Vascular Surgery": [
        "Are you experiencing any claudication or rest pain in your legs?",
        "What is your ankle-brachial index?",
        "Have you had any duplex ultrasound of your carotid arteries?",
        "Do you have any history of aortic aneurysm or dissection?",
        "What was the maximum diameter on your last aortic imaging?",
        "Have you had any peripheral arterial interventions previously?",
        "Do you have any critical limb ischemia with non-healing wounds?",
        "Are you a candidate for endovascular aneurysm repair?",
        "Have you had any deep vein thrombosis treated with anticoagulation?",
        "Is there any varicose vein disease requiring intervention?",
        "What is the Rutherford classification for your peripheral artery disease?",
        "Have you had any carotid endarterectomy or stenting?",
        "Is there any mesenteric ischemia requiring urgent revascularization?",
        "Have you been evaluated for lymphedema or chronic venous insufficiency?",
        "What are your atherosclerotic risk factors including lipid profile?",
        "Is there any indication for femoropopliteal bypass surgery?",
        "Have you had any arteriovenous fistula creation for dialysis access?",
        "Do you have any Raynaud phenomenon suggesting vasospastic disease?",
        "Is there any pseudoaneurysm or arteriovenous fistula post-procedure?",
        "Have you had duplex surveillance of your graft or stent?",
    ],
}

# Fill remaining specialties with placeholder phrases (extend as needed)
REMAINING_SPECIALTIES = [
    "Plastic Surgery", "Oral and Maxillofacial Surgery", "Thoracic Surgery",
]

for _spec in REMAINING_SPECIALTIES:
    SPECIALIST_PHRASES[_spec] = [
        f"Can you describe your main complaint for {_spec.lower()}?",
        "How long have you had these symptoms?",
        "Have you had any previous treatment for this condition?",
        "Are you on any medications related to this problem?",
        "Have you had any imaging or laboratory tests recently?",
        "Is there any relevant family history?",
        "Are you experiencing any pain? If so, please rate it.",
        "Have you had any surgeries in this area before?",
        "Are there any allergies I should know about?",
        "What are your current vital signs?",
        "Has this condition affected your daily activities?",
        "Are you on any blood-thinning medications?",
        "Have you noticed any changes in color, sensation, or function?",
        "Do you have any systemic conditions that may be relevant?",
        "Have you had any complications from previous treatments?",
        "What is your current functional status?",
        "Are you a smoker or do you use tobacco products?",
        "Do you have any implants or prosthetics in this area?",
        "Have you been counselled about surgical risks and alternatives?",
        "What are your expectations from this consultation?",
    ]


# Audio generation with pyttsx3

TTS_VARIANTS = [
    {"rate": 120, "voice_idx": 0},  # slow, voice 0
    {"rate": 150, "voice_idx": 0},  # normal, voice 0
    {"rate": 180, "voice_idx": 0},  # fast, voice 0
    {"rate": 130, "voice_idx": 1},  # medium-slow, voice 1
    {"rate": 165, "voice_idx": 1},  # medium-fast, voice 1
]


def _generate_wav(text: str, out_path: str, rate: int, voice_idx: int) -> bool:
    """Render text to WAV using pyttsx3. Returns True on success."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)

        voices = engine.getProperty("voices")
        if voices and voice_idx < len(voices):
            engine.setProperty("voice", voices[voice_idx].id)

        # pyttsx3 requires the path to exist
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        engine.save_to_file(text, out_path)
        engine.runAndWait()
        engine.stop()
        return Path(out_path).exists()
    except Exception as exc:
        print(f"  [TTS ERROR] {exc} — skipping {out_path}")
        return False


def generate_dataset(
    output_dir: str = "notebooks/data/medical_speech",
    train_split: float = 0.8,
    max_per_specialty: int | None = None,
    dry_run: bool = False,
) -> None:
    """
    Generate the full Whisper fine-tuning dataset.

    Args:
        output_dir:       Root directory for audio files and metadata.
        train_split:      Fraction of examples that go to train/ (rest → val/).
        max_per_specialty: If set, limit to N phrases per specialty (useful for testing).
        dry_run:          If True, print file list without generating audio.
    """
    root      = Path(output_dir)
    train_dir = root / "train"
    val_dir   = root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict] = []
    total_files   = 0
    total_errors  = 0

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Generating medical speech dataset → {root}")
    print(f"Specialties: {len(SPECIALIST_PHRASES)} | Variants per phrase: {len(TTS_VARIANTS)}")

    for spec_name, phrases in SPECIALIST_PHRASES.items():
        spec_slug = spec_name.lower().replace(" ", "_").replace("/", "_")
        print(f"\n  {spec_name} ({len(phrases)} phrases × {len(TTS_VARIANTS)} variants)...")

        phrases_to_use = phrases[:max_per_specialty] if max_per_specialty else phrases

        for q_idx, phrase in enumerate(phrases_to_use, start=1):
            for v_idx, variant in enumerate(TTS_VARIANTS, start=1):
                # Decide split
                split    = "train" if random.random() < train_split else "val"
                out_dir  = train_dir if split == "train" else val_dir
                filename = f"{spec_slug}_q{q_idx:03d}_v{v_idx}.wav"
                wav_path = str(out_dir / filename)
                txt_path = wav_path.replace(".wav", ".txt")

                if not dry_run:
                    ok = _generate_wav(
                        text       = phrase,
                        out_path   = wav_path,
                        rate       = variant["rate"],
                        voice_idx  = variant["voice_idx"],
                    )
                    if ok:
                        # Write transcript alongside the audio
                        Path(txt_path).write_text(phrase, encoding="utf-8")
                        total_files += 1
                    else:
                        total_errors += 1
                else:
                    print(f"    {filename} → '{phrase[:60]}...'")
                    total_files += 1

                metadata_rows.append({
                    "file":      filename,
                    "split":     split,
                    "specialty": spec_name,
                    "q_idx":     q_idx,
                    "v_idx":     v_idx,
                    "rate":      variant["rate"],
                    "voice_idx": variant["voice_idx"],
                    "transcript": phrase,
                })

    #  Write metadata CSV 
    meta_path = root / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
        writer.writeheader()
        writer.writerows(metadata_rows)

    #  Summary 
    train_count = sum(1 for r in metadata_rows if r["split"] == "train")
    val_count   = sum(1 for r in metadata_rows if r["split"] == "val")

    print(f"\n{'─' * 60}")
    print(f"  Dataset complete!")
    print(f"  Total examples  : {len(metadata_rows):,}")
    print(f"  Train           : {train_count:,}")
    print(f"  Val             : {val_count:,}")
    print(f"  Audio files     : {total_files:,}")
    print(f"  Errors          : {total_errors}")
    print(f"  Metadata CSV    : {meta_path}")
    print(f"{'─' * 60}\n")


# CLI

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic medical speech data for Whisper fine-tuning"
    )
    parser.add_argument("--out",      default="notebooks/data/medical_speech",
                        help="Output directory (default: notebooks/data/medical_speech)")
    parser.add_argument("--split",    type=float, default=0.8,
                        help="Train split fraction (default: 0.8)")
    parser.add_argument("--max",      type=int, default=None,
                        help="Max phrases per specialty for quick test run")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print file list without generating audio")
    args = parser.parse_args()

    generate_dataset(
        output_dir        = args.out,
        train_split       = args.split,
        max_per_specialty = args.max,
        dry_run           = args.dry_run,
    )
