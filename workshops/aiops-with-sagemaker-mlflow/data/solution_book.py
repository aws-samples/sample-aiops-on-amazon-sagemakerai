# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

treatment_protocols = {
    "chest pain, shortness of breath, sweating": {
        "condition": "Suspected Acute Coronary Syndrome",
        "triage_level": "Emergency",
        "protocol": [
            "1. Administer aspirin 325 mg immediately",
            "2. Obtain 12-lead ECG within 10 minutes",
            "3. Draw cardiac biomarkers (troponin)",
            "4. Establish IV access and administer oxygen if SpO2 < 94%",
            "5. Prepare for cardiology consultation and possible catheterization",
        ]
    },
    "persistent cough, mild fever, fatigue": {
        "condition": "Upper Respiratory Infection",
        "triage_level": "Non-urgent",
        "protocol": [
            "1. Assess vital signs and oxygen saturation",
            "2. Perform chest auscultation to rule out pneumonia",
            "3. Recommend rest, fluids, and over-the-counter antipyretics",
            "4. Prescribe cough suppressant if needed",
            "5. Schedule follow-up if symptoms persist beyond 7 days",
        ]
    },
    "severe headache, blurred vision, nausea": {
        "condition": "Suspected Hypertensive Crisis or Intracranial Event",
        "triage_level": "Emergency",
        "protocol": [
            "1. Check blood pressure immediately",
            "2. Perform neurological assessment (GCS, pupil response)",
            "3. Order urgent CT head scan",
            "4. Administer IV antihypertensive if BP critically elevated",
            "5. Consult neurology for further evaluation",
        ]
    },
    "joint pain, swelling, morning stiffness": {
        "condition": "Suspected Rheumatoid Arthritis",
        "triage_level": "Urgent",
        "protocol": [
            "1. Assess affected joints and range of motion",
            "2. Order blood tests: ESR, CRP, rheumatoid factor, anti-CCP",
            "3. Order X-rays of affected joints",
            "4. Prescribe NSAIDs for symptom relief",
            "5. Refer to rheumatology for further evaluation",
        ]
    },
    "abdominal pain, bloating, irregular bowel movements": {
        "condition": "Suspected Irritable Bowel Syndrome",
        "triage_level": "Urgent",
        "protocol": [
            "1. Obtain detailed dietary and symptom history",
            "2. Perform abdominal examination",
            "3. Order basic blood work and stool tests to rule out infection",
            "4. Recommend dietary modifications (low-FODMAP trial)",
            "5. Schedule gastroenterology referral if symptoms persist",
        ]
    },
    "dizziness, rapid heartbeat, fainting": {
        "condition": "Suspected Cardiac Arrhythmia",
        "triage_level": "Emergency",
        "protocol": [
            "1. Obtain 12-lead ECG immediately",
            "2. Continuous cardiac monitoring",
            "3. Check electrolytes and thyroid function",
            "4. Establish IV access",
            "5. Consult cardiology for possible Holter monitoring or electrophysiology study",
        ]
    },
    "sore throat, runny nose, mild headache": {
        "condition": "Common Cold / Viral Pharyngitis",
        "triage_level": "Non-urgent",
        "protocol": [
            "1. Perform throat examination and rapid strep test",
            "2. Check for fever and lymphadenopathy",
            "3. Recommend rest, fluids, and symptomatic treatment",
            "4. Prescribe analgesics for pain relief",
            "5. Advise return if symptoms worsen or fever develops",
        ]
    },
    "lower back pain, numbness in legs, difficulty walking": {
        "condition": "Suspected Lumbar Disc Herniation / Cauda Equina",
        "triage_level": "Emergency",
        "protocol": [
            "1. Perform urgent neurological examination",
            "2. Assess bladder and bowel function",
            "3. Order urgent MRI of lumbar spine",
            "4. Administer pain management (NSAIDs, muscle relaxants)",
            "5. Consult neurosurgery if cauda equina syndrome suspected",
        ]
    },
    "skin rash, itching, swelling around eyes": {
        "condition": "Suspected Allergic Reaction",
        "triage_level": "Urgent",
        "protocol": [
            "1. Assess airway and breathing for anaphylaxis signs",
            "2. Administer antihistamine (diphenhydramine or cetirizine)",
            "3. Monitor for progression of symptoms",
            "4. Identify and remove potential allergen",
            "5. Prescribe epinephrine auto-injector if risk of anaphylaxis",
        ]
    },
    "frequent urination, excessive thirst, unexplained weight loss": {
        "condition": "Suspected Diabetes Mellitus",
        "triage_level": "Urgent",
        "protocol": [
            "1. Check fasting blood glucose and HbA1c",
            "2. Perform urinalysis for glucose and ketones",
            "3. Assess for diabetic ketoacidosis if type 1 suspected",
            "4. Provide initial dietary and lifestyle counseling",
            "5. Refer to endocrinology for comprehensive management plan",
        ]
    },
    "high fever, severe body aches, chills": {
        "condition": "Suspected Influenza or Systemic Infection",
        "triage_level": "Urgent",
        "protocol": [
            "1. Obtain vital signs including temperature and oxygen saturation",
            "2. Perform rapid influenza and COVID-19 testing",
            "3. Draw blood cultures if sepsis suspected",
            "4. Administer antipyretics and IV fluids for hydration",
            "5. Initiate antiviral therapy (oseltamivir) if influenza confirmed within 48 hours",
        ]
    },
    "persistent heartburn, difficulty swallowing, chest discomfort after eating": {
        "condition": "Suspected Gastroesophageal Reflux Disease (GERD)",
        "triage_level": "Urgent",
        "protocol": [
            "1. Obtain detailed symptom history and medication review",
            "2. Perform cardiac workup to rule out cardiac cause of chest discomfort",
            "3. Prescribe proton pump inhibitor (PPI) trial for 4-8 weeks",
            "4. Recommend dietary and lifestyle modifications (avoid triggers, elevate head of bed)",
            "5. Refer for upper endoscopy if symptoms persist or alarm features present",
        ]
    },
    "sudden confusion, slurred speech, weakness on one side": {
        "condition": "Suspected Acute Stroke (CVA)",
        "triage_level": "Emergency",
        "protocol": [
            "1. Activate stroke code and note time of symptom onset",
            "2. Perform NIHSS (National Institutes of Health Stroke Scale) assessment",
            "3. Order urgent CT head without contrast to differentiate ischemic vs hemorrhagic stroke",
            "4. Check blood glucose and coagulation studies",
            "5. Administer IV alteplase (tPA) if within treatment window and no contraindications",
        ]
    },
    "chronic fatigue, pale skin, shortness of breath on exertion": {
        "condition": "Suspected Iron Deficiency Anemia",
        "triage_level": "Urgent",
        "protocol": [
            "1. Order complete blood count (CBC) with differential and reticulocyte count",
            "2. Check serum iron, ferritin, TIBC, and vitamin B12 levels",
            "3. Perform peripheral blood smear review",
            "4. Initiate oral iron supplementation if iron deficiency confirmed",
            "5. Investigate underlying cause (GI bleeding, dietary deficiency) and refer as needed",
        ]
    },
    "ankle swelling, weight gain, shortness of breath when lying down": {
        "condition": "Suspected Congestive Heart Failure",
        "triage_level": "Emergency",
        "protocol": [
            "1. Obtain chest X-ray to assess for pulmonary congestion and cardiomegaly",
            "2. Order BNP or NT-proBNP levels and echocardiogram",
            "3. Administer IV diuretics (furosemide) for fluid overload",
            "4. Monitor daily weights, fluid intake and output",
            "5. Consult cardiology for comprehensive heart failure management plan",
        ]
    },
}
