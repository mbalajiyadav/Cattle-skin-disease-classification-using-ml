import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import numpy as np

l1 = ['anorexia', 'abdominal_pain', 'anaemia', 'abortions', 'acetone', 'aggression', 'arthrogyposis', 'ankylosis', 'anxiety', 'bellowing', 'blood_loss', 'blood_poisoning', 'blisters', 'colic', 'Condemnation_of_livers', 'conjunctivae', 'coughing', 'depression', 'discomfort', 'dyspnea', 'dysentery', 'diarrhoea', 'dehydration', 'drooling', 'dull', 'decreased_fertility', 'diffculty_breath', 'emaciation', 'encephalitis', 'fever', 'facial_paralysis', 'frothing_of_mouth', 'frothing', 'gaseous_stomach', 'highly_diarrhoea', 'high_pulse_rate', 'high_temp', 'high_proportion', 'hyperaemia', 'hydrocephalus', 'isolation_from_herd', 'infertility', 'intermittent_fever', 'jaundice', 'ketosis', 'loss_of_appetite', 'lameness', 'lack_of-coordination', 'lethargy', 'lacrimation', 'milk_flakes', 'milk_watery', 'milk_clots', 'mild_diarrhoea', 'moaning', 'mucosal_lesions', 'milk_fever', 'nausea', 'nasel_discharges', 'oedema', 'pain', 'painful_tongue', 'pneumonia', 'photo_sensitization', 'quivering_lips', 'reduction_milk_vields', 'rapid_breathing', 'rumenstasis', 'reduced_rumination', 'reduced_fertility', 'reduced_fat', 'reduces_feed_intake', 'raised_breathing', 'stomach_pain', 'salivation', 'stillbirths', 'shallow_breathing', 'swollen_pharyngeal', 'swelling', 'saliva', 'swollen_tongue', 'tachycardia', 'torticollis', 'udder_swelling', 'udder_heat', 'udder_hardeness', 'udder_redness', 'udder_pain', 'unwillingness_to_move', 'ulcers', 'vomiting', 'weight_loss', 'weakness'
]

disease = ['mastitis', 'blackleg', 'bloat', 'coccidiosis', 'cryptosporidiosis', 'displaced_abomasum', 'gut_worms', 'listeriosis', 'liver_fluke', 'necrotic_enteritis', 'peri_weaning_diarrhoea', 'rift_valley_fever', 'rumen_acidosis', 'traumatic_reticulitis', 'calf_diphtheria', 'foot_rot', 'foot_and_mouth', 'ragwort_poisoning', 'wooden_tongue', 'infectious_bovine_rhinotracheitis', 'acetonaemia', 'fatty_liver_syndrome', 'calf_pneumonia', 'schmallen_berg_virus', 'trypanosomosis', 'fog_fever'
]

l2 = []
for x in range(0, len(l1)):
    l2.append(0)
# adds data


# Load data
df_train = pd.read_csv("D:/MAJOR PROJECT/Cattle_disease_prediction_system-main/Cattle_disease_prediction_system-main/Training_base_cleaned.csv")
df_test = pd.read_csv("D:/MAJOR PROJECT/Cattle_disease_prediction_system-main/Cattle_disease_prediction_system-main/Testing base.csv")

# Preprocess data
df_train.replace({
    'prognosis': {
        'mastitis': 0, 'blackleg': 1, 'bloat': 2, 'coccidiosis': 3, 'cryptosporidiosis': 4,
        'displaced_abomasum': 5, 'gut_worms': 6, 'listeriosis': 7, 'liver_fluke': 8, 'necrotic_enteritis': 9,
        'peri_weaning_diarrhoea': 10, 'rift_valley_fever': 11, 'rumen_acidosis': 12, 'traumatic_reticulitis': 13,
        'calf_diphtheria': 14, 'foot_rot': 15, 'foot_and_mouth': 16, 'ragwort_poisoning': 17, 'wooden_tongue': 18,
        'infectious_bovine_rhinotracheitis': 19, 'acetonaemia': 20, 'fatty_liver_syndrome': 21,
        'calf_pneumonia': 22, 'schmallen_berg_virus': 23, 'trypanosomosis': 24, 'fog_fever': 25
    }}, inplace=True)

df_test.replace({
    'prognosis': {
        'mastitis': 0, 'blackleg': 1, 'bloat': 2, 'coccidiosis': 3, 'cryptosporidiosis': 4,
        'displaced_abomasum': 5, 'gut_worms': 6, 'listeriosis': 7, 'liver_fluke': 8, 'necrotic_enteritis': 9,
        'peri_weaning_diarrhoea': 10, 'rift_valley_fever': 11, 'rumen_acidosis': 12, 'traumatic_reticulitis': 13,
        'calf_diphtheria': 14, 'foot_rot': 15, 'foot_and_mouth': 16, 'ragwort_poisoning': 17, 'wooden_tongue': 18,
        'infectious_bovine_rhinotracheitis': 19, 'acetonaemia': 20, 'fatty_liver_syndrome': 21,
        'calf_pneumonia': 22, 'schmallen_berg_virus': 23, 'trypanosomosis': 24, 'fog_fever': 25
    }}, inplace=True)


X_train = df_train[l1]
y_train = df_train[["prognosis"]]
np.ravel(y_train)

X_test = df_test[l1]
y_test = df_test[["prognosis"]]
np.ravel(y_test)

# Train Decision Tree model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, np.ravel(y_train))
clf.fit(X_train, y_train)

# Streamlit App
def decision_tree_prediction(symptoms):
    l2 = [0] * len(l1)
    
    for k in range(len(l1)):
        for z in symptoms:
            if z == l1[k]:
                l2[k] = 1

    input_test = [l2]
    predict = clf.predict(input_test)
    predicted = predict[0]

    for a in range(len(disease)):
        if predicted == a:
            return disease[a]
    return "Not Found"

def main():
    st.title("Cattle Disease Prediction System")
    
    # Sidebar
    st.sidebar.header("Select your symptoms")
    symptoms = [
        st.sidebar.selectbox(f"Select Symptom {i}", [""] + sorted(l1)) for i in range(1, 6)
    ]

    result = ""  # Initialize result with an empty string

    if st.sidebar.button("Predict Disease"):
        result = decision_tree_prediction(symptoms)
    
    st.text_area("Result", result, height=70)

if __name__ == "__main__":
    main()