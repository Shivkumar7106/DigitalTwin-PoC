import os
import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

def generate_ehr_data(num_patients=500, output_dir='data/raw/ehr', seed=42):
    """
    Generate synthetic Electronic Health Record (EHR) data for Indian patients.
    Focuses on Type 2 Diabetes and Cardiovascular risk.
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    Faker.seed(seed)
    
    fake = Faker('en_IN')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating data for {num_patients} patients...")
    
    # 1. GENERATE PATIENTS
    patients = []
    
    cities = ['Mumbai', 'Bengaluru', 'Delhi', 'Chennai', 'Hyderabad', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
    blood_types = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
    blood_probs = [0.29, 0.21, 0.40, 0.08, 0.005, 0.005, 0.005, 0.005] # Approximate Indian distribution
    
    for _ in range(num_patients):
        gender = np.random.choice(['Male', 'Female'], p=[0.52, 0.48])
        if gender == 'Male':
            first_name = fake.first_name_male()
        else:
            first_name = fake.first_name_female()
            
        last_name = fake.last_name()
        
        # Age skewed towards 40-65
        age = int(np.clip(np.random.normal(50, 12), 25, 75))
        dob = datetime.now() - timedelta(days=age * 365.25)
        
        # BMI distribution (mean ~26, std=4)
        bmi = round(np.clip(np.random.normal(26.0, 4.0), 18.0, 40.0), 1)
        
        smoking_status = np.random.choice(['Never', 'Former', 'Current'], p=[0.65, 0.20, 0.15])
        alcohol_use = np.random.choice(['None', 'Occasional', 'Regular'], p=[0.50, 0.35, 0.15])
        physical_activity = np.random.choice(['Sedentary', 'Light', 'Moderate', 'Active'], p=[0.30, 0.35, 0.25, 0.10])
        
        fam_hx_diabetes = np.random.choice([True, False], p=[0.40, 0.60])
        fam_hx_cvd = np.random.choice([True, False], p=[0.30, 0.70])
        
        # Genetic risk score correlated with family history
        base_risk = np.random.uniform(0.1, 0.5)
        if fam_hx_diabetes:
            base_risk += 0.25
        if fam_hx_cvd:
            base_risk += 0.15
        genetic_risk_score = min(round(base_risk + np.random.uniform(0, 0.1), 3), 1.0)
        
        patients.append({
            'patient_id': str(uuid.uuid4()),
            'first_name': first_name,
            'last_name': last_name,
            'gender': gender,
            'date_of_birth': dob.strftime('%Y-%m-%d'),
            'age': age,
            'city': np.random.choice(cities),
            'bmi': bmi,
            'blood_type': np.random.choice(blood_types, p=blood_probs),
            'smoking_status': smoking_status,
            'alcohol_use': alcohol_use,
            'physical_activity_level': physical_activity,
            'family_history_diabetes': fam_hx_diabetes,
            'family_history_cvd': fam_hx_cvd,
            'genetic_risk_score': genetic_risk_score
        })
        
    patients_df = pd.DataFrame(patients)
    
    # 2. GENERATE CONDITIONS
    conditions = []
    
    for idx, p in patients_df.iterrows():
        pid = p['patient_id']
        age = p['age']
        bmi = p['bmi']
        
        # Calculate risk probabilities
        t2d_prob = 0.05 + (age - 25) * 0.005 + (bmi - 23) * 0.02 + (0.2 if p['family_history_diabetes'] else 0)
        t2d_prob = np.clip(t2d_prob, 0.01, 0.9)
        
        has_t2d = np.random.random() < t2d_prob
        has_prediabetes = False
        if not has_t2d:
            pre_prob = 0.15 + (age - 25) * 0.003 + (bmi - 23) * 0.015
            has_prediabetes = np.random.random() < np.clip(pre_prob, 0.05, 0.6)
            
        htn_prob = 0.1 + (age - 30) * 0.008 + (bmi - 23) * 0.015 + (0.15 if has_t2d else 0)
        has_htn = np.random.random() < np.clip(htn_prob, 0.05, 0.8)
        
        lipid_prob = 0.15 + (bmi - 23) * 0.02 + (0.2 if has_t2d else 0)
        has_dyslipidemia = np.random.random() < np.clip(lipid_prob, 0.05, 0.8)
        
        cad_prob = 0.01 + (age - 40) * 0.003 + (0.05 if has_t2d else 0) + (0.05 if has_htn else 0) + (0.03 if p['smoking_status'] == 'Current' else 0)
        if age > 50 and np.random.random() < np.clip(cad_prob, 0.0, 0.5):
            has_cad = True
        else:
            has_cad = False
            
        has_obesity = bmi >= 25.0
        
        def add_condition(code, name, severe=False):
            onset_years_ago = np.random.uniform(0.5, min(age - 20, 20))
            onset_date = (datetime.now() - timedelta(days=onset_years_ago * 365)).strftime('%Y-%m-%d')
            severity = 'severe' if severe else np.random.choice(['mild', 'moderate'], p=[0.6, 0.4])
            conditions.append({
                'patient_id': pid,
                'condition_code': code,
                'condition_name': name,
                'onset_date': onset_date,
                'status': 'active',
                'severity': severity
            })

        if has_t2d:
            is_severe = np.random.random() < 0.25
            add_condition('44054006', 'Type 2 Diabetes Mellitus', severe=is_severe)
            patients_df.at[idx, 'has_t2d'] = True
            patients_df.at[idx, 't2d_severe'] = is_severe
        elif has_prediabetes:
            add_condition('714628002', 'Prediabetes')
            patients_df.at[idx, 'has_t2d'] = False
            patients_df.at[idx, 't2d_severe'] = False
            patients_df.at[idx, 'has_prediabetes'] = True
        else:
            patients_df.at[idx, 'has_t2d'] = False
            patients_df.at[idx, 't2d_severe'] = False
            patients_df.at[idx, 'has_prediabetes'] = False

        if has_htn:
            add_condition('38341003', 'Hypertension')
            patients_df.at[idx, 'has_htn'] = True
        else:
            patients_df.at[idx, 'has_htn'] = False
            
        if has_dyslipidemia:
            add_condition('371073000', 'Dyslipidemia')
            patients_df.at[idx, 'has_dyslipidemia'] = True
        else:
            patients_df.at[idx, 'has_dyslipidemia'] = False

        if has_cad:
            add_condition('53741008', 'Coronary Artery Disease')
            
        if has_obesity:
            add_condition('414916001', 'Obesity')
            
    conditions_df = pd.DataFrame(conditions)
    
    # 3. GENERATE OBSERVATIONS
    observations = []
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2 * 365) # Past 2 years
    
    for idx, p in patients_df.iterrows():
        pid = p['patient_id']
        has_t2d = p.get('has_t2d', False)
        t2d_severe = p.get('t2d_severe', False)
        has_pre = p.get('has_prediabetes', False)
        has_htn = p.get('has_htn', False)
        has_dys = p.get('has_dyslipidemia', False)
        
        # Base values for this patient
        if has_t2d:
            base_hba1c = np.random.uniform(9.0, 12.0) if t2d_severe else np.random.uniform(6.6, 8.9)
            base_fbg = np.random.uniform(200, 300) if t2d_severe else np.random.uniform(126, 199)
        elif has_pre:
            base_hba1c = np.random.uniform(5.7, 6.4)
            base_fbg = np.random.uniform(100, 125)
        else:
            base_hba1c = np.random.uniform(4.5, 5.6)
            base_fbg = np.random.uniform(70, 99)
            
        if has_htn:
            base_sbp = np.random.uniform(130, 180)
            base_dbp = np.random.uniform(85, 110)
        else:
            base_sbp = np.random.uniform(100, 125)
            base_dbp = np.random.uniform(65, 80)
            
        if has_dys:
            base_tc = np.random.uniform(200, 300)
            base_ldl = np.random.uniform(130, 220)
            base_hdl = np.random.uniform(25, 45)
            base_tg = np.random.uniform(150, 400)
        else:
            base_tc = np.random.uniform(140, 199)
            base_ldl = np.random.uniform(60, 129)
            base_hdl = np.random.uniform(40, 70)
            base_tg = np.random.uniform(70, 149)
            
        base_creat = np.random.uniform(0.7, 1.2)
        if has_t2d and t2d_severe:
            base_creat += np.random.uniform(0.2, 1.0)
            
        base_egfr = np.random.uniform(90, 120)
        if base_creat > 1.2:
            base_egfr = np.random.uniform(40, 89)
            
        # Generate 8 quarterly observations
        for i in range(8):
            obs_date = start_date + timedelta(days=i*90 + np.random.randint(-15, 15))
            str_date = obs_date.strftime('%Y-%m-%d')
            
            # Add some noise to base values for each reading
            def add_obs(loinc, name, val, unit, r_low, r_high):
                observations.append({
                    'patient_id': pid,
                    'observation_date': str_date,
                    'loinc_code': loinc,
                    'observation_name': name,
                    'value': round(val, 2),
                    'unit': unit,
                    'reference_range_low': r_low,
                    'reference_range_high': r_high
                })
                
            add_obs('4548-4', 'HbA1c', max(4.0, base_hba1c + np.random.normal(0, 0.2)), '%', '4.0', '5.6')
            add_obs('1558-6', 'Fasting Blood Glucose', max(60, base_fbg + np.random.normal(0, 5)), 'mg/dL', '70', '99')
            add_obs('8480-6', 'Systolic Blood Pressure', max(80, base_sbp + np.random.normal(0, 4)), 'mmHg', '90', '120')
            add_obs('8462-4', 'Diastolic Blood Pressure', max(50, base_dbp + np.random.normal(0, 3)), 'mmHg', '60', '80')
            add_obs('2093-3', 'Total Cholesterol', max(100, base_tc + np.random.normal(0, 10)), 'mg/dL', '125', '200')
            add_obs('18262-6', 'LDL Cholesterol', max(30, base_ldl + np.random.normal(0, 8)), 'mg/dL', '0', '100')
            add_obs('2085-9', 'HDL Cholesterol', max(15, base_hdl + np.random.normal(0, 3)), 'mg/dL', '40', '60')
            add_obs('2571-8', 'Triglycerides', max(40, base_tg + np.random.normal(0, 15)), 'mg/dL', '0', '150')
            add_obs('2160-0', 'Creatinine', max(0.4, base_creat + np.random.normal(0, 0.05)), 'mg/dL', '0.6', '1.2')
            add_obs('48642-3', 'eGFR', max(10, base_egfr + np.random.normal(0, 2)), 'mL/min/1.73m2', '90', '120')
            add_obs('39156-5', 'BMI', p['bmi'] + np.random.normal(0, 0.5), 'kg/m2', '18.5', '22.9')
            
    observations_df = pd.DataFrame(observations)
    
    # 4. GENERATE MEDICATIONS
    medications = []
    
    for idx, p in patients_df.iterrows():
        pid = p['patient_id']
        has_t2d = p.get('has_t2d', False)
        t2d_severe = p.get('t2d_severe', False)
        has_htn = p.get('has_htn', False)
        has_dys = p.get('has_dyslipidemia', False)
        
        start_dt_str = (datetime.now() - timedelta(days=np.random.randint(100, 700))).strftime('%Y-%m-%d')
        
        def add_med(name, rxnorm, dose, freq):
            medications.append({
                'patient_id': pid,
                'medication_name': name,
                'rxnorm_code': rxnorm,
                'dosage': dose,
                'frequency': freq,
                'start_date': start_dt_str,
                'status': 'active'
            })
            
        if has_t2d:
            add_med('Metformin', '860975', '500 mg', 'Twice daily')
            if t2d_severe:
                med_choice = np.random.choice(['Glimepiride', 'Sitagliptin', 'Insulin Glargine'])
                if med_choice == 'Glimepiride':
                    add_med('Glimepiride', '25255', '2 mg', 'Once daily')
                elif med_choice == 'Sitagliptin':
                    add_med('Sitagliptin', '312961', '100 mg', 'Once daily')
                else:
                    add_med('Insulin Glargine', '274783', '15 units', 'Once daily at bedtime')
                    
        if has_htn:
            med_choice = np.random.choice(['Amlodipine', 'Telmisartan', 'Enalapril'])
            if med_choice == 'Amlodipine':
                add_med('Amlodipine', '17238', '5 mg', 'Once daily')
            elif med_choice == 'Telmisartan':
                add_med('Telmisartan', '141883', '40 mg', 'Once daily')
            else:
                add_med('Enalapril', '3827', '5 mg', 'Once daily')
                
        if has_dys:
            med_choice = np.random.choice(['Atorvastatin', 'Rosuvastatin'])
            if med_choice == 'Atorvastatin':
                add_med('Atorvastatin', '83367', '20 mg', 'Once daily at bedtime')
            else:
                add_med('Rosuvastatin', '301542', '10 mg', 'Once daily at bedtime')
                
    medications_df = pd.DataFrame(medications)
    
    # Cleanup helper columns from patients dataframe
    cols_to_drop = [c for c in ['age', 'has_t2d', 't2d_severe', 'has_prediabetes', 'has_htn', 'has_dyslipidemia'] if c in patients_df.columns]
    patients_df = patients_df.drop(columns=cols_to_drop)
    
    # Save to CSV
    patients_df.to_csv(os.path.join(output_dir, 'patients.csv'), index=False)
    conditions_df.to_csv(os.path.join(output_dir, 'conditions.csv'), index=False)
    observations_df.to_csv(os.path.join(output_dir, 'observations.csv'), index=False)
    medications_df.to_csv(os.path.join(output_dir, 'medications.csv'), index=False)
    
    # Print summaries
    print("\nGeneration Complete!")
    print(f"Patients: {len(patients_df)}")
    print(f"Conditions: {len(conditions_df)}")
    print(f"Observations: {len(observations_df)}")
    print(f"Medications: {len(medications_df)}")
    
    print("\nSummary Statistics:")
    print("-------------------")
    print("Gender Distribution:\n", patients_df['gender'].value_counts(normalize=True).round(3))
    
    if len(conditions_df) > 0:
        print("\nTop Conditions Prevalence:")
        cond_counts = conditions_df['condition_name'].value_counts()
        for cond, count in cond_counts.items():
            print(f"- {cond}: {count} ({(count/num_patients)*100:.1f}%)")
            
    return {
        'patients': patients_df,
        'conditions': conditions_df,
        'observations': observations_df,
        'medications': medications_df
    }

if __name__ == '__main__':
    # Default to data/raw/ehr in the current working directory, which should be the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_directory = os.path.join(project_root, 'data', 'raw', 'ehr')
    generate_ehr_data(num_patients=500, output_dir=output_directory)
