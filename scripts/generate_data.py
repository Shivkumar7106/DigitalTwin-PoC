import argparse
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation.ehr_generator import generate_ehr_data
from src.data_generation.sensor_generator import generate_sensor_data

def main():
    parser = argparse.ArgumentParser(description="Master data generation script for Digital Twin PoC.")
    parser.add_argument('--patients', type=int, default=500, help="Number of patients to generate")
    parser.add_argument('--days', type=int, default=30, help="Number of days for sensor data")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--output-dir', type=str, default='data/raw', help="Output root directory")
    
    args = parser.parse_args()
    
    ehr_dir = os.path.join(args.output_dir, 'ehr')
    sensor_dir = os.path.join(args.output_dir, 'sensors')
    os.makedirs(ehr_dir, exist_ok=True)
    os.makedirs(sensor_dir, exist_ok=True)
    
    print(f"Starting Digital Twin data generation pipeline...")
    print(f"Configuration: {args.patients} patients, {args.days} days, seed {args.seed}")
    
    # Step 1: Generate EHR data
    print("\n--- Step 1: Generating Synthetic Indian EHR Data ---")
    generate_ehr_data(num_patients=args.patients, output_dir=ehr_dir, seed=args.seed)
    
    # Step 2: Generate Sensor data
    print("\n--- Step 2: Generating Synthetic Wearable Sensor Streams (CGM, HR, Sleep, Steps) ---")
    patients_csv = os.path.join(ehr_dir, 'patients.csv')
    observations_csv = os.path.join(ehr_dir, 'observations.csv')
    
    generate_sensor_data(
        patients_csv=patients_csv,
        observations_csv=observations_csv,
        output_dir=sensor_dir,
        num_days=args.days,
        seed=args.seed
    )
    
    print(f"\n[SUCCESS] Data generation complete! All files saved to {args.output_dir}")

if __name__ == '__main__':
    main()
