import os
import numpy as np
import wfdb
from tqdm import tqdm
import json

def convert_record(record_name, input_dir='data/raw', output_dir='data/processed'):
    """
    Convert a single PhysioNet record to numpy format
    """
    try:
        # Read the record
        record = wfdb.rdrecord(os.path.join(input_dir, record_name))
        
        # Save ECG signal (first channel)
        ecg_signal = record.p_signal[:, 0]
        np.save(os.path.join(output_dir, f'{record_name}_ecg.npy'), ecg_signal)
        
        # Try to read apnea annotations
        apn_file = os.path.join(input_dir, f'{record_name}.apn')
        if os.path.exists(apn_file):
            annotation = wfdb.rdann(os.path.join(input_dir, record_name), 'apn')
            labels = annotation.symbol  # 'A' for apnea, 'N' for normal
            np.save(os.path.join(output_dir, f'{record_name}_labels.npy'), labels)
            
            # Count labels
            apnea_count = np.sum(labels == 'A')
            normal_count = np.sum(labels == 'N')
            other_labels = [l for l in labels if l not in ['A', 'N']]
            
            return {
                'record': record_name,
                'success': True,
                'ecg_length': len(ecg_signal),
                'duration_min': len(ecg_signal) / record.fs / 60,
                'labels_length': len(labels),
                'apnea_count': int(apnea_count),
                'normal_count': int(normal_count),
                'other_labels': list(set(other_labels))
            }
        else:
            # No annotation file (x01-x35 records)
            return {
                'record': record_name,
                'success': True,
                'ecg_length': len(ecg_signal),
                'duration_min': len(ecg_signal) / record.fs / 60,
                'labels_length': 0,
                'apnea_count': 0,
                'normal_count': 0,
                'other_labels': []
            }
            
    except Exception as e:
        return {
            'record': record_name,
            'success': False,
            'error': str(e)
        }

def convert_all_records():
    """
    Convert all PhysioNet records to numpy format
    """
    print("=" * 60)
    print("CONVERTING PHYSIONET DATA TO NUMPY FORMAT")
    print("=" * 60)
    
    input_dir = 'data/raw'
    output_dir = 'data/processed'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all main records (not the 'r' versions with respiration)
    dat_files = [f for f in os.listdir(input_dir) if f.endswith('.dat') and 'r' not in f]
    record_names = sorted([f.replace('.dat', '') for f in dat_files])
    
    print(f"Found {len(record_names)} records to convert")
    print("Starting conversion...\n")
    
    results = []
    successful = 0
    
    for record_name in tqdm(record_names, desc="Converting"):
        result = convert_record(record_name, input_dir, output_dir)
        results.append(result)
        
        if result['success']:
            successful += 1
            # Print summary for first few records
            if successful <= 5:
                if result['labels_length'] > 0:
                    print(f"  {record_name}: {result['duration_min']:.1f} min, "
                          f"{result['apnea_count']} apnea min")
                else:
                    print(f"  {record_name}: {result['duration_min']:.1f} min (no labels)")
    
    # Save conversion summary
    summary = {
        'total_records': len(record_names),
        'successful': successful,
        'failed': len(record_names) - successful,
        'records_with_labels': sum(1 for r in results if r.get('labels_length', 0) > 0),
        'total_apnea_minutes': sum(r.get('apnea_count', 0) for r in results),
        'total_normal_minutes': sum(r.get('normal_count', 0) for r in results)
    }
    
    with open(os.path.join(output_dir, 'conversion_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total records: {summary['total_records']}")
    print(f"Successfully converted: {summary['successful']}")
    print(f"Records with apnea labels: {summary['records_with_labels']}")
    print(f"Total apnea minutes: {summary['total_apnea_minutes']}")
    print(f"Total normal minutes: {summary['total_normal_minutes']}")
    
    # Show some failed records if any
    failed_records = [r['record'] for r in results if not r['success']]
    if failed_records:
        print(f"\nFailed records ({len(failed_records)}):")
        for record in failed_records[:10]:
            print(f"  {record}")
    
    print(f"\nFiles saved to: {output_dir}/")
    print("File naming:")
    print("  [record]_ecg.npy     - ECG signal")
    print("  [record]_labels.npy  - Apnea labels (if available)")
    
    return results

def test_conversion():
    """
    Test that we can load the converted files
    """
    print("\n" + "=" * 60)
    print("TESTING CONVERTED DATA")
    print("=" * 60)
    
    test_dir = 'data/processed'
    
    # Find first ECG file
    ecg_files = [f for f in os.listdir(test_dir) if f.endswith('_ecg.npy')]
    
    if not ecg_files:
        print("No ECG files found. Conversion may have failed.")
        return False
    
    test_file = ecg_files[0]
    record_name = test_file.replace('_ecg.npy', '')
    
    print(f"Testing record: {record_name}")
    
    try:
        # Load ECG
        ecg = np.load(os.path.join(test_dir, test_file))
        print(f"✓ ECG loaded: shape={ecg.shape}")
        print(f"  Duration: {len(ecg)/100/60:.1f} minutes (at 100 Hz)")
        
        # Try to load labels
        labels_file = os.path.join(test_dir, f'{record_name}_labels.npy')
        if os.path.exists(labels_file):
            labels = np.load(labels_file)
            print(f"✓ Labels loaded: shape={labels.shape}")
            print(f"  Apnea count: {np.sum(labels == 'A')}")
            print(f"  Normal count: {np.sum(labels == 'N')}")
            print(f"  Sample labels: {labels[:10]}")
        
        print("\n✅ Conversion test successful!")
        return True
        
    except Exception as e:
        print(f"✗ Error loading files: {e}")
        return False

if __name__ == "__main__":
    # Convert all records
    results = convert_all_records()
    
    # Test the conversion
    test_conversion()
    
    print("\n" + "=" * 60)
    print("NEXT STEP: Create exploration notebook")
    print("=" * 60)