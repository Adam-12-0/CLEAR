import re
import pandas as pd

def parse_log_file(file_path):
    data = []
    current_entry = {}
    metrics_counter = 0  # To count how many metrics have been read

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Match input file path (ends with .avi and does not start with 'Processed')
            if line.endswith('.avi') and not line.startswith('Processed'):
                # If current_entry is not empty, it means we've reached a new entry
                if current_entry:
                    data.append(current_entry)
                    current_entry = {}
                    metrics_counter = 0
                current_entry['Input File'] = line

                

            # Match processed line
            elif line.startswith('Processed'):
                # Extract input and output file paths
                match = re.match(r'Processed (.*?) -> (.*)', line)
                if match:
                    current_entry['Processed From'] = match.group(1)
                    current_entry['Processed To'] = match.group(2)
                    # Extract noise type from the input file path
                    noise_type = extract_noise_type(match.group(2))
                    current_entry['Noise Type'] = noise_type

            # Match Average PSNR
            elif line.startswith('Average PSNR:'):
                psnr = float(line.split(':')[1].strip())
                current_entry['Average PSNR'] = psnr
                metrics_counter += 1

            # Match Average SSIM
            elif line.startswith('Average SSIM:'):
                ssim = float(line.split(':')[1].strip())
                current_entry['Average SSIM'] = ssim
                metrics_counter += 1

            # Match Average VIF
            elif line.startswith('Average VIF:'):
                vif = float(line.split(':')[1].strip())
                current_entry['Average VIF'] = vif
                metrics_counter += 1

            # Match Average Processing Time per Frame
            elif line.startswith('Average Processing Time per Frame:'):
                time_str = line.split(':')[1].strip().split()[0]  # Get the time value
                processing_time = float(time_str)
                current_entry['Average Processing Time per Frame'] = processing_time
                metrics_counter += 1

            # Check if we've collected all metrics for the current entry
            if metrics_counter == 4:
                data.append(current_entry)
                current_entry = {}
                metrics_counter = 0

        # Add the last entry if it's not already added
        if current_entry:
            data.append(current_entry)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)
    return df

def extract_noise_type(file_path):
    # Use regular expression to extract noise type from the file path
    match = re.search(r'UCF-101-(.*?)/', file_path)
    if match:
        return match.group(1)
    else:
        return 'Unknown'

# Example usage:
file_path = '~/CAP5415/bi_5_20_25/nohup.out'  # Replace with your actual file path
df = parse_log_file(file_path)
df.to_csv('bi_5_20_25_processed.csv', index=False)
# Display the DataFrame
#print(df)

# Group by 'Noise Type' and calculate the mean of each metric
average_scores_by_noise = df.groupby('Noise Type').mean()

# Display the averaged scores
print(average_scores_by_noise)
