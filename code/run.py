import os
import subprocess
import shutil
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# Paths
src_dir = "src"
bin_dir = "bin"
results_dir = "results"

results_file = os.path.join(results_dir, "results.csv")

# Ensure the bin and results directories exist
os.makedirs(bin_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Clear bin and results directories
for directory in [bin_dir, results_dir]:
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# 
loop = 1

# Initialize CSV file with header
header = ["ID", "Remarks", "N", "Average_GFLOPS/s", "Averange_Time/s", "GFLOPS"] + [f"GFLOPS/s_Iteration{i+1}" for i in range(loop)]
with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

# Compile and run each .cc file
id_counter = 1

N_values = [256, 512, 768, 1024, 1280, 1536, 1792, 2048,
            3072, 4096, 5120, 6144, 7168, 8192,
            10240, 12288, 16384, 20480]
# N_values = [256, 512, 768, 1024, 1280, 1536, 1792, 2048]

for src_file in os.listdir(src_dir):
    id_counter = 1
    for N in N_values:
        if src_file.endswith(".cc"):
            base_filename = os.path.splitext(src_file)[0]
            exe_file = os.path.join(bin_dir, f"{base_filename}_{N}")

            # Compile the .cc file
            if base_filename == "v0":
                if N > 2048:
                    print(f"Skipping {src_file} for N={N} as it exceeds the limit for v0.")
                    continue
                compile_command = f"g++ -o {exe_file} {os.path.join(src_dir, src_file)} -DN={N}"
            else:
                compile_command = f"g++ -fopenmp -O3 -o {exe_file} {os.path.join(src_dir, src_file)} -DN={N} -march=native" 
            compile_result = subprocess.run(compile_command, shell=True, capture_output=True, text=True)
            if compile_result.returncode != 0:
                print(f"Compilation failed for {src_file} with N={N}:\n{compile_result.stderr}")
                continue

            # Initialize results list for this file
            # results = [id_counter, f"Remarks for {base_filename}", N]
            results = [id_counter, base_filename, N]

            # Run the executable 10 times and collect results
            gflops_values = []
            gflops_per_sec_values = []
            time_values = []
            for i in range(loop):
                run_command = exe_file
                result = subprocess.run(run_command, shell=True, capture_output=True, text=True)
                output = result.stdout.strip().splitlines()
                error_output = result.stderr.strip()
                
                if result.returncode != 0:
                    print(f"Error running {base_filename} with N={N}:\n{error_output}")
                    break
                
                if len(output) != 4:
                    print(f"Unexpected output format in {base_filename} with N={N}! Output does not have 4 lines.")
                    print("Standard output:", output)
                    print("Standard error:", error_output)
                    break
                
                try:
                    n_value = int(output[0].split('=')[-1].strip())
                    gflops_per_sec = float(output[1].split('=')[-1].strip())
                    gflops = float(output[2].split('=')[-1].strip())
                    time_s = float(output[3].split('=')[-1].strip())
                    gflops_values.append(gflops)
                    gflops_per_sec_values.append(gflops_per_sec)
                    time_values.append(time_s)
                    print(f"Run {i+1} for {base_filename} with N={N} completed. Time: {time_s:.2f} GFLOPS/s: {gflops_per_sec:.2f}")
                except ValueError as ve:
                    print(f"Error parsing output in {base_filename} with N={N}: {ve}")
                    print("Output:", output)
                    break
                
                # time.sleep(1)

            if len(gflops_per_sec_values) == loop and len(gflops_values) == loop:
                # Calculate the average GFLOPS and GFLOPS/s
                average_gflops = sum(gflops_values) / len(gflops_values)
                average_gflops_per_sec = sum(gflops_per_sec_values) / len(gflops_per_sec_values)
                average_time = sum(time_values) / len(time_values)

                results += [average_gflops_per_sec, average_time, average_gflops]
                # results += gflops_values
                results += gflops_per_sec_values

                # Write the results to the CSV file
                with open(results_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(results)

        id_counter += 1
    print(f"Completed all runs for N={N}.")