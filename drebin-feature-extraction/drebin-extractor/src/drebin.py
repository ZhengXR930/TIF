import staticAnalyzer
import click
import sys
import pandas as pd
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import multiprocessing
# @click.command()
# @click.argument('apk_file')
# @click.argument('dst_dir')
# @click.option('-p', '--print', is_flag=True, default=False, help='Write analysis result to stdout')
# @click.option('-f', '--alt_format', is_flag=True, default=False, help='Output results in alternative JSON format, which requires less space.')
# def eprint(*args, **kwargs):
#     """Prints the provided arguments to stderr."""
#     print(*args, file=sys.stderr, **kwargs)

years = ['2022']
months = ['10']
info_folder = '/cs/academic/phd3/xinrzhen/xinran/invariant_training/raw_dataset/androzoo_family_with_gray'
log_file_path = f'/cs/academic/phd3/xinrzhen/xinran/drebin-feature-extraction/drebin-extractor/src/logs/feature_process_new_sample.log'
sha256_folder = '/scratch1/NOT_BACKED_UP/cavallarogrp/datasets/processed_dataset/Androzoo_raw'
feature_folder = '/scratch1/NOT_BACKED_UP/cavallarogrp/datasets/processed_dataset/Androzoo_info/drebin_22_23'
logging.basicConfig(
    filename=log_file_path,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

def process_apk(sha256, dst_dir, year, month):
# def process_apk(args):
    """
    Function to process each APK file. This will be executed in parallel.
    """
    # sha256, dst_dir, year, month = args
    start_time = time.time()
    apk_file = os.path.join(sha256_folder, sha256 + ".apk")

    try:
        write_output = True  # Set to True to write output; customize as needed
        logging.info(f"Processing {apk_file}")
        staticAnalyzer.run(apk_file, dst_dir, write_output, year, month, alt_format=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Completed processing {apk_file} in {elapsed_time:.2f} seconds")
    except Exception as e:
        logging.info(' --- '.join(['[FAILED]', apk_file, str(e)]))

def main():
    sha256 = "d4a716d53361dad59251684faa044d2da24c12b0194e02f0f572f41c8b30ede6"
    sha256 = sha256.upper()


    # dst_dir = "/scratch1/NOT_BACKED_UP/cavallarogrp/datasets/processed_dataset/Androzoo_info/drebin_new_sample"
    # if not os.path.exists(dst_dir):
    #     os.makedirs(dst_dir)
    # # df = pd.read_csv(f"{info_folder}/2022-10.txt", header=None, names=["sha256", "label", "date", "family"])
    # sha256s = []
    # with open('/cs/academic/phd3/xinrzhen/xinran/OOD_detection/dataset_analysis/additional_data/no_feature_samples_14_17.txt', 'r') as f:
    #     for line in f:
    #         sha = line.strip()
    #         sha256s.append(sha)
    # print(f"All processed sha256: {len(sha256s)}")
    # existing_files = set(os.path.splitext(file)[0] for file in os.listdir(dst_dir) if file.endswith('.json'))
    # sha256s_filted = set(sha256s) - existing_files
    # print(f"Total {len(sha256s)} APKs, {len(sha256s_filted)} APKs need to be processed.")

    # with ThreadPoolExecutor() as executor:
    #     futures = {executor.submit(process_apk, sha256, dst_dir, '2022', '10'): sha256 for sha256 in sha256s_filted}
        
    #     for future in as_completed(futures):
    #         sha256 = futures[future]
    #         try:
    #             future.result() 
    #             print(f"Completed processing {sha256}")
    #         except Exception as e:
    #             print(f"Error processing {sha256}: {str(e)}")
    #             logging.info(' --- '.join(['[FAILED]', sha256, str(e)]))
    

    # for year in years:
    #     for month in months:
    #         print(f"Processing {year}-{month}")
    #         dst_dir = os.path.join(feature_folder, f"{year}-{month}")
    #         if not os.path.exists(dst_dir):
    #             os.makedirs(dst_dir)
    #         df = pd.read_csv(f"{info_folder}/{year}-{month}.txt", header=None, names=["sha256", "label", "date", "family"])
    #         sha256s = df['sha256'].values.tolist()
    #         existing_files = set(os.path.splitext(file)[0] for file in os.listdir(dst_dir) if file.endswith('.json'))
    #         sha256s_filted = set(sha256s) - existing_files
    #         logging.info(f"Total {len(sha256s)} APKs, {len(sha256s_filted)} APKs need to be processed.")
    #         # for sha256 in sha256s_filted:
    #         #     process_apk(sha256, dst_dir, year, month)

    #         with ThreadPoolExecutor() as executor:  
    #             futures = {executor.submit(process_apk, sha256, dst_dir, year, month): sha256 for sha256 in sha256s_filted}
                
    #             for future in as_completed(futures):
    #                 sha256 = futures[future]
    #                 try:
    #                     future.result() 
    #                     print(f"Completed processing {sha256}")
    #                 except Exception as e:
    #                     print(f"Error processing {sha256}: {str(e)}")
    #                     logging.info(' --- '.join(['[FAILED]', sha256, str(e)]))

if __name__ == '__main__':
    main()