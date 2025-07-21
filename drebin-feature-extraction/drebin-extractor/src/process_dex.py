import os
import zipfile
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

apk_info_folder = "/cs/academic/phd3/xinrzhen/xinran/invariant_training/raw_dataset/androzoo_family_with_gray"

def apk2dext(sampleFile, dexDir):
    try:
        apkfile = zipfile.ZipFile(sampleFile, 'r')
        apkname = str(apkfile.filename).split('/')[-1][:-4]

        if not os.path.exists(dexDir):
            os.makedirs(dexDir, exist_ok=True)

        if not os.path.isdir('dex'):
            os.mkdir('dex')
        for tempfile in apkfile.namelist():
            if tempfile.endswith('.dex'):
                dexfilename = apkname + '.dex'
                print('dexfile:', dexfilename)
                with open(os.path.join(dexDir, dexfilename), 'wb+') as f:
                    f.write(apkfile.read(tempfile))
    except zipfile.BadZipFile:
        print(f"Bad zip file: {sampleFile}")
    except Exception as e:
        print(f"Error processing {sampleFile}: {e}")

def exists_dex(dexDir):
    exist_dex_list = []
    for root, dirs, files in os.walk(dexDir):
        for file in files:
            sha256 = file.split('.')[0]
            exist_dex_list.append(sha256)
    print(f"exist file list length: {len(exist_dex_list)}")
    return exist_dex_list

def process_month(year, m):
    print(f"Processing {year}-{m}")
    dexDir = f'/scratch1/NOT_BACKED_UP/cavallarogrp/datasets/feature_space/raw_dex/{year}-{m}'
    os.makedirs(dexDir, exist_ok=True)
    apk_info_path = f"{apk_info_folder}/{year}-{m}.txt"
    df = pd.read_csv(apk_info_path, header=None, names=["sha256", "label", "date", "family"])
    sha256s = df['sha256'].values.tolist()
    exists_dex_list = exists_dex(dexDir)
    sha256s_new = list(set(sha256s) - set(exists_dex_list))

    print(f"new dex file list length: {len(sha256s_new)}, original dex file list length: {len(sha256s)}, exist dex file list length: {len(exists_dex_list)}")

    with ThreadPoolExecutor() as executor:
        futures = []
        for sha256 in sha256s_new:
            sampleFile = f"/scratch1/NOT_BACKED_UP/cavallarogrp/datasets/processed_dataset/Androzoo_raw/{sha256}.apk"
            if not os.path.exists(sampleFile):
                print(f"{sampleFile} does not exist")
            else:
                futures.append(executor.submit(apk2dext, sampleFile, dexDir))
        
        # Wait for all futures to complete
        for future in tqdm(futures, desc=f"Finalizing APKs for {year}-{m}"):
            future.result()
            
    print(f"Finished processing {year}-{m}")


if __name__ == '__main__':
    years = ["2022", "2023"]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    for year in years:
        for month in months:
            process_month(year, month)
