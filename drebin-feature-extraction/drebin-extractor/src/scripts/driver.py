import argparse
import atexit
import glob

import requests

import staticAnalyzer
from controllerunit import *

input_dir = 'working-dir/samples'
output_dir = 'working-dir/results'


def main():
    args = parse_args()
    set_input_dir(input_dir)
    set_output_dir(output_dir)

    atexit.register(notify)

    while True:
        print("Cleaning up working dirs...")
        cleanup_dir(input_dir)
        cleanup_dir(output_dir)

        build_dir = os.path.join('working-dir', 'working-dir')

        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
            safe_mkdir(build_dir)

        print("Retrieving awaiting tasks...")
        tasks = retrieve_awaiting_batch(args.nproc, args.run_tag, 'drebin')
        shas = [str(t['sample_sha']) for t in tasks]

        if not shas:
            print("All done!")
            exit()

        print("Copying samples to VM...")
        copy_samples_batch(shas, args.nproc)

        input_files = glob.glob(input_dir + '/*')

        # for f in input_files:
        #     process_file(f)
        pool = mp.Pool(args.nproc)
        print("Processing samples...")
        pool.map(process_file, input_files)

        pool.close()
        pool.join()

        print("Copying files to NAS...")
        results = glob.glob(output_dir + '/*')
        copy_results_batch(results, args.nproc)

        print("Uploading features to DB...")
        upload_results_features(results, 'drebin', args.nproc)
        update_results_statuses(shas, 'drebin', args.run_tag)

        # exit()


def notify():
    pushover_url = 'https://api.pushover.net/1/messages.json'
    params = {
        'token': 'ad1ht22wpy718j8xrqej6gqikj4tn8',
        'user': 'uxrhz3v6tfe8an65edd8dm7bvmqzhg',
        'priority': -1,
        'message': 'Drebin has stalled! D:'
    }
    requests.post(pushover_url, params)


def process_file(fpath):
    working_dir = os.path.join('working-dir', fpath)
    safe_mkdir(working_dir)
    staticAnalyzer.run(fpath, working_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_tag', help='The experiment tag to process')
    parser.add_argument('-n', '--nproc', default=mp.cpu_count() - 1, type=int,
                        help='The number of processors to use.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
