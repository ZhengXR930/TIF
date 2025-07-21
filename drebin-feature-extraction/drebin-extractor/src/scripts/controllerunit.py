import errno
import multiprocessing as mp
import os
import shutil
import ujson as json

from pymongo import MongoClient

storage_root = '/media/nas/datasets/android/samples/Androzoo'

db = MongoClient(host='10.202.28.0',
                 username='cdlive',
                 password='1I#tf4Kfn717!@JJ',
                 authSource='cdlive')['s2live']

input_dir = ''
output_dir = ''


def set_input_dir(d):
    global input_dir
    input_dir = d


def set_output_dir(d):
    global output_dir
    output_dir = d


def retrieve_awaiting_batch(batch_size, run_tag, engine):
    """Retrieve a set of tasks from the analysis database.

    Args:
        batch_size (str):
        engine (str):

    Returns:

    """
    return db['results'].find(
        {'run_tag': run_tag,
         'analysis_engine': engine,
         'status': 'awaiting'}).sort(
        [('priority', -1)]).limit(batch_size)


def cleanup_dir(d):
    shutil.rmtree(d)
    safe_mkdir(d)


def copy_samples_batch(shas, nproc):
    """

    Args:
        shas (list): The shas of samples to copy over.
        target_dir (str): The working dir to copy them to.
        nproc (int): Number of processors to use.

    """

    # for sha in shas:
    #     import_sample(sha)
    pool = mp.Pool(nproc)
    pool.map_async(import_sample, shas)
    pool.close()
    pool.join()


def import_sample(sha):
    source = resolve_path(sha)
    shutil.copy2(source, input_dir)


def copy_results_batch(files_list, nproc):
    """Copy results to correct paths.

    Args:
        files_list (list): List of files where the first 64 characters are
            the sha and the remaining characters are the target filename.
        nproc (int): Number of processors to use.

    """
    pool = mp.Pool(nproc)
    pool.map_async(export_result, files_list)
    pool.close()
    pool.join()


def export_result(fname):
    basename = os.path.basename(fname)
    sha, f = basename[:64], basename[64:]
    target_dir = resolve_result_path(sha)
    safe_mkdir(target_dir)
    shutil.copy2(fname, os.path.join(target_dir, f))


def update_results_statuses(shas, engine, run_tag, status='done'):
    db['results'].update_many({'sample_sha': {'$in': shas},
                               'run_tag': run_tag,
                               'analysis_engine': engine},
                              {'$set': {'status': status}})


def upload_results_features(files_list, collection, nproc):
    pool = mp.Pool(nproc)
    result = pool.map_async(open_and_load, files_list)
    pool.close()
    pool.join()
    new_docs = result.get()
    new_docs = [x for x in new_docs if x is not None]
    db[collection].insert_many(new_docs)


def open_and_load(fname):
    try:
        return json.load(open(fname, 'rt'))
    except Exception:
        return None


def resolve_path(sha):
    prefix_dir = '/'.join([x for x in sha[:3]])
    return '{}/{}/{}.apk'.format(storage_root, prefix_dir, sha)


def resolve_result_path(sha):
    sample_path = resolve_path(sha)
    return sample_path.replace('samples', 'results')


def safe_mkdir(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def seconds_to_time(seconds):
    """Return a nicely formatted time given the number of seconds."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)
