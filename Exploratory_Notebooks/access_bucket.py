# may need some packages
# google-cloud-storage
# gcsfs
# s3fs
import numpy as np
from google.cloud import storage
import io
def get_data_from_gcp(nrows=10000, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    storage_client = storage.Client()
    bucket = storage_client.bucket('kicks_specs')

    # blob = bucket.blob(blob_name)

    blob = bucket.blob('toms.npy')
    with io.BytesIO() as in_memory_file:
        blob.download_to_file(in_memory_file)
        in_memory_file.seek(0)
        image = np.load(in_memory_file)
    return print(image)

    # return print(data)

get_data_from_gcp()

# gs://kicks_specs/toms.npy
