from google.cloud.storage import Client
import pandas as pd

def read_gcs_csv(client: Client, bucket: str, prefix: str) -> pd.DataFrame:
    '''
    Read CSVs hosted on GCS to a pandas dataframe.
    '''
    files = [
        "/".join(["gs://{}".format(bucket), f.name])
        for f in client.list_blobs(bucket, prefix=prefix)
        if f.name.endswith("csv")
    ]

    ds = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    return ds