import xarray as xr
import rasterio
import rioxarray
import gc
import earthaccess
assert(earthaccess.login(strategy="netrc").authenticated)

# Find a granule, load it into memory, take mean, call gc

gdal_config = {
    'GDAL_HTTP_COOKIEFILE': '~/cookies.txt',
    'GDAL_HTTP_COOKIEJAR': '~/cookies.txt',
    'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'TIF',
    'GDAL_HTTP_UNSAFESSL': 'YES',
    'GDAL_HTTP_MAX_RETRY': '10',
    'GDAL_HTTP_RETRY_DELAY': '0.5',
    'VSI_CACHE': 'FALSE',
    'CPL_VSIL_CURL_NON_CACHED': '/vsicurl/https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/',
    'GDAL_CACHEMAX': 0
}

if __name__ == "__main__":
    bbox_3857 = [-13571302.7073,5853627.9860,-13418428.6507,5966199.5809]
    bbox_4326 = rasterio.warp.transform_bounds(3857, 4326, *bbox_3857)

    granule = earthaccess.search_data(
        short_name="HLSL30",
        bounding_box=bbox_4326,
        temporal=("2024-06-01", "2024-09-01")
    )[0]

    with rasterio.env.Env(**gdal_config):
        data_urls = [item["URL"] for item in granule["umm"]["RelatedUrls"]]
        b5_link = next(filter(lambda x: x.endswith("B05.tif") and x.startswith("https"), data_urls))
        b5 = rioxarray.open_rasterio(b5_link, cache=False)
        b5 = b5.squeeze(drop=True)
        b5 = b5.compute()
        b5_nonan = b5.where(b5 != b5.attrs["_FillValue"])
        b5_mean = b5_nonan.mean()
        print(b5_mean)
        b5.close()
        del b5, b5_nonan, b5_mean
        
    gc.collect()
    