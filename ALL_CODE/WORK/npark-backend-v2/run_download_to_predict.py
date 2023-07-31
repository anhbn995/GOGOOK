def main():
    from download_and_processing_cloud.greencover_api import run
    from main2 import main
    import os
    from config.default import NPARK_DATA_FOLDER

    start_date = 1
    end_date = None
    list_month, list_year = run.get_current_time()
    run.main_download_and_predict(list_month, list_year, start_date,
                                  end_date, CLOUD_COVER=1.0, num_band=4, use_box=True, crs='EPSG:4326')
    print('Finish download and classification', list_month, list_year)
    year = list_year[0]
    months = list_month
    for month in months:
        thang_nam = {
            "statistic_mosaic_path": f'{NPARK_DATA_FOLDER or os.getcwd()}/data_projects/Green Cover Npark Singapore/results/mosaic/T{month}-{year}.tif',
            "view_mosaic_path": f'{NPARK_DATA_FOLDER or os.getcwd()}/data_projects/Green Cover Npark Singapore/results/view/T{month}-{year}.tif',
            "result_path": f'{NPARK_DATA_FOLDER or os.getcwd()}/data_projects/Green Cover Npark Singapore/results/classification/T{month}-{year}.tif',
            "forest_path": f'{NPARK_DATA_FOLDER or os.getcwd()}/data_projects/Green Cover Npark Singapore/results/forest/T{month}-{year}.tif',
            "cloud_path": f'{NPARK_DATA_FOLDER or os.getcwd()}/data_projects/Green Cover Npark Singapore/results/cloud/T{month}-{year}.shp',
            "month": '{:02}'.format(month),
            "year": year
        }
        main(thang_nam['statistic_mosaic_path'], thang_nam['view_mosaic_path'], thang_nam['result_path'],
             thang_nam['forest_path'], thang_nam['cloud_path'], 'Sentinel',
             '{:02}'.format(month), '{:02}'.format(month - 1), year, year)
    print("OKE MEN")


if __name__ == "__main__":
    main()
