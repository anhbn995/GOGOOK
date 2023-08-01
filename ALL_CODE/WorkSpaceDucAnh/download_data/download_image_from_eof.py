from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import os, re
from tqdm import tqdm

def get_name_file_in_URL(text):
    pattern = r"^https://aws\.eofactory\.ai/.*?/download\?name=.*?&type=tif$"
    match = re.search(pattern, text)
    if match:
        struct_name = '=.*?&type=tif'
        return re.search(struct_name, text).group(0)[1:].replace('&type=','.')
    else:
        return None


link_driver_chrome = r"E:\WorkSpaceDucAnh\download_data\chromedriver_win32\chromedriver.exe"

dir_download=r"F:\Jodhpur"
list_link_download = [
"https://aws.eofactory.ai/1179/c66dac07cf9e406f8a6589e76a329824/imageries/8e61ed4902e54cd0ab4b87f4a1a797de/download?name=S2B_MSIL1C_20221128T054159_N0400_R005_T43RCJ_20221128T072913&type=tif",
"https://aws.eofactory.ai/1179/c66dac07cf9e406f8a6589e76a329824/imageries/d893e2e7bc6045e297d7ca393795bb1e/download?name=S2B_MSIL1C_20221128T054159_N0400_R005_T43RCK_20221128T072913&type=tif",
"https://aws.eofactory.ai/1179/c66dac07cf9e406f8a6589e76a329824/imageries/b2dd85d05d764dc6b4d6a24ef73ac83b/download?name=S2B_MSIL1C_20221128T054159_N0400_R005_T43RBK_20221128T072913&type=tif"
]

# xem qua đường dẫn ảnh sẽ được download xuống.
# list_fp_download = [os.path.join(dir_download, get_name_file_in_URL(url)) for url in list_link_download]
# print(list_fp_download)

os.makedirs(dir_download, exist_ok=True)
check_done = 0
downloaded = []

# kiểm tra download rồi thì thôi
for url in list_link_download:
    fp = os.path.join(dir_download, get_name_file_in_URL(url))
    if os.path.isfile(fp):
        check_done += 1
        downloaded.append(url)
        
if check_done == len(list_link_download):
    print("Đã download hết rồi !!!")

# download những file chưa download 
else:
    list_link_download = list(set(list_link_download) - set(downloaded))
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--start-maximized")
    download_dir = os.path.join(os.path.expanduser('~'), 'Downloads', dir_download)
    prefs = {'download.default_directory' : download_dir}
    chrome_options.add_experimental_option('prefs', prefs)
    service = Service(link_driver_chrome)
    service.start()

    driver = webdriver.Remote(service.service_url, chrome_options.to_capabilities())
    for link_download in list_link_download:
        link_download = link_download.replace('https://aws', 'https://api2')
        driver.get(link_download);  
        print(link_download)
    
    check_done = 0
    list_tmp = []
    with tqdm(total = len(list_link_download)) as progress_bar:
        while check_done < len(list_link_download):
            list_link_download_ok = list(set(list_link_download) - set(list_tmp))
            for url in list_link_download_ok:
                fp = os.path.join(dir_download, get_name_file_in_URL(url))
                if os.path.isfile(fp):
                    list_tmp.append(fp)
                    check_done += 1
                    progress_bar.update(1)
    # if check_done == len(list_link_download):
    #     driver.quit()