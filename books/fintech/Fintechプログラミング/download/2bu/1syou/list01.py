import requests
import pandas as pd
from datetime import datetime, date
import zipfile

def download_gaincap(filename, path):
    tsdflg=True
    url_gc_path = "http://ratedata.gaincapital.com/"

    for i in range(len(filename)):
        # ファイルの取得
        url_file_path = filename[i]
        url_date_path="2016/12 December/"
        url =  url_gc_path + url_date_path+url_file_path
        res = requests.get(url)

        # ファイルの保存
        hd_path = path + url_file_path
        f = open(hd_path, 'wb')
        f.write(res.content)
        f.close()

        # zipファイルからの読み込み
        file_csv = "USD_JPY_Week%d.csv" %(i+1)
        print(file_csv)
        try:
            f = zipfile.ZipFile(hd_path).open(file_csv)
            #f = zipfile.ZipFile(hd_path).open()
            dfx=pd.read_csv(f, index_col=3)[['RateBid']]
            f.close()
            if tsdflg:
                tsd=dfx.copy()
                tsdflg=False
            else:
                tsd=tsd.append(dfx)
        except Exception:
            print("Error open zip file %s. The error is" %filename[i])
    return tsd

if __name__ == '__main__':
    filename0=["USD_JPY_Week1.zip",
             "USD_JPY_Week2.zip",
             "USD_JPY_Week3.zip",
             "USD_JPY_Week4.zip"]
    # 重要！！！
    # 実行の際には以下のパスを適切に修正してください
    path0="C:\\users\\moriya\\documents\\Database\\gaincap\\"
    ts=download_gaincap(filename0, path0)
    print(ts[-5:])
print("end----------------------------------")


