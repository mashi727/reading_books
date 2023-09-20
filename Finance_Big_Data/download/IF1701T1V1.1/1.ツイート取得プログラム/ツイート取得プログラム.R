# ライブラリの読み込み
library("twitteR")

##############################################################
# このプログラムを実行するには予め下記のサイトで
# Twitterアプリケーション登録をしてください。
# https://apps.twitter.com/

# Twitterアプリケーション登録情報の入力（①～④）
consumerKey <- "①"
consumerSecret <- "②"
accessToken <- "③"
accessSecret <- "④"

# httr_oauth_chcheを保存
options(httr_oauth_cache = TRUE)

# 認証情報の取得
setup_twitter_oauth(consumerKey, consumerSecret,
                    accessToken, accessSecret)
##############################################################

# ユーザ名一覧
usernames <- c("WSJmarkets", "FXstreetNews",
               "IBDinvestors", "ForexLive",
               "USATODAY", "washingtonpost")

# タイムラインを取得
for (user in usernames) {
  print(user)
  # タイムラインから3000件を抽出
  print(paste(">> Read TimeLine from by @", user, sep = ""))
  UserTimeLines <- userTimeline(user, n = 3000)
  print("read TimeLine")

  # ListをDataFrameに変換
  x <- twListToDF(UserTimeLines)
  print(length(x))

  # CSVファイルへ書き出し
  filname <- paste(user,".csv", sep = "")
  write.csv(x, filname, quote = TRUE, row.names = FALSE, append = FALSE)
  print("save csv file")
  
  # 連続でダウンロードして負荷をかけないように次の処理まで2分待つ
  Sys.sleep(120)
}
