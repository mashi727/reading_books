# ライブラリの読み込み
# GAP統計量用（関数clusGap()）
library(cluster)
# 感情分析用（関数get_sentiment()）
library(syuzhet)
# グラフ化用（関数fviz_gap_stat()）
library(factoextra)

# ユーザ名を指定
usernames <- c("WSJmarkets", "FXstreetNews",
               "IBDinvestors", "ForexLive",
               "USATODAY", "washingtonpost")

# ユーザ名の数分、括弧内の処理を繰り返し
for (user in usernames) {
  # テキスト - クレンジング処理済みのデータの読み込み
  filname <- paste(user, "_sample.csv", sep = "")
  df_tw <- read.csv(filname, header = TRUE, skip = 0)
  
  # データ・フレームに変換
  df_tw <- data.frame(df_tw)

  # ヘッドラインの行を抽出
  df_tw$text <- as.vector(df_tw$text)
  
  # お気に入り数とリツイート数を抽出して合計
  interest <- df_tw$favoriteCount + df_tw$retweetCount

  # ヘッドラインを感情分析
  mySentiment <- get_sentiment(df_tw$text, method = "syuzhet")

  # データ・セット作成
  # データセットの作成とCSVファイル出力
  df <- cbind(Sentiment = mySentiment, Interest = interest)

  # 確認用にデータ・セットを書き出し
  filname <- paste(user, "_df.csv", sep = "")
  write.csv(df, filname, quote = FALSE, row.names = FALSE)
  
  # データセットからGAP統計量を計算
  gap_stat <- clusGap(df, FUN = kmeans, K.max = 10, B = 1000)

  # GAP統計量をグラフ化
  df_plot <- fviz_gap_stat(gap_stat)

  # グラフをビットマップファイルで書き出し
  filname <- paste(user, "_gap.bmp", sep = "")
  png(filename = filname)
  plot(df_plot)
  dev.off()
}
