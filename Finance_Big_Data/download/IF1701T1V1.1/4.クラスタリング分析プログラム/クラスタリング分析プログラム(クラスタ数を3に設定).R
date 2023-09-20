# ライブラリの読み込み
# 感情分析用（関数get_sentiment()）
library(syuzhet)
# グラフ化用（関数fviz_gap_stat()）
library(factoextra)

# ユーザ名を指定
usernames <- c("WSJmarkets", "FXstreetNews", "IBDinvestors",
               "ForexLive", "USATODAY", "washingtonpost")

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
  # 感情分析結果、お気に入り数とリツイート数の合計を関心度とする
  df <- cbind(Sentiment = mySentiment, interest = interest)

  # クラスタ数3を設定
  km.res <- "3"
  
  # K平均法で計算
  km.res <- kmeans(df, km.res)
  
  # 計算結果をグラフ化
  df_plot <- fviz_cluster(km.res, data = df, frame.type = "norm") + theme_minimal()

  # グラフをビットマップファイルで書き出し
  filname <- paste(user, "_cluster.bmp", sep = "")
  png(filename = filname)
  plot(df_plot)
  dev.off()
}
