# ライブラリの読み込み
# GAP統計量用（関数clusGap()）
library(cluster)
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
  
  # 機械的にクラスター数を決めるため、GAP統計量計算
  gap_stat <- clusGap(df, FUN = kmeans, K.max = 10, B = 1000)
  
  # 最大のGAP統計量を抽出
  km.res <- grep(max(gap_stat$Tab[,3]), gap_stat$Tab[,3])
  
  # "gap"列の要素数を取得
  n <- length(gap_stat$Tab[,3])
  
  # 二番目に大きいGAP統計量からクラスタ数を抽出
  sec <- sort(gap_stat$Tab[,3])[n-1]
  sec <- grep(sec, gap_stat$Tab[,3])

  # 三番目に大きいGAP統計量からクラスタ数を抽出
  thi <- sort(gap_stat$Tab[,3])[n-2]
  thi <- grep(thi, gap_stat$Tab[,3])

  # 最大のGAP統計量時にクラスタ数が"1"または"10"
  if(km.res == 1 || km.res == 10){
    
    # 二番目に大きいGAP統計量時にクラスタ数が"1"または"10"
    if(sec == 1 || sec == 10){
      
      # 三番目に大きいGAP統計量からクラスタ数を代入
      km.res <- thi
    }
    
    # 二番目に大きいGAP統計量時にクラスタ数が"2"または"9"
    else{
      
      # 二番目に大きいGAP統計量からクラスタ数を代入
      km.res <- sec
    }
  }

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
proc.time()-t