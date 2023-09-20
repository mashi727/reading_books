# サンプリングのためのライブラリ読み込み
library(dplyr)

# テキスト - クリーニング処理関数
# 関数名：cleand.data()
cleand.data = function(sentences){
  cl_text <- iconv(sentences, "latin1", "ASCII", sub="")
  cl_text <- gsub("<.*?>", "", cl_text)
  cl_text <- gsub("amp\\;", "", cl_text)
  cl_text <- gsub('(f|ht)tp\\S+\\s*',"", cl_text)
  cl_text <- gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", cl_text)
  cl_text <- gsub("@\\w+", "", cl_text)
  cl_text <- gsub("[[:punct:]]", " ", cl_text)
  cl_text <- gsub("[[:digit:]]", "", cl_text)
  cl_text <- gsub("[ \t]{2,}", " ", cl_text)
  cl_text <- gsub("\\n", " ", cl_text)
  df_tw$created <- as.POSIXct(df_tw$created, tz = "EST")
  return(cl_text)
}

# ユーザー名指定
usernames <- c("WSJmarkets", "FXstreetNews", "IBDinvestors",
               "ForexLive", "USATODAY", "washingtonpost")

# 全てのユーザー分に対して処理を繰り返す
for (user in usernames) {
  # 事前にＰＣへダウンロードしたタイムラインデータの読み込み
  filname <- paste(user,".csv", sep = "")
  df_tw <- read.csv(filname, header = TRUE, skip = 0)

  # データ期間を絞り込み、データを抽出
  grep_text1 <- grep("2016-07", df_tw$created)
  grep_text2 <- grep("2016-08", df_tw$created)
  grep_text3 <- grep("2016-09", df_tw$created)
  grep_text4 <- grep("2016-10", df_tw$created)
  grep_text5 <- grep("2016-11", df_tw$created)
  grep_text6 <- grep("2016-12", df_tw$created)
  df_list <- c(grep_text1, grep_text2, grep_text3, grep_text4, grep_text5, grep_text6)
  df_tw <- data.frame(df_tw[df_list,])

  # テキスト - クリーニング処理
  df_tw <- df_tw[complete.cases(df_tw$text),]
  
  # 関数cleand.data()を呼び出してクレンジング処理
  df_tw$text <- cleand.data(df_tw$text)
  
  # サンプリング
  df_tw <- sample_n(tbl = df_tw, size = 50)
  
  # 日時でソートし、行番号を振り直し
  sortlist <- order(df_tw$created)
  df_tw <- df_tw[sortlist,]
  rownames(df_tw) <- c(1:nrow(df_tw))

  # クレンジング・データをＣＳＶファイルに書き出し
  filname <- paste(user, "_sample.csv", sep = "")
  write.csv(df_tw, filname, quote = FALSE, row.names = FALSE)
}
