# ユーザ名の設定
user <- "washingtonpost"

# ラベル番号の設定
label_num <- 29

# ラベル番号から日時を検索するため
# 中間ファイルの読み出し
filename <- paste(user, "_sample.csv", sep = "")
df_sample <- read.csv(filename, header = TRUE, skip = 0)
df_sample <- data.frame(df_sample)

# 日時を検索してオリジナルのツイートを表示するため，
# ダウンロードファイルの読み出し
filename <- paste(user,".csv", sep = "")
df_tw <- read.csv(filename, header = TRUE, skip = 0)

# 中間ファイルの日時をキーにダウンロードファイルのレコードを検索
num_row <- grep(df_sample$created[label_num],df_tw$created)

# 感情値と関心度の数値を表示するため、
# グラフ作成に使用した中間ファイルを読み出し
filename <- paste(user, "_df.csv", sep = "")
df_sentiment <- read.csv(filename, header = TRUE, skip = 0)

# オリジナルのツイートを表示
print(as.character(df_tw$text[num_row]))

# 感情値と関心度の数値を表示
print(df_sentiment[label_num,])