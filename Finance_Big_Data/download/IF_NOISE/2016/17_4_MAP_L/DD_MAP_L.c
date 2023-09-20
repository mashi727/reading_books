////////////////////////////////////////////////////////
//                                                    //
//  Borland C++ ドラッグ＆ドロップ信号処理プログラム  //
//  MAP推定（Lotterらの分布）                         //
//                                                    //
//                        Presented by Arata KAWAMURA //
//                                         2015.11.25.//
////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define  MEM_SIZE 16000                                     // 音声メモリのサイズ
#define  FFT_SIZE 512                                       // FFT点数
///////////////////////////////////////////////
//                                           //
//                                           //
//            FFT関数の構築                  //
//                                           //
//                                           //
///////////////////////////////////////////////
static double Wr[FFT_SIZE+1],Wi[FFT_SIZE+1];                // FFT 重み
static double Fr[FFT_SIZE+1],Fi[FFT_SIZE+1],Xr[FFT_SIZE+1],Xi[FFT_SIZE+1]; //実部と虚部
static double xin[FFT_SIZE+1],z[FFT_SIZE+1];                // FFT入力と出力
static double Xamp[FFT_SIZE+1];                             // 振幅スペクトル
static double omega;                                        // FFT角周波数
static int    St,br[FFT_SIZE+1];                            // FFTステージ番号，ビット反転

void bitr( void );                                          // ビット反転用関数の宣言
void fft ( void );                                          // FFT関数の宣言
void Wnk ( void );                                          // FFTバタフライ(BF)演算の重み関数の宣言
void ifft( void );                                          // IFFT関数の宣言
void init( void );                                          // FFT関連変数の初期設定

///////////////////////////////////////////////
//                                           //
//                初期設定                   //
//                                           //
///////////////////////////////////////////////
void init(void){
    int i;
    St=log((double)FFT_SIZE)/log(2.0)+0.5;                  // FFTアルゴリズムの段数の算出 +0.5はintへの対応
    omega=2.0*M_PI/FFT_SIZE;
    bitr();Wnk();
}

void bitr( void ){                                          // ビット反転
    int loop,i,j;
    br[0]=0;                                                // 0番目は0
    br[1]=FFT_SIZE/2;                                       // 1番目はN/2
    loop=1;
    for(j=0;j<St-1;j++){
        br[loop*2]=br[loop]/2;
        loop=loop*2;
        for(i=1;i<loop;i++){
            br[loop+i]=br[loop]+br[i];
        }
    }
}

void Wnk(void){                                            // 重みの計算
    int i;
    for(i=0;i<FFT_SIZE/2;i++){
        Wr[i]=cos(omega*i);                                // 重み実部
        Wi[i]=sin(omega*i);                                // 重み虚部
    }
}

///////////////////////////////////////////////
//                                           //
//                   FFT                     //
//                                           //
///////////////////////////////////////////////
void fft( void ){
    int _2_s,_2_s_1,roop,l,m,p;
    int s,j,k;
    double Wxmr,Wxmi;
    for(j=0;j<FFT_SIZE;j++){                                // FFT入力の設定
        Fr[br[j]]=xin[j];                                   // 入力
        Fi[br[j]]=0.0;
    }
    _2_s=1;
    for(s=1;s<=St;s++){                                     // ステージ回繰り返し
        _2_s_1=_2_s;
        _2_s=_2_s*2;
        roop=FFT_SIZE/_2_s;
        for(j=0;j<roop;j++){                                // DFT繰り返し
            for(k=0;k<_2_s_1;k++){                          // BF演算繰り返し
                l=_2_s*j+k;                                 // BFの上入力番号
                m=_2_s_1*(2*j+1)+k;                         // BFの下入力番号
                p=roop*k;                                   // 下入力への重み番号
                Wxmr=Fr[m]*Wr[p]+Fi[m]*Wi[p];               // 重み×下入力の実部
                Wxmi=Fi[m]*Wr[p]-Fr[m]*Wi[p];               // 重み×下入力の虚部
                Xr[m]=Fr[m]=Fr[l]-Wxmr;                     // BFの下出力の実部
                Xi[m]=Fi[m]=Fi[l]-Wxmi;                     // BFの下出力の虚部
                Xr[l]=Fr[l]=Fr[l]+Wxmr;                     // BFの上出力の実部
                Xi[l]=Fi[l]=Fi[l]+Wxmi;                     // BFの上出力の虚部
            }
        }
    }

    for(j=0;j<FFT_SIZE;j++){
        Xamp[j]=sqrt(Fr[j]*Fr[j]+Fi[j]*Fi[j]);              // 振幅スペクトルの算出
    }
}

///////////////////////////////////////////////
//                                           //
//                  IFFT                     //
//                                           //
///////////////////////////////////////////////
void ifft( void ){                                          // 逆フーリエ変換
    int _2_s,_2_s_1,roop,l,m,p;
    int s,j,k;
    double Wxmr,Wxmi;

    for(j=0;j<FFT_SIZE;j++){                                // 逆FFT入力の設定
        Fr[br[j]]=Xr[j];
        Fi[br[j]]=Xi[j];
    }
    _2_s=1;
    for(s=1;s<=St;s++){                                     // ステージ回繰り返し
        _2_s_1=_2_s;
        _2_s=_2_s*2;
        roop=FFT_SIZE/_2_s;
        for(j=0;j<roop;j++){                                // FFT繰り返し
            for(k=0;k<_2_s_1;k++){                          // BF演算繰り返し
                l=_2_s*j+k;                                 // BFの上入力番号
                m=_2_s_1*(2*j+1)+k;                         // BFの下入力番号
                p=roop*k;                                   // 下入力への重み番号
                Wxmr=Fr[m]*Wr[p]-Fi[m]*Wi[p];               // 重み×下入力の実部
                Wxmi=Fi[m]*Wr[p]+Fr[m]*Wi[p];               // 重み×下入力の虚部
                z[m]=Fr[m]=Fr[l]-Wxmr;                      // BFの下出力の実部
                Fi[m]=Fi[l]-Wxmi;                           // BFの下出力の虚部
                z[l]=Fr[l]=Fr[l]+Wxmr;                      // BFの上出力の実部
                Fi[l]=Fi[l]+Wxmi;                           // BFの上出力の虚部
            }
        }
    }
}


///////////////////////////////////////////////
//                                           //
//              メイン関数                   //
//                                           //
///////////////////////////////////////////////
int main(int argc, char **argv){
    ///////////////////////////////
    //                           //
    //  ファイル読み書き用変数   //
    //                           //
    ///////////////////////////////
    FILE *f1, *f2;
    unsigned short tmp1;                                    // 2バイト変数
    unsigned long  tmp2;                                    // 4バイト変数
    int Fs;                                                 // 入力サンプリング周波数
    int ch;                                                 // 入力チャネル数
    int len;                                                // 入力信号の長さ
    
    char outname[256]={0};                                  // 出力ファイル名
    unsigned short channel;                                 // 出力チャンネル数
    unsigned short BytePerSample;                           // 1サンプル当たりのバイト数
    unsigned long  file_size;                               // 出力ファイルサイズ
    unsigned long  Fs_out;                                  // 出力サンプリング周波数
    unsigned long  BytePerSec;                              // 1秒当たりのバイト数
    unsigned long  data_len;                                // 出力波形のサンプル数
    unsigned long  fmt_chnk    =16;                         // fmtチャンクのバイト数.PCMの場合は16bit
    unsigned short BitPerSample=16;                         // 1サンプルのビット数
    unsigned short fmt_ID      =1;                          // fmt ID. PCMの場合は1


//************************************************************************//

    ///////////////////////////////
    //                           //
    //      信号処理用変数       //
    //                           //
    ///////////////////////////////
    int      t      = 0;                                    // 時刻の変数
    long int t_out  = 0;                                    // 終了時刻計測用の変数
    int      add_len= 0;                                    // 出力信号を延長するサンプル数
    short    input, output;                                 // 読込み変数と書出し変数
    double   s[MEM_SIZE+1]={0};                             // 入力データ格納用変数
    double   y[MEM_SIZE+1]={0};                             // 出力データ格納用変数

    long int l,i;                                           // FFT用変数
    double   x[FFT_SIZE+1] ={0};                            // FFTの入力
    double   x1[FFT_SIZE+1]={0};                            // 半フレームシフト時の入力保持用
    double   z1[FFT_SIZE+1]={0};                            // ハーフオーバーラップの出力保持用

    double   lambda[FFT_SIZE+1]={0};                        // ノイズ推定値
    double   G[FFT_SIZE+1]     ={0};                        // スペクトルゲイン
    int      NM;                                            // ノイズを推定するフレームの数
    int      cnt = 0;                                       // フレーム番号のカウンタ

    double   gamma[FFT_SIZE+1]={0},gamma1[FFT_SIZE+1]={0};  // 事後SNR
    double   xi[FFT_SIZE+1]={0};                            // 事前SNR

    double   tmp;                                           // 途中計算用
    double   u, v, mu;                                      // ゲインに用いるパラメータ

    ///////////////////////////////
    //                           //
    //      変数の初期設定       //
    //                           //
    ///////////////////////////////

    init();                                                 // ビット反転，乗み係数の計算
    l = 0;                                                  // FFT開始時刻管理
    NM= 4;                                                  // 初期ノイズ推定のフレーム数

    v=0.126,mu=1.74;                                        // Lotterらの分布形状を決めるパラメータ


//************************************************************************//


    //////////////////////////////////////
    //                                  //
    // 実行ファイルの使い方に関する警告 //
    //                                  //
    //////////////////////////////////////
    if( argc != 2 ){                                        // 使用方法が間違った場合の警告
        fprintf( stderr, "Usage: ./a input.wav \n" );
        exit(-1);
    }
    if( (f1 = fopen(argv[1], "rb")) == NULL ){              // 出力ファイルが開けない場合の警告
        fprintf( stderr, "Cannot open %s\n", argv[1] );
        exit(-2);
    }

    ////////////////////////////////////////////
    //                                        //
    //  入力waveファイルのヘッダ情報読み込み  //
    //                                        //
    ////////////////////////////////////////////
    printf("Wave data is\n");
    fseek(f1, 22L, SEEK_SET);                               // チャネル情報位置に移動
    fread ( &tmp1, sizeof(unsigned short), 1, f1);          // チャネル情報読込 2Byte
    ch=tmp1;                                                // 入力チャネル数の記録
    fread ( &tmp2, sizeof(unsigned long), 1, f1);           // サンプリング周波数の読込 4Byte
    Fs = tmp2;                                              // サンプリング周波数の記録
    fseek(f1, 40L, SEEK_SET);                               // サンプル数情報位置に移動
    fread ( &tmp2, sizeof(unsigned long), 1, f1);           // データのサンプル数取得 4Byte
    len=tmp2/2/ch;                                          // 音声の長さの記録 (2Byteで1サンプル)

    printf("Channel       = %d ch\n",  ch);                 // 入力チャネル数の表示
    printf("Sample rate   = %d Hz \n", Fs);                 // 入力サンプリング周波数の表示
    printf("Sample number = %d\n",     len);                // 入力信号の長さの表示

    //////////////////////////////////////
    //                                  //
    //     出力waveファイルの準備       //
    //                                  //
    //////////////////////////////////////
    Fs_out        = Fs;                                     // 出力サンプリング周波数を設定
    channel       = ch;                                     // 出力チャネル数を設定
    data_len      = channel*(len+add_len)*2;                // 出力データの長さ = 全Byte数 (1サンプルで2Byte)
    file_size     = 36+data_len;                            // 全体ファイルサイズ
    BytePerSec    = Fs_out*channel*2;                       // 1秒当たりのバイト数
    BytePerSample = channel*2;                              // 1サンプル当たりのバイト数
    sscanf(argv[1],"%[^-^.]s", outname);                    // 入力ファイル名取得 (拡張子除外)
    strcat(outname,"_output.wav");                          // outname="入力ファイル名_output.wav"
    f2=fopen(outname,"wb");                                 // 出力ファイルオープン．存在しない場合は作成される

    //////////////////////////////////////
    //                                  //
    //     出力ヘッダ情報書き込み       //
    //                                  //
    //////////////////////////////////////
    fprintf(f2, "RIFF");                                    // "RIFF"
    fwrite(&file_size,    sizeof(unsigned long ), 1, f2);   // ファイルサイズ
    fprintf(f2, "WAVEfmt ");                                // "WAVEfmt"
    fwrite(&fmt_chnk,     sizeof(unsigned long ), 1, f2);   // fmt_chnk=16 (ビット数)
    fwrite(&fmt_ID,       sizeof(unsigned short), 1, f2);   // fmt ID=1 (PCM)
    fwrite(&channel,      sizeof(unsigned short), 1, f2);   // 出力チャネル数
    fwrite(&Fs_out,       sizeof(unsigned long ), 1, f2);   // 出力のサンプリング周波数
    fwrite(&BytePerSec,   sizeof(unsigned long ), 1, f2);   // 1秒当たりのバイト数
    fwrite(&BytePerSample,sizeof(unsigned short ),1, f2);   // 1サンプル当たりのバイト数
    fwrite(&BitPerSample, sizeof(unsigned short ),1, f2);   // 1サンプルのビット数(16ビット)
    fprintf(f2, "data");                                    // "data"
    fwrite(&data_len,     sizeof(unsigned long ), 1, f2);   // 出力データの長さ

    printf("\nOutput WAVE data is\n");
    printf("Channel       = %d ch\n",  channel);            // 出力チャネル数の表示
    printf("Sample rate   = %d Hz \n", Fs_out);             // 出力サンプリング周波数の表示
    printf("Sample number = %d\n",     data_len/ch/2);      // 出力信号の長さの表示

    ///////////////////////////////////
    //                               //
    //        メインループ           //
    //                               //
    ///////////////////////////////////
    fseek(f1, 44L, SEEK_SET);                               // 音声データ開始位置に移動
    while(1){                                               // メインループ
        if(fread( &input, sizeof(short), 1,f1) < 1){        // 音声を input に読込み
            if( t_out > len+add_len ) break;                // ループ終了判定
            else input=0;                                   // ループ継続かつ音声読込み終了なら input=0
        }
        s[t] = input/32768.0;                               // 音声の最大値を1とする(正規化)


//************************************************************************//

        ////////////////////////////////////////////////////
        //                                                //
        //              Signal Processing                 //
        //                                                //
        //  現在時刻tの入力 s[t] から出力 y[t] をつくる   //
        //                                                //
        //  ※ tは0からMEM_SIZE-1までをループ             //
        //                                                //
        ////////////////////////////////////////////////////

        x[l] = x1[l];                                       // FFT_SIZE/2までの入力をx[l]に格納
        x1[l]= x[FFT_SIZE/2+l]=s[t];                        // FFT_SIZE/2以降の入力をx[l]に格納
        y[t] = (z[l]+z1[l])/FFT_SIZE;                       // IFFTで得た出力
        z1[l]= z[FFT_SIZE/2+l];                             // ハーフオーバーラップ用に出力記録

        if( l>=FFT_SIZE/2){                                 // 半フレームごとにFFTを実行
            for(i=0;i<FFT_SIZE;i++){
                xin[i]=x[i]*0.5*(1.0-cos(2.0*M_PI*i/(double)FFT_SIZE));// 窓関数をかける
            }
            fft();                                          // FFT
            ///////////////////////////
            //                       //
            //       雑音推定        //
            //                       //
            ///////////////////////////
            if(cnt<NM){                                     // 初期NMフレームの平均がノイズ推定値
                for(i=0;i<FFT_SIZE;i++) lambda[i]=lambda[i]+Xamp[i]*Xamp[i]/(double)NM;
                cnt++;                                      // フレーム番号更新
            }

            for(i=0;i<FFT_SIZE;i++){
                ///////////////////////////
                //                       //
                //   事前＆事後SNR推定   //
                //                       //
                ///////////////////////////
                gamma1[i] = gamma[i];                       // 過去のγを記録
                if(lambda[i]!=0){
                    gamma[i]=Xamp[i]*Xamp[i]/lambda[i];     // 事後SNR γの更新
                    // 事前SNR ξの更新．Decision Directed method
                    xi[i]  = 0.98*gamma1[i]*G[i]*G[i] + 0.02*( fabs(gamma[i]-1.0) + (gamma[i]-1.0) )*0.5;
                }
                ////////////////////////////////////////////
                //                                        //
                //                 MAP推定                //
                //         音声振幅：Lotterらの分布       //
                //         音声位相：一様分布             //
                //         ノイズ  ：ガウス分布           //
                ////////////////////////////////////////////
                if(gamma[i] * xi[i] > 0){
                    u = 0.5-0.25*mu / sqrt(gamma[i]*xi[i]); // ゲインに用いるuの計算
                    tmp=u*u+(v-0.5)*0.5/gamma[i];          // ゲインの根号内を計算
                    if( tmp > 0){
                        G[i] = u+sqrt(u*u+(v-0.5)*0.5/gamma[i]);// Lotterらのスペクトルゲイン
                    }
                }

                G[i]=( fabs(G[i])+G[i] )/2.0;               // 負のゲインは0にする
                if(G[i]>1.0) G[i]=1.0;                      // ゲインの最大値は1
                Xr[i]= G[i]*Xr[i];                          // 実部にゲインを乗じる
                Xi[i]= G[i]*Xi[i];                          // 虚部にゲインを乗じる
            }
            ifft();                                         // IFFT
            l=0;
        }
        l++;                                                // FFT用の時刻管理

//************************************************************************//

//        y[t] = atan(y[t])/(M_PI/2.0);                       // クリップ防止
        output = y[t]*32767;                                // 出力を整数化
        fwrite(&output, sizeof(short), 1, f2);              // 結果の書き出し
        if(ch==2){                                          // ステレオ入力の場合
            if(fread(&input, sizeof(short), 1, f1)<1) break;// Rchのカラ読み込み
            fwrite(&output, sizeof(short), 1, f2);          // Rch書き込み(=Lch)
        }
        t=(t+1)%MEM_SIZE;                                   // 時刻 t の更新
        t_out++;                                            // ループ終了時刻の計測
    }
    fclose(f1);                                             // 入力ファイルを閉じる
    fclose(f2);                                             // 出力ファイルを閉じる
    return 0;                                               // メイン関数の終了
}
