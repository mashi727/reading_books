using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ExgBdCtrl
{
    class EXGBD
    {

        // 波形データフォーマット
        public const int CH_SIZE = 2;
        public const int BUF_SIZE_PER_CH = 4;

        // ラベル表記
        public const string GAIN_L_LABEL = "x200";
        public const string GAIN_H_LABEL = "x1000";

        public const string FILT_L_LABEL = "30Hz";
        public const string FILT_H_LABEL = "100Hz";

        public const string FREQ_100_LABEL = "100Hz";
        public const string FREQ_200_LABEL = "200Hz";
        public const string FREQ_500_LABEL = "500Hz";
        public const string FREQ_1000_LABEL = "1000Hz";
        public const string FREQ_UNKNOWN_LABEL = "???Hz";

        // 設定値
        public enum GAIN { Low, High };
        public enum FILT { Low, High };
        public enum FREQ { F100Hz, F200Hz, F500Hz, F1000Hz };
        public enum LED { On, Off };

        // プロパティ
        public GAIN Ch1Gain { get; private set; }
        public GAIN Ch2Gain { get; private set; }
        public FILT Ch1Filt { get; private set; }
        public FILT Ch2Filt { get; private set; }
        public FREQ Freq { get; set; }
        public LED Led { get; set; }

        public string Ch1GainLabel { get { return (Ch1Gain == GAIN.High) ? GAIN_H_LABEL : GAIN_L_LABEL; } }
        public string Ch2GainLabel { get { return (Ch2Gain == GAIN.High) ? GAIN_H_LABEL : GAIN_L_LABEL; } }
        public string Ch1FiltLabel { get { return (Ch1Filt == FILT.High) ? FILT_H_LABEL : FILT_L_LABEL; } }
        public string Ch2FiltLabel { get { return (Ch2Filt == FILT.High) ? FILT_H_LABEL : FILT_L_LABEL; } }
        public string FreqLabel
        {
            get
            {
                string ret = FREQ_UNKNOWN_LABEL;
                switch (Freq)
                {
                    case FREQ.F100Hz:
                        ret = FREQ_100_LABEL;
                        break;
                    case FREQ.F200Hz:
                        ret = FREQ_200_LABEL;
                        break;
                    case FREQ.F500Hz:
                        ret = FREQ_500_LABEL;
                        break;
                    case FREQ.F1000Hz:
                        ret = FREQ_1000_LABEL;
                        break;
                    default:
                        break;
                }
                return ret;
            }
        }
        public byte[] CommandBuf { get; }

        // EXGBDコマンドフォーマット
        private const int COMM_BUF_SIZE = 4;

        private const byte GAIN_H = 0x00;
        private const byte GAIN_L = 0x01;
        private const byte FILT_H = 0x00;
        private const byte FILT_L = 0x10;
        private const byte FREQ_100HZ = 10;
        private const byte FREQ_200HZ = 20;
        private const byte FREQ_500HZ = 50;
        private const byte FREQ_1000HZ = 100;
        private const byte LED_ON = 0x01;
        private const byte LED_OFF = 0x00;

        private const int CH1_CTRL_IDX = 0;
        private const int CH2_CTRL_IDX = 1;
        private const int SMPL_FREQ_IDX = 2;
        private const int LED_ONOFF_IDX = 3;

        /// <summary>
        /// コンストラクタ
        /// </summary>
        /// <param name="ch1Gain"></param>
        /// <param name="ch2Gain"></param>
        /// <param name="ch1Filt"></param>
        /// <param name="ch2Filt"></param>
        /// <param name="freq"></param>
        /// <param name="led"></param>
        public EXGBD(GAIN ch1Gain, GAIN ch2Gain, FILT ch1Filt, FILT ch2Filt, FREQ freq, LED led)
        {
            CommandBuf = new byte[COMM_BUF_SIZE];
            buildCommBuf(ch1Gain, ch2Gain, ch1Filt, ch2Filt, freq, led);
        }

        /// <summary>
        /// コマンドデータの生成（処理部）
        /// </summary>
        /// <param name="ch1Gain"></param>
        /// <param name="ch2Gain"></param>
        /// <param name="ch1Filt"></param>
        /// <param name="ch2Filt"></param>
        /// <param name="freq"></param>
        /// <param name="led"></param>
        /// <returns>コマンドデータ</returns>
        private byte[] buildCommBuf(GAIN ch1Gain, GAIN ch2Gain, FILT ch1Filt, FILT ch2Filt, FREQ freq, LED led)
        {
            Ch1Gain = ch1Gain;
            Ch2Gain = ch2Gain;
            Ch1Filt = ch1Filt;
            Ch2Filt = ch2Filt;
            Freq = freq;
            Led = led;

            CommandBuf[CH1_CTRL_IDX] = (byte)(getGainValue(Ch1Gain) + getFiltValue(Ch1Filt));
            CommandBuf[CH2_CTRL_IDX] = (byte)(getGainValue(Ch2Gain) + getFiltValue(Ch2Filt));
            CommandBuf[SMPL_FREQ_IDX] = getFreqValue(Freq);
            CommandBuf[LED_ONOFF_IDX] = getLedValue(Led);
            return CommandBuf;
        }

        /// <summary>
        /// コマンドデータの生成（全項目指定）
        /// </summary>
        /// <param name="ch1Gain"></param>
        /// <param name="ch2Gain"></param>
        /// <param name="ch1Filt"></param>
        /// <param name="ch2Filt"></param>
        /// <param name="freq"></param>
        /// <param name="led"></param>
        /// <returns>コマンドデータ</returns>
        public byte[] BuildCommandBuf(GAIN ch1Gain, GAIN ch2Gain, FILT ch1Filt, FILT ch2Filt, FREQ freq, LED led)
        {
            return buildCommBuf(ch1Gain, ch2Gain, ch1Filt, ch2Filt, freq, led);
        }

        /// <summary>
        /// コマンドデータの生成（ゲイン指定）
        /// </summary>
        /// <param name="ch1Gain"></param>
        /// <param name="ch2Gain"></param>
        /// <returns>コマンドデータ</returns>
        public byte[] BuildCommandBuf(GAIN ch1Gain, GAIN ch2Gain)
        {
            Ch1Gain = ch1Gain;
            Ch2Gain = ch2Gain;
            CommandBuf[CH1_CTRL_IDX] = (byte)(getGainValue(Ch1Gain) + getFiltValue(Ch1Filt));
            CommandBuf[CH2_CTRL_IDX] = (byte)(getGainValue(Ch2Gain) + getFiltValue(Ch2Filt));
            return CommandBuf;
        }

        /// <summary>
        /// コマンドデータの生成（フィルタ指定）
        /// </summary>
        /// <param name="ch1Filt"></param>
        /// <param name="ch2Filt"></param>
        /// <returns>コマンドデータ</returns>
        public byte[] BuildCommandBuf(FILT ch1Filt, FILT ch2Filt)
        {
            Ch1Filt = ch1Filt;
            Ch2Filt = ch2Filt;
            CommandBuf[CH1_CTRL_IDX] = (byte)(getGainValue(Ch1Gain) + getFiltValue(Ch1Filt));
            CommandBuf[CH2_CTRL_IDX] = (byte)(getGainValue(Ch2Gain) + getFiltValue(Ch2Filt));
            return CommandBuf;
        }

        /// <summary>
        /// コマンドデータの生成（サンプリング周波数指定）
        /// </summary>
        /// <param name="freq"></param>
        /// <returns>コマンドデータ</returns>
        public byte[] BuildCommandBuf(FREQ freq)
        {
            Freq = freq;
            CommandBuf[SMPL_FREQ_IDX] = getFreqValue(Freq);
            return CommandBuf;
        }

        /// <summary>
        /// コマンドデータの生成（LED指定）
        /// </summary>
        /// <param name="led"></param>
        /// <returns>コマンドデータ</returns>
        public byte[] BuildCommandBuf(LED led)
        {
            Led = led;
            CommandBuf[LED_ONOFF_IDX] = getLedValue(Led);
            return CommandBuf;
        }

        /// <summary>
        /// ゲインのコマンド設定値の取得
        /// </summary>
        /// <param name="setting"></param>
        /// <returns>設定値</returns>
        private byte getGainValue(GAIN setting)
        {
            return (setting == GAIN.High) ? GAIN_H : GAIN_L;
        }

        /// <summary>
        /// フィルタのコマンド設定値の取得
        /// </summary>
        /// <param name="setting"></param>
        /// <returns>設定値</returns>
        private byte getFiltValue(FILT setting)
        {
            return (setting == FILT.High) ? FILT_H : FILT_L;
        }

        /// <summary>
        /// サンプリング周波数のコマンド設定値の取得
        /// </summary>
        /// <param name="setting"></param>
        /// <returns>設定値</returns>
        private byte getFreqValue(FREQ setting)
        {
            byte ret;
            switch(setting)
            {
                case FREQ.F100Hz:
                    ret = FREQ_100HZ;
                    break;
                case FREQ.F200Hz:
                    ret = FREQ_200HZ;
                    break;
                case FREQ.F500Hz:
                    ret = FREQ_500HZ;
                    break;
                case FREQ.F1000Hz:
                    ret = FREQ_1000HZ;
                    break;
                default:
                    ret = 0;
                    break;
            }
            return ret;
        }

        /// <summary>
        /// LEDのコマンド設定値の取得
        /// </summary>
        /// <param name="setting"></param>
        /// <returns>設定値</returns>
        private byte getLedValue(LED setting)
        {
            return (setting == LED.On) ? LED_ON : LED_OFF;
        }

    }
}
