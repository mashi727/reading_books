using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Windows.Devices.Bluetooth;
using Windows.Devices.Bluetooth.GenericAttributeProfile;
using Windows.Devices.Enumeration;
using System.Runtime.InteropServices.WindowsRuntime;
using System.IO;
using System.IO.Pipes;
using System.Threading;
using System.Diagnostics;


namespace ExgBdCtrl
{
    class Program
    {
        #region 定数
        // BLE 通信
        private const string BLE_DEVICE_SERVICE_UUID = "2a540f59-6469-4397-9f31-67d0bcfaccd8";
        private const string BLECHAR_WAVE_VAL_UUID = "77adbd38-b1aa-43e1-b570-9fff72223d29";
        private const string BLECHAR_BD_CTRL_UUID = "370d54c9-a18d-438b-a446-355d1bcab107";

        // ファイル保存
        private const int INPUT_CH_NUM = 2;                 // 入力チャネル数
#if DEBUG
        private const int SAVE_DEFAULT_FILE_SIZE = 10000;   // 保存用データバッファのサイズ
#else
        private const int SAVE_DEFAULT_FILE_SIZE = 1000000;	// 保存用データバッファのサイズ
#endif
        private const int SAVE_BUF_SLOT_NUM = 2;            // 保存用データバッファのスロット数
        private const string FILE_DEFAULT_PREFIX = "rec_";  // 保存ファイル名の接頭辞（初期値）
        private const string FILE_ASCII_EXT = ".txt";       // 保存ファイル名の拡張子（テキストファイル）
        private const string FILE_BINARY_EXT = ".dat";      // 保存ファイル名の拡張子（バイナリファイル）
        private const string ASCII_SEPARATOR = "\t";        // ASCII形式で保存時のチャネル間の分割文字
        private const string ATTR_FILE_SUFFIX = "_attr";    // 属性ファイル名の接尾辞
        private const string ATTR_START_TITLE = "Start time:";      // 属性ファイルの保存開始時間タイトル
        private const string ATTR_END_TITLE = "End time:";          // 属性ファイルの保存終了時間タイトル
        private const string ATTR_GAIN1_TITLE = "Gain ch1:";        // 属性ファイルのハードウェアゲインCh1タイトル
        private const string ATTR_GAIN2_TITLE = "Gain ch2:";        // 属性ファイルのハードウェアゲインCh2タイトル
        private const string ATTR_FILT1_TITLE = "Filter ch1:";      // 属性ファイルのハードウェアフィルCh1タタイトル
        private const string ATTR_FILT2_TITLE = "Filter ch2:";      // 属性ファイルのハードウェアフィルCh2タタイトル
        private const string ATTR_FREQ_TITLE = "Sampling rate:";    // 属性ファイルのサンプリング周波数タイトル

        // コマンドライン引数
        private const string HELP_OPTION = "/?";
        private const string USAGE_DESCRIPTION = "\n[Description]\n" +
            "  Console application for EXG BD control via BLE.\n\n" +
            "[Features]\n- Control EXG BD (Gain, Filter, Sampling rate, LED on/off)\n" +
            "- Acquiring waveform\n" +
            "- Save waveform data to file\n" +
            "- Send waveform data to named pipe\n\n" +
            "[Usage]\n1. ExgBdCtrl\n" +
            "    Console mode. Control by command on console.\n\n" +
            "2. ExgBdCtrl <pipe_name>\n" +
            "    Interprocess communication mode.\n" +
            "    Waveforms are sent to the parent process via named pipe\n" +
            "      Ex) ExgBdCtrl wav_pipe\n\n" +
            "3. ExgBdCtrl /?\n" +
            "    Show description.\n\n" +
            "  For details, see the support page of the January 2017 issue\n" +
            "  of the Interface magazine by CQ Publishing Co.,Ltd.";
        #endregion

        #region 変数
        // EXG BD オブジェクト
        private static EXGBD exgbd;

        // BLE 通信
        private static GattCharacteristic characteristic_notify;
        private static GattCharacteristic characteristic_command;

        // ファイル保存
        private static int saveBufSlotIdx;                  // バッファスロットインデックス
        private static int saveDataCounter;                 // 取得データ数カウンタ
        private static int saveDataLength;                  // 保存データ長
        private static int[][] saveWaveBuf;                 // 保存用波形データバッファ
        private static Task flushFileTask;                  // ファイル書き出し用ワーカースレッド
        private static string saveFilePrefix;               // 保存ファイル名の接頭辞
        private static CmdInterface.FILE_FORMAT saveFileFormat; // 保存ファイルフォーマット
        private static bool saveFileFlag;                   // ファイル保存中フラグ
        private static string startSaveTime;                // ファイル保存の開始時刻

        // 波形送信用名前付きパイプ
        private static NamedPipeServerStream server;

        // アプリケーション終了フラグ
        private static bool loopFlag;
        #endregion

        /// <summary>
        /// メインエントリ
        /// </summary>
        /// <param name="args">名前付きパイプ名または"/?"</param>
        static void Main(string[] args)
        {
            if (args.Length > 0)
            {
                string firstArg = Convert.ToString(args[0]);
                if (firstArg == HELP_OPTION)
                {
                    Console.WriteLine(USAGE_DESCRIPTION);
                    return;
                }
                else
                {
                    setupNamedPipe(firstArg);
                }
            }

            initialize();

            connectBLE();

            mainLoop();

            finalize();
        }

        /// <summary>
        /// 名前付きパイプの設定
        /// </summary>
        /// <param name="pipeName">名前付きパイプ名</param>
        private static void setupNamedPipe(string pipeName)
        {
            try
            {
                char[] invalidChars = System.IO.Path.GetInvalidFileNameChars();

                if (pipeName.IndexOfAny(invalidChars) < 0)
                {
                        server = new NamedPipeServerStream(pipeName);
                }
                else
                {
                    new Exception("Including invalid character(s).");
                }
                server.WaitForConnection();
            }
            catch(Exception ex)
            {
                Debug.WriteLine("Error: Invalid pipe name.\n" + ex.Message);   
            }
        }

        /// <summary>
        /// 初期化処理
        /// </summary>
        private static void initialize()
        {
            exgbd = new EXGBD(EXGBD.GAIN.High, EXGBD.GAIN.High, EXGBD.FILT.Low, EXGBD.FILT.Low, EXGBD.FREQ.F200Hz, EXGBD.LED.Off);

            saveBufSlotIdx = 0;
            saveDataCounter = 0;
            saveDataLength = SAVE_DEFAULT_FILE_SIZE;

            saveWaveBuf = new int[SAVE_BUF_SLOT_NUM][];
            for (int i = 0; i < SAVE_BUF_SLOT_NUM; i++)
            {
                saveWaveBuf[i] = new int[saveDataLength * INPUT_CH_NUM];
            }

            saveFilePrefix = FILE_DEFAULT_PREFIX;
            saveFileFormat = CmdInterface.FILE_FORMAT.Ascii;
            saveFileFlag = false;
            startSaveTime = "";

            loopFlag = true;
        }

        /// <summary>
        /// 終了処理
        /// </summary>
        private static void finalize()
        {
            if (saveFileFlag && (saveDataCounter > 0))
            {
                flushFileTask = Task.Factory.StartNew(() => flashFile(saveBufSlotIdx, saveDataCounter));
                flushFileTask.Wait();
            }
            if (server != null)
            {
                if (server.IsConnected)
                {
                    server.Disconnect();
                }
                server.Close();
            }
        }

        /// <summary>
        /// コマンド受信用メインループ
        /// </summary>
        private static void mainLoop()
        {
            while (loopFlag)
            {
                string cmd_str = Console.ReadLine();
#if DEBUG
                using (StreamWriter sw = new StreamWriter(@"C:\Users\TT\Documents\Python Scripts\temp.txt"))
                {
                    sw.WriteLine("Receive char: \"" + cmd_str + "\"");
                    sw.Flush();
                }
#endif
                CmdInterface ci = new CmdInterface(cmd_str);

                switch (ci.CmdType)
                {
                    case CmdInterface.CMD_TYPE.Set:
                        procSetCommand(ci);
                        break;
                    case CmdInterface.CMD_TYPE.Req:
                        procReqCommand(ci);
                        break;
                    case CmdInterface.CMD_TYPE.File:
                        procFileCommand(ci);
                        break;
                    default:
                        break;
                }
                ci = null;
            }
        }

        /// <summary>
        /// 設定コマンドの処理
        /// </summary>
        /// <param name="ci">コマンドオブジェクト</param>
        private static void procSetCommand(CmdInterface ci)
        {
            if (ci.CmdArg == null)
            {
                return;
            }

            switch (ci.CmdCode)
            {
                case CmdInterface.CMD_CODE.Acq:
                    // Not supported
                    break;
                case CmdInterface.CMD_CODE.Gain:
                    EXGBD.GAIN[] gains = (EXGBD.GAIN[])ci.CmdArg;
                    sendCommand(exgbd.BuildCommandBuf(gains[0], gains[1]));
                    break;
                case CmdInterface.CMD_CODE.Filt:
                    EXGBD.FILT[] filts = (EXGBD.FILT[])ci.CmdArg;
                    sendCommand(exgbd.BuildCommandBuf(filts[0], filts[1]));
                    break;
                case CmdInterface.CMD_CODE.Freq:
                    sendCommand(exgbd.BuildCommandBuf((EXGBD.FREQ)ci.CmdArg));
                    break;
                case CmdInterface.CMD_CODE.Led:
                    sendCommand(exgbd.BuildCommandBuf((EXGBD.LED)ci.CmdArg));
                    break;
                case CmdInterface.CMD_CODE.Save:
                    saveFileFlag = (bool)ci.CmdArg;
                    break;
                default:
                    break;
            }
        }

        /// <summary>
        /// リクエストコマンドの処理
        /// </summary>
        /// <param name="ci">コマンドオブジェクト</param>
        private static void procReqCommand(CmdInterface ci)
        {
            switch (ci.CmdCode)
            {
                case CmdInterface.CMD_CODE.DevName:
                    Console.WriteLine(CmdInterface.CMD_NOT_SUPPORTED);
                    break;
                case CmdInterface.CMD_CODE.ConnStatus:
                    string rep_str = (server.IsConnected) ? CmdInterface.CONN_STATUS_CONNECTED : CmdInterface.CONN_STATUS_UNCONNECTED;
                    Console.WriteLine(rep_str);
                    break;
                case CmdInterface.CMD_CODE.Quit:
                    Console.WriteLine(CmdInterface.CMD_QUIT_ACQ);
                    loopFlag = false;
                    break;
                case CmdInterface.CMD_CODE.Acq:
                    Console.WriteLine(CmdInterface.GetStrForReqAcqCmd(true));
                    break;
                case CmdInterface.CMD_CODE.Gain:
                    EXGBD.GAIN[] gains = { exgbd.Ch1Gain, exgbd.Ch2Gain };
                    Console.WriteLine(CmdInterface.GetStrForReqGainCmd(gains));
                    break;
                case CmdInterface.CMD_CODE.Filt:
                    EXGBD.FILT[] filts = { exgbd.Ch1Filt, exgbd.Ch2Filt };
                    Console.WriteLine(CmdInterface.GetStrForReqFiltCmd(filts));
                    break;
                case CmdInterface.CMD_CODE.Freq:
                    Console.WriteLine(CmdInterface.GetStrForReqFreqCmd(exgbd.Freq));
                    break;
                case CmdInterface.CMD_CODE.Led:
                    Console.WriteLine(CmdInterface.GetStrForReqLedCmd(exgbd.Led));
                    break;
                case CmdInterface.CMD_CODE.Save:
                    Console.WriteLine(CmdInterface.GetStrForReqSaveCmd(saveFileFlag));
                    break;
                default:
                    break;
            }
        }

        /// <summary>
        /// 保存ファイル設定コマンドの処理
        /// </summary>
        /// <param name="ci">コマンドオブジェクト</param>
        private static void procFileCommand(CmdInterface ci)
        {
            // Size
            if ((!saveFileFlag) && (ci.FileDataLength > 0))
            {
                if (flushFileTask.Status != TaskStatus.Running)
                {
                    saveDataLength = ci.FileDataLength;
                    for (int i = 0; i < SAVE_BUF_SLOT_NUM; i++)
                    {
                        saveWaveBuf[i] = new int[saveDataLength * INPUT_CH_NUM];
                    }
                }
            }
            
            // Prefix
            saveFilePrefix = ci.FilePrefix;

            // Format
            saveFileFormat = ci.FileFormat;
        }

        /// <summary>
        /// BLEデバイスの接続
        /// </summary>
        private static async void connectBLE()
        {
            try
            {
                DeviceInformationCollection device = await DeviceInformation.FindAllAsync(
                    GattDeviceService.GetDeviceSelectorFromUuid(
                    new Guid(BLE_DEVICE_SERVICE_UUID)));
                if (device.Count < 1) { throw new Exception("Device not found"); }

                var service = await GattDeviceService.FromIdAsync(device.First().Id);
                var characteristics = service.GetCharacteristics(new Guid(BLECHAR_WAVE_VAL_UUID));
                if (characteristics.Count < 1) { throw new Exception("Notify service not found"); }
                characteristic_notify = characteristics.First();
                characteristic_notify.ValueChanged += WaveData_ValueChanged;

                GattCommunicationStatus status =
                    await characteristic_notify.WriteClientCharacteristicConfigurationDescriptorAsync(
                    GattClientCharacteristicConfigurationDescriptorValue.Notify);

                if (status == GattCommunicationStatus.Unreachable)
                {
                    throw new Exception("Service is not reachable");
                }

                Console.WriteLine("Connected");

                characteristics = service.GetCharacteristics(new Guid(BLECHAR_BD_CTRL_UUID));
                if (characteristics.Count < 1) { throw new Exception("Command service not found"); }
                characteristic_command = characteristics.First();

                byte[] command = exgbd.CommandBuf;
                await characteristic_command.WriteValueAsync(command.AsBuffer());

                GattReadResult result = await characteristic_command.ReadValueAsync();
                byte[] buffer = (result.Value.ToArray());
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.WriteLine("Ready");
            }
        }

        /// <summary>
        /// EXGBDへのコマンド送信
        /// </summary>
        /// <param name="command"></param>
        private static async void sendCommand(byte[] command)
        {
            if (characteristic_command != null)
            {
                await characteristic_command.WriteValueAsync(command.AsBuffer());
            }
        }

        /// <summary>
        /// 波形受信イベントハンドラ
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="args"></param>
        private static void WaveData_ValueChanged(GattCharacteristic sender, GattValueChangedEventArgs args)
        {
            var buffer = args.CharacteristicValue.ToArray();

            int length = buffer.Length / sizeof(Int16);
            Int16[] ch1Data = new Int16[EXGBD.BUF_SIZE_PER_CH];
            Int16[] ch2Data = new Int16[EXGBD.BUF_SIZE_PER_CH];

            try
            {
                for (int i = 0; i < length / 2; i++)
                {
                    ch1Data[i] = (BitConverter.ToInt16(buffer, i * 2 * sizeof(Int16)));
                    ch2Data[i] = (BitConverter.ToInt16(buffer, (i * 2 + 1) * sizeof(Int16)));
                }
            }
            catch
            {
                Console.WriteLine("Packet length: " + length);
            }

            if ((server !=null) && (server.IsConnected))
            {
                server.Write(buffer, 0, buffer.Length);
            }

            if (saveFileFlag)
            {
                if(saveDataCounter == 0)
                {
                    DateTime dt = DateTime.Now;
                    startSaveTime = getDateTimeStr(dt);
                }

                for (int i = 0; i < ch1Data.Length; i++)
                {
                    saveWaveBuf[saveBufSlotIdx][saveDataCounter * INPUT_CH_NUM] = (int)ch1Data[i];
                    saveWaveBuf[saveBufSlotIdx][saveDataCounter * INPUT_CH_NUM + 1] = (int)ch2Data[i];
                    saveDataCounter++;
                }

                if (saveDataCounter >= saveDataLength)
                {
                    int writeIndex = saveBufSlotIdx;
                    saveBufSlotIdx = (saveBufSlotIdx + 1) % SAVE_BUF_SLOT_NUM;
                    int writeCount = saveDataCounter;
                    saveDataCounter = 0;
                    flushFileTask = Task.Factory.StartNew(() => flashFile(writeIndex, writeCount));
                }
            }
            else
            {
                if (saveDataCounter > 0)
                {
                    int writeIndex = saveBufSlotIdx;
                    int writeCount = saveDataCounter;
                    flushFileTask = Task.Factory.StartNew(() => flashFile(writeIndex, writeCount));
                }
                saveBufSlotIdx = 0;
                saveDataCounter = 0;
            }
        }

        /// <summary>
        /// 波形データのファイル出力
        /// </summary>
        /// <param name="index">出力バッファインデックス</param>
        /// <param name="length">出力データ長</param>
        private static void flashFile(int index, int length)
        {
            // ファイル名の生成
            DateTime dt = DateTime.Now;
            string dtStr = getDateTimeStr(dt);
            string file_ext = (saveFileFormat == CmdInterface.FILE_FORMAT.Binary) ? FILE_BINARY_EXT : FILE_ASCII_EXT;
            string filename = saveFilePrefix + dtStr + file_ext;

            // データの保存
            if (saveFileFormat == CmdInterface.FILE_FORMAT.Binary)
            {
                using (BinaryWriter bw = new BinaryWriter(File.Open(filename, FileMode.Create)))
                {
                    int writeSize = length * INPUT_CH_NUM;
                    byte[] ba = new byte[writeSize * sizeof(int)];
                    Buffer.BlockCopy(saveWaveBuf[index], 0, ba, 0, ba.Length);
                    bw.Write(ba);
                    bw.Flush();
                }
            }
            else
            {
                using (StreamWriter sw = new StreamWriter(File.Open(filename, FileMode.Create)))
                {
                    for (int i = 0; i < length; i++)
                    {
                        string[] valStr = new string[INPUT_CH_NUM];
                        for (int j = 0; j < INPUT_CH_NUM; j++)
                        {
                            valStr[j] = saveWaveBuf[index][i * INPUT_CH_NUM + j].ToString();
                        }
                        sw.WriteLine(string.Join(ASCII_SEPARATOR, valStr));
                    }
                    sw.Flush();
                }
            }

            // 属性ファイル名の生成
            string attrFilename = saveFilePrefix + dtStr + ATTR_FILE_SUFFIX + FILE_ASCII_EXT;

            // 属性の保存
            using (StreamWriter sw = new StreamWriter(File.Open(attrFilename, FileMode.Create)))
            {
                sw.WriteLine(ATTR_START_TITLE + ASCII_SEPARATOR + startSaveTime);
                sw.WriteLine(ATTR_END_TITLE + ASCII_SEPARATOR + dtStr);
                sw.WriteLine(ATTR_GAIN1_TITLE + ASCII_SEPARATOR + exgbd.Ch1GainLabel);
                sw.WriteLine(ATTR_GAIN2_TITLE + ASCII_SEPARATOR + exgbd.Ch2GainLabel);
                sw.WriteLine(ATTR_FILT1_TITLE + ASCII_SEPARATOR + exgbd.Ch1FiltLabel);
                sw.WriteLine(ATTR_FILT2_TITLE + ASCII_SEPARATOR + exgbd.Ch2FiltLabel);
                sw.WriteLine(ATTR_FREQ_TITLE + ASCII_SEPARATOR + exgbd.FreqLabel);

                sw.Flush();
            }
        }

        /// <summary>
        /// DateTime型をファイル名用の文字列に変換する
        /// </summary>
        /// <param name="dt">変換するDateTimeオブジェクト</param>
        /// <returns>時刻文字列</returns>
        private static string getDateTimeStr(DateTime dt)
        {
            return (dt.Year.ToString()
                    + dt.Month.ToString("00")
                    + dt.Day.ToString("00") + "_"
                    + dt.Hour.ToString("00")
                    + dt.Minute.ToString("00")
                    + dt.Second.ToString("00")
                    + dt.Millisecond.ToString("000"));
        }

    }
}
