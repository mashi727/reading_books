using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ExgBdCtrl
{
    class CmdInterface
    {
        public enum CMD_TYPE { Set, Req, File, Unknown }

        public enum CMD_CODE { Acq, Gain, Filt, Freq, Led, Save, DevName, ConnStatus, Quit, Unknown }

        public enum FILE_FORMAT { Ascii, Binary, Unknown }

        // リクエスト応答文字列
        public const string CMD_NOT_SUPPORTED = "Command not supported";
        public const string CONN_STATUS_CONNECTED = "Coonected";
        public const string CONN_STATUS_UNCONNECTED = "Unconnected";
        public const string CMD_QUIT_ACQ = "Quit application...";

        private const string COMMAND_SET_STR = "SET";
        private const string COMMAND_REQ_STR = "REQ";
        private const string COMMAND_FILE_STR = "FILE";

        private const int NUM_ELEMENTS_SET_COMMAND = 4;     // SET XXXX YYYY ZZZZ
        private const int NUM_ELEMENTS_REQ_COMMAND = 2;     // REQ XXXX
        private const int NUM_ELEMENTS_FILE_COMMAND = 4;    // FILE LENGTH PREFIX FORMAT

        private const int CMD_TYPE_INDEX = 0;
        private const int CMD_CODE_INDEX = 1;

        private const int CMD_SET_NUM_ARGS = 2;
        private const int CMD_SET_ARG1_INDEX = 2;
        private const int CMD_SET_ARG2_INDEX = 3;
        private const int CMD_SET_SINGLE_ARG_INDEX = 1;

        private const int CMD_FILE_LENGTH_INDEX = 1;
        private const int CMD_FILE_PREFIX_INDEX = 2;
        private const int CMD_FILE_FORMAT_INDEX = 3;

        private static Dictionary<string, CMD_CODE> CODE_SETCMD = new Dictionary<string, CMD_CODE>()
            {
                {"ACQ_", CMD_CODE.Acq},
                {"GAIN", CMD_CODE.Gain},
                {"FILT", CMD_CODE.Filt},
                {"SMPL", CMD_CODE.Freq},
                {"LED_", CMD_CODE.Led},
                {"SAVE", CMD_CODE.Save}
            };

        private static Dictionary<string, CMD_CODE> CODE_REQCMD = new Dictionary<string, CMD_CODE>()
            {
                {"DEVN", CMD_CODE.DevName},
                {"CONN", CMD_CODE.ConnStatus},
                {"QUIT", CMD_CODE.Quit},
                {"ACQ_", CMD_CODE.Acq},
                {"GAIN", CMD_CODE.Gain},
                {"FILT", CMD_CODE.Filt},
                {"SMPL", CMD_CODE.Freq},
                {"LED_", CMD_CODE.Led},
                {"SAVE", CMD_CODE.Save}
            };

        private static Dictionary<string, bool> ARG_ACQ = new Dictionary<string, bool>()
            {
                {"0000", false},
                {"0001", true}
            };

        private static Dictionary<string, EXGBD.GAIN> ARG_GAIN = new Dictionary<string, EXGBD.GAIN>()
            {
                {"0200", EXGBD.GAIN.Low},
                {"1000", EXGBD.GAIN.High}
            };

        private static Dictionary<string, EXGBD.FILT> ARG_FILT = new Dictionary<string, EXGBD.FILT>()
            {
                {"0030", EXGBD.FILT.Low},
                {"0100", EXGBD.FILT.High}
            };

        private static Dictionary<string, EXGBD.FREQ> ARG_FREQ = new Dictionary<string, EXGBD.FREQ>()
            {
                {"0100", EXGBD.FREQ.F100Hz},
                {"0200", EXGBD.FREQ.F200Hz},
                {"0500", EXGBD.FREQ.F500Hz},
                {"1000", EXGBD.FREQ.F1000Hz}
            };

        private static Dictionary<string, EXGBD.LED> ARG_LED = new Dictionary<string, EXGBD.LED>()
            {
                {"0000", EXGBD.LED.Off},
                {"0001", EXGBD.LED.On}
            };

        private static Dictionary<string, bool> ARG_SAVE = new Dictionary<string, bool>()
            {
                {"0000", false},
                {"0001", true}
            };

        private static Dictionary<string, FILE_FORMAT> ARG_FILEFORMAT = new Dictionary<string, FILE_FORMAT>()
            {
                {"ASCII", FILE_FORMAT.Ascii},
                {"BINARY", FILE_FORMAT.Binary}
            };

        // プロパティ
        public CMD_TYPE CmdType { get; private set; }
        public CMD_CODE CmdCode { get; private set; }
        public object CmdArg { get; private set; }
        public int FileDataLength { get; private set; }
        public string FilePrefix { get; private set; }
        public FILE_FORMAT FileFormat { get; private set; }


        public CmdInterface(string cmd)
        {
            // Initialize
            CmdType = CMD_TYPE.Unknown;
            CmdCode = CMD_CODE.Unknown;
            CmdArg = null;
            FileDataLength = 0;
            FilePrefix = null;
            FileFormat = FILE_FORMAT.Unknown;

            // Interpret command
            string[] cmdWords = cmd.Split(new char[] { ' ' });
            if (cmdWords.Length > 0)
            {
                switch (cmdWords[CMD_TYPE_INDEX])
                {
                    case COMMAND_SET_STR:
                        CmdType = CMD_TYPE.Set;
                        interpretSetCmd(cmdWords);
                        break;
                    case COMMAND_REQ_STR:
                        CmdType = CMD_TYPE.Req;
                        interpretReqCmd(cmdWords);
                        break;
                    case COMMAND_FILE_STR:
                        CmdType = CMD_TYPE.File;
                        interpretFileCmd(cmdWords);
                        break;
                    default:
                        break;
                }
            }
        }

        private void interpretSetCmd(string[] cmdWords)
        {
            if (cmdWords.Length >= NUM_ELEMENTS_SET_COMMAND)
            {
                foreach (string code in CODE_SETCMD.Keys)
                {
                    if (cmdWords[CMD_CODE_INDEX] == code)
                    {
                        CmdCode = CODE_SETCMD[code];
                        break;
                    }
                }

                string[] args = new string[CMD_SET_NUM_ARGS];
                args[0] = cmdWords[CMD_SET_ARG1_INDEX];
                args[1] = cmdWords[CMD_SET_ARG2_INDEX];

                switch (CmdCode)
                {
                    case CMD_CODE.Acq:
                        CmdArg = getArgForSetAcqCmd(args);
                        break;
                    case CMD_CODE.Gain:
                        CmdArg = getArgForSetGainCmd(args);
                        break;
                    case CMD_CODE.Filt:
                        CmdArg = getArgForSetFiltCmd(args);
                        break;
                    case CMD_CODE.Freq:
                        CmdArg = getArgForSetFreqCmd(args);
                        break;
                    case CMD_CODE.Led:
                        CmdArg = getArgForSetLedCmd(args);
                        break;
                    case CMD_CODE.Save:
                        CmdArg = getArgForSetSaveCmd(args);
                        break;
                    default:
                        break;
                }
            }
        }

        private void interpretReqCmd(string[] cmdWords)
        {
            if (cmdWords.Length >= NUM_ELEMENTS_REQ_COMMAND)
            {
                foreach (string code in CODE_REQCMD.Keys)
                {
                    if (cmdWords[CMD_CODE_INDEX] == code)
                    {
                        CmdCode = CODE_REQCMD[code];
                        break;
                    }
                }
            }
        }

        private void interpretFileCmd(string[] cmdWords)
        {
            if (cmdWords.Length >= NUM_ELEMENTS_FILE_COMMAND)
            {
                int dataLength = 0;
                if(int.TryParse(cmdWords[CMD_FILE_LENGTH_INDEX], out dataLength))
                {
                    FileDataLength = dataLength;
                }

                FilePrefix = cmdWords[CMD_FILE_PREFIX_INDEX];

                if (ARG_FILEFORMAT.ContainsKey(cmdWords[CMD_FILE_FORMAT_INDEX]))
                {
                    FileFormat = ARG_FILEFORMAT[cmdWords[CMD_FILE_FORMAT_INDEX]];
                }
                else
                {
                    FileFormat = FILE_FORMAT.Unknown;
                }
            }
        }

        private Nullable<bool> getArgForSetAcqCmd(string[] args)
        {
            if (ARG_ACQ.ContainsKey(args[CMD_SET_SINGLE_ARG_INDEX]))
            {
                return ARG_ACQ[args[CMD_SET_SINGLE_ARG_INDEX]];
            }
            else
            {
                return null;
            }
        }

        private EXGBD.GAIN[] getArgForSetGainCmd(string[] args)
        {
            EXGBD.GAIN[] gains = new EXGBD.GAIN[args.Length];
            for (int i = 0; i < args.Length; i++)
            {
                if (ARG_GAIN.ContainsKey(args[i]))
                {
                    gains[i] = ARG_GAIN[args[i]];
                }
                else
                {
                    return null;
                }
            }
            return gains;
        }

        private EXGBD.FILT[] getArgForSetFiltCmd(string[] args)
        {
            EXGBD.FILT[] filts = new EXGBD.FILT[args.Length];
            for (int i = 0; i < args.Length; i++)
            {
                if (ARG_FILT.ContainsKey(args[i]))
                {
                    filts[i] = ARG_FILT[args[i]];
                }
                else
                {
                    return null;
                }
            }
            return filts;
        }

        private Nullable<EXGBD.FREQ> getArgForSetFreqCmd(string[] args)
        {
            if (ARG_FREQ.ContainsKey(args[CMD_SET_SINGLE_ARG_INDEX]))
            {
                return ARG_FREQ[args[CMD_SET_SINGLE_ARG_INDEX]];
            }
            else
            {
                return null;
            }
        }

        private Nullable<EXGBD.LED> getArgForSetLedCmd(string[] args)
        {
            if (ARG_LED.ContainsKey(args[CMD_SET_SINGLE_ARG_INDEX]))
            {
                return ARG_LED[args[CMD_SET_SINGLE_ARG_INDEX]];
            }
            else
            {
                return null;
            }
        }

        private Nullable<bool> getArgForSetSaveCmd(string[] args)
        {
            if (ARG_SAVE.ContainsKey(args[CMD_SET_SINGLE_ARG_INDEX]))
            {
                return ARG_SAVE[args[CMD_SET_SINGLE_ARG_INDEX]];
            }
            else
            {
                return null;
            }
        }



        public static string GetStrForReqAcqCmd(bool isAcq)
        {
            return ARG_ACQ.First(x => x.Value == isAcq).Key;
        }

        public static string GetStrForReqGainCmd(EXGBD.GAIN[] gains)
        {
            string[] gain_strs = new string[gains.Length];
            for(int i=0; i<gains.Length; i++)
            {
                gain_strs[i] = ARG_GAIN.First(x => x.Value == gains[i]).Key;
            }
            return string.Join(" ", gain_strs);
        }

        public static string GetStrForReqFiltCmd(EXGBD.FILT[] filts)
        {
            string[] filt_strs = new string[filts.Length];
            for (int i = 0; i < filts.Length; i++)
            {
                filt_strs[i] = ARG_FILT.First(x => x.Value == filts[i]).Key;
            }
            return string.Join(" ", filt_strs);
        }

        public static string GetStrForReqFreqCmd(EXGBD.FREQ freq)
        {
            return ARG_FREQ.First(x => x.Value == freq).Key;
        }

        public static string GetStrForReqLedCmd(EXGBD.LED led)
        {
            return ARG_LED.First(x => x.Value == led).Key;
        }

        public static string GetStrForReqSaveCmd(bool isSave)
        {
            return ARG_SAVE.First(x => x.Value == isSave).Key;
        }
    }
}
