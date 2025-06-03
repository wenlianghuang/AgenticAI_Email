import ctypes
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import argparse
# 設定音量常數
#SPEAKER_VOLUME = 0.10  # 系統喇叭音量設為30%

def set_master_volume(speaker_volume):
    
    """將系統主喇叭音量設定為30%"""
    try:
        speaker_volume = float(speaker_volume)
        speaker_volume = round(speaker_volume / 100, 2)

        # 獲取喇叭設備
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))
        
        # 獲取當前音量
        current_volume = volume.GetMasterVolumeLevelScalar()
        
        # 設置喇叭音量為30%
        #volume.SetMasterVolumeLevelScalar(SPEAKER_VOLUME, None)
        volume.SetMasterVolumeLevelScalar(speaker_volume, None)
        #print(f"已將系統喇叭音量從 {int(current_volume*100)}% 調整為 {int(speaker_volume*100)}%")
        return f"✅ The system speaker volume is set to {int(speaker_volume*100)}%."
        #return True
    except Exception as e:
        print(f"設置系統喇叭音量時發生錯誤: {e}")
        #return False
