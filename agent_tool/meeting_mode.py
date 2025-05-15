import subprocess
import os
from pydantic import BaseModel

class MeetingModeType(BaseModel):
    brightness: int
    master_volume: int
    app_volume: int

def meeting_mode(brightness: int, master_volume: int, app_volume: int) -> str:
    """
    Start the Meeting Model with specified brightness and volume settings."""
    try:
        #subprocess.run(["C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Set_Brightness.exe"],check=True)

        #subprocess.run(["C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Check_Internet.exe"],check=True)
        #subprocess.run(["C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Vedio_Recording.exe"],check=True)
        '''
        brightness_exe_path = "C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Set_Brightness.exe"
        brightness_exe_dir = os.path.dirname(brightness_exe_path)
        
        adudio_volume_exe_path = "C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Set_Audio_Volume.exe"
        adudio_volume_exe_dir = os.path.dirname(adudio_volume_exe_path)
        
        subprocess.run([brightness_exe_path,"--brightness",brightness], check=True, cwd=brightness_exe_dir)
        subprocess.run([adudio_volume_exe_path,"--master-volume",master_volume,"--app-volume",app_volume], check=True, cwd=adudio_volume_exe_dir)
        '''
        command_audio = [
            "C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Set_Audio_Volume.exe",
            f"--app-volume={app_volume}",
            f"--master-volume={master_volume}"
        ]
        command_brightness = [
            "C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Set_Brightness.exe",
            f"--brightness={brightness}"
        ]
        command_battery = [
            "C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Set_Battery.exe",
            
        ]
        brightness_exe_path = "C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Set_Brightness.exe"
        brightness_exe_dir = os.path.dirname(brightness_exe_path)
        
        adudio_volume_exe_path = "C:\\Users\\matt\\Desktop\\Agentic_AI_Tool\\dist\\Set_Audio_Volume.exe"
        adudio_volume_exe_dir = os.path.dirname(adudio_volume_exe_path)
        
        #subprocess.run([brightness_exe_path,"--brightness",brightness], check=True, cwd=brightness_exe_dir)
        #subprocess.run([adudio_volume_exe_path,"--master-volume",master_volume,"--app-volume",app_volume], check=True, cwd=adudio_volume_exe_dir)
        subprocess.run(command_brightness, check=True, text=True,capture_output=True)
        subprocess.run(command_audio, check=True, text=True,capture_output=True)
        subprocess.run(command_battery, check=True, text=True,capture_output=True)
        return f"✅ Start Meeting Model successfully. the brightness volume is {brightness}, the master volume is {master_volume}, the app volume is {app_volume}. The battery is in save mode now."
    except subprocess.CalledProcessError as e:
        return f"❌ Failed to start Meeting Model: {str(e)}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}"