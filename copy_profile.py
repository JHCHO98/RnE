import shutil
import os

# 사용자 프로필 경로 복사
original_profile = r"C:/Users/Owner/AppData/Local/Google/Chrome/User Data/Profile 2"
custom_profile = r"C:/Users/Owner/Desktop/selenium_profile"

# 프로필 복사는 최초 1회만 하면 됨
if not os.path.exists(custom_profile):
    shutil.copytree(original_profile, custom_profile, dirs_exist_ok=True)
