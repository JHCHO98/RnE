from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

options = Options()
options.add_argument("user-data-dir=C:/Users/Owner/Desktop/selenium_profile")  # 사용자 프로필 경로
options.add_argument("profile-directory=selenium_profile")  # 정확히 로그인된 프로필 이름
options.add_argument("https://www.youtube.com")  # 실행 직후 이동할 URL

driver = webdriver.Chrome(options=options)
driver.get("https://www.youtube.com")

time.sleep(5)  # 충분히 로딩 대기

html = driver.page_source
with open("main.html", "w", encoding="utf-8") as file:
    file.write(html)

driver.quit()
