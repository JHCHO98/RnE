from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

# 크롬 드라이버 실행
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 유튜브 검색
keyword = "윤석열"
driver.get(f"https://www.youtube.com/results?search_query={keyword}")
time.sleep(3)

# 스크롤 조금만 (영상 수 제한)
for _ in range(2):
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(2)

# 제목, 채널, 링크 가져오기
titles = driver.find_elements(By.ID, 'video-title')
channels = driver.find_elements(By.CSS_SELECTOR, "#channel-info #text")

video_info_list = []

for title, channel in zip(titles, channels):
    title_text = title.get_attribute('title')
    channel_name = channel.text
    link = title.get_attribute('href')

    if title_text and link:
        video_info_list.append({
            "title": title_text,
            "channel": channel_name,
            "link": link
        })

# 각 영상에서 댓글 수집
for video in video_info_list:
    driver.get(video["link"])
    time.sleep(5)

    # 댓글 로딩 위해 스크롤
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(3)

    comments_text = []
    try:
        comments = driver.find_elements(By.CSS_SELECTOR, "#content-text")
        for c in comments[:3]:
            comments_text.append(c.text)
    except:
        comments_text = ["(댓글 로딩 실패)"]

    # 부족한 댓글 수 채우기
    while len(comments_text) < 3:
        comments_text.append("")

    # 출력
    print(f"제목: {video['title']}")
    print(f"채널: {video['channel']}")
    print(f"링크: {video['link']}")
    print(f"댓글1: {comments_text[0]}")
    print(f"댓글2: {comments_text[1]}")
    print(f"댓글3: {comments_text[2]}")
    print("-" * 50)

driver.quit()
