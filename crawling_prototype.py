from bs4 import BeautifulSoup

with open("main.html", "r", encoding="utf-8") as file:
    html = file.read()

soup = BeautifulSoup(html, "html.parser")

shorts = []

# Shorts 기준 탐색
short_items = soup.select("ytd-rich-item-renderer.style-scope.ytd-rich-shelf-renderer")
for item in short_items:
    # 제목
    title_tag = item.select_one(".shortsLockupViewModelHostMetadataTitle a")
    if title_tag:
        title = title_tag.get_text(strip=True)
        href = title_tag["href"]
        full_url = f"https://www.youtube.com{href}"
        shorts.append({
            "type": "shorts",
            "title": title,
            "url": full_url
        })

# 예시 출력
for s in shorts:
    print(f"[SHORTS] {s['title']} → {s['url']}")

videos = []
shorts = []

for item in soup.select("ytd-rich-item-renderer"):
    if item.get("class") and "ytd-rich-shelf-renderer" in item["class"]:
        
        # 쇼츠
        title_tag = item.select_one(".shortsLockupViewModelHostMetadataTitle a")
        
        if title_tag:
            
            title = title_tag.get_text(strip=True)
            href = title_tag.get("href")
            
            if href:
                shorts.append({
                    "type": "shorts",
                    "title": title,
                    "url": f"https://www.youtube.com{href}"
                })
            


    else:
        # 일반 영상
        title_tag = item.select_one("#video-title")
        channel_tag = item.select_one("ytd-channel-name")
        if not channel_tag:
            continue  # 광고일 가능성 높음

        title_link = item.select_one("a#video-title-link")
        title_tag = item.select_one("yt-formatted-string#video-title")

        if title_link and title_link.has_attr("href") and title_tag:
            url = "https://www.youtube.com" + title_link["href"]
            title = title_tag.get_text(strip=True)
            channel = channel_tag.get_text(strip=True)

            videos.append({
                "title": title,
                "url": url,
                "channel": channel
            })

for v in videos:
    print(f"[영상] {v['title']} ({v['channel']}) → {v['url']}")
for s in shorts:
    print(f"[쇼츠] {s['title']} → {s['url']}")
