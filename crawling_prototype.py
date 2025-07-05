from bs4 import BeautifulSoup
from YoutubeAPI import YoutubeAPI


with open("main.html", "r", encoding="utf-8") as file:
    html = file.read()

soup = BeautifulSoup(html, "html.parser")


videos = []
shorts = []

results=[]

for item in soup.select("ytd-rich-item-renderer"):
    try:
        if item.get("class") and "ytd-rich-shelf-renderer" in item["class"]:
            
            # 쇼츠
            title_tag = item.select_one(".shortsLockupViewModelHostMetadataTitle a")
            
            if title_tag:
                
                title = title_tag.get_text(strip=True)
                href = title_tag.get("href")[8:] 

                api = YoutubeAPI(href)
                comments = api.get_comments()

                if href:
                    results.append({
                        "type": "shorts",
                        "title": title,
                        "url": href,
                        "channel": api.get_channel_name(),
                        'comments': comments
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
                url = title_link["href"][9:]

                #/watch?v=bSg03t5F2sk
                title = title_tag.get_text(strip=True)
                channel = channel_tag.get_text(strip=True)
                channel_=channel
                if channel[-3:]=='인증됨':
                    channel_= channel[:-3]
                if channel[:len(channel_)//2] == channel_[len(channel_)//2:]:
                    channel = channel[:len(channel_)//2]


                api = YoutubeAPI(url)
                comments = api.get_comments()
                results.append({
                    "type": "video",
                    "title": title,
                    "url": url,
                    "channel": channel,
                    'comments': comments
                })
    except Exception as e:
        continue
import json
with open("video.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"총 {len(results)}개 항목이 저장되었습니다.")
