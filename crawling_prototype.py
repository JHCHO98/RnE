from bs4 import BeautifulSoup

def extract_video_info_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

    # 각 영상 하나는 ytd-rich-item-renderer
    video_blocks = soup.find_all('ytd-rich-item-renderer')

    for block in video_blocks:
        # 영상 제목 추출
        title_tag = block.find('yt-formatted-string', {'id': 'video-title'})
        title = title_tag.text.strip() if title_tag else '제목 없음'

        # 채널 이름 추출
        channel_tag = block.find('ytd-channel-name')
        channel_link = channel_tag.find('a') if channel_tag else None
        channel = channel_link.text.strip() if channel_link else '채널명 없음'

        print(f'제목: {title}')
        print(f'채널: {channel}')
        print('-' * 40)

# 예시 실행
extract_video_info_from_html('main.html')
