
class YoutubeAPI:
    def __init__(self,video_id,DEVELOPER_KEY='AIzaSyAYolk9AFe-eAlIUH3DF2b27mJKVn7OdvM'):

        self.video_id = video_id
        self.DEVELOPER_KEY = DEVELOPER_KEY

    def get_comments(self):
        
        from googleapiclient.discovery import build
        
        video_id = self.video_id
        DEVELOPER_KEY = self.DEVELOPER_KEY

        YOUTUBE_API_SERVICE_NAME = "youtube"
        YOUTUBE_API_VERSION = "v3"
        
        youtube= build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
        request = youtube.commentThreads().list(maxResults=5,order='relevance', part="snippet,replies", videoId=video_id, textFormat="plainText", )

        import pandas as pd

        df = pd.DataFrame(columns=["comment", "replies", "user_name", "date"])

        comments = []

        try:
            response = request.execute()

            comments = []

            for item in response["items"]:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = snippet["textDisplay"]

                comments.append(text)
            
            return comments


        except Exception as e:
            raise e
    
    def get_channel_name(self):
        from googleapiclient.discovery import build
        
        video_id = self.video_id
        DEVELOPER_KEY = self.DEVELOPER_KEY

        YOUTUBE_API_SERVICE_NAME = "youtube"
        YOUTUBE_API_VERSION = "v3"
        try:
            youtube= build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
            request = youtube.videos().list(part="snippet", id=video_id)

            response = request.execute()
            
            if response["items"]:
                channel_name = response["items"][0]["snippet"]["channelTitle"]
                return channel_name
            else:
                return None
        except Exception as e:
            raise e


if __name__ == "__main__":
    api= YoutubeAPI(video_id="OoPxf0pQrFg")

    comments = api.get_comments()
    channel_name = api.get_channel_name()

    for comment in comments:
        print(comment)
