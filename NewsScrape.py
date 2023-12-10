import requests


# Set your News API key
env = '9ba9a80fa8a74e00b67d7ddbf2411054'
def NewsScrape(query):
    try:
        # Make a request to the News API
        url = f'https://newsapi.org/v2/everything?q={query}&apiKey={env}'
        response = requests.get(url)
        news_data = response.json()

        # Parse and return the news results
        articles = news_data.get('articles', [])
        results = []
        for article in articles:
            item = {
                'title': article.get('title', ''),
                'link': article.get('url', ''),
                'text': article.get('description', '')
            }
            results.append(item)

        return results
    except Exception as e:
        print(e)
        return None
