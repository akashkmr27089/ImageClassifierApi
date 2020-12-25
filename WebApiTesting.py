import requests

resp = requests.get('https://jsonplaceholder.typicode.com/todos/1')
if resp.status_code != 200:
    # This means something went wrong.
    raise ApiError('GET /tasks/ {}'.format(resp.status_code))
print(resp.json())
data = resp.json()
print('User Id : {} \nId : {} \nTitle : {} \nCompleted : {}'.format(data['userId'], data['id'], data['title'], data['completed']))