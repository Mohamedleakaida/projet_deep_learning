import urllib.request, json

url = 'http://localhost:8000/analyze'
boundary = '----FormBoundary'

with open('images classification/enh_Te-glTr_0005.jpg', 'rb') as f:
    file_data = f.read()

part_header = (
    '--' + boundary + '\r\n'
    'Content-Disposition: form-data; name="file"; filename="enh_Te-glTr_0005.jpg"\r\n'
    'Content-Type: image/jpeg\r\n\r\n'
).encode()
body = part_header + file_data + ('\r\n--' + boundary + '--\r\n').encode()

req = urllib.request.Request(
    url, data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}
)
r = urllib.request.urlopen(req)
d = json.loads(r.read())
print('HTTP:', r.status)
print('Predicted:', d['predicted_class'], d['confidence'], '%')
print('Is normal:', d['is_normal'])
print('Overlay:', d['overlay_image'] is not None)
print('Scores:', [(s['class'], s['probability']) for s in d['scores'][:3]])
