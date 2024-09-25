from utils import DictToClass, make_settings


d = DictToClass()
d.teste = 2
print(d)
d = make_settings(['curl_air'])
# d.save_as_json('settings.json')

print(d.Cameras.keys)