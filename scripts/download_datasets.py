import os
import gdown
import zipfile
import tarfile


print('Datasets will be downloaded and extracted to ./data/DAVIS, ./data/MOSE')


"""
DAVIS dataset
"""
os.makedirs('./data/DAVIS/2017', exist_ok=True)

print('Downloading DAVIS 2017 trainval...')
gdown.download('https://drive.google.com/uc?id=1kiaxrX_4GuW6NmiVuKGSGVoKGWjOdp6d',
               output='./data/DAVIS/2017/DAVIS-2017-trainval-480p.zip',
               quiet=False)

print('Downloading DAVIS 2017 testdev...')
gdown.download('https://drive.google.com/uc?id=1fmkxU2v9cQwyb62Tj1xFDdh2p4kDsUzD',
               output='./data/DAVIS/2017/DAVIS-2017-test-dev-480p.zip',
               quiet=False)

print('Extracting DAVIS datasets...')
with zipfile.ZipFile('./data/DAVIS/2017/DAVIS-2017-trainval-480p.zip', 'r') as zip_file:
    zip_file.extractall('./data/DAVIS/2017/')
os.rename('./data/DAVIS/2017/DAVIS', './data/DAVIS/2017/trainval')

with zipfile.ZipFile('./data/DAVIS/2017/DAVIS-2017-test-dev-480p.zip', 'r') as zip_file:
    zip_file.extractall('./data/DAVIS/2017/')
os.rename('./data/DAVIS/2017/DAVIS', './data/DAVIS/2017/test-dev')


os.remove('./data/DAVIS/2017/DAVIS-2017-trainval-480p.zip')
os.remove('./data/DAVIS/2017/DAVIS-2017-test-dev-480p.zip')


"""
MOSE dataset
"""
print('Downloading MOSE valid ...')

os.makedirs('./data/MOSE', exist_ok=True)
gdown.download('https://drive.google.com/uc?id=1yFoacQ0i3J5q6LmnTVVNTTgGocuPB_hR', output='./data/MOSE/valid.tar.gz', quiet=False)

print('Extracting MOSE valid...')
with tarfile.open(os.path.join('./data/MOSE', 'valid.tar.gz'), 'r') as tfile:
    tfile.extractall('./data/MOSE/')

os.remove('./data/MOSE/valid.tar.gz')

print('Done.')